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
#define TMEM_NCOLS 256
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
#define SMEM_QUEUE_OFF 207616
#define SMEM_QUEUE_STAGE_BYTES 4
#define SMEM_QUEUE_STRIDE 4
#define SMEM_WEIGHT_WORDS_OFF 207872
#define SMEM_WEIGHT_WORDS_STAGE_BYTES 8192
#define SMEM_WEIGHT_WORDS_STRIDE 8192
#define SMEM_EXHAUSTED_OFF 207680
#define SMEM_EXHAUSTED_STAGE_BYTES 64
#define SMEM_EXHAUSTED_STRIDE 64
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



__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_mega_mhc_12e5bcc41e7940e7d511(const __grid_constant__ CUtensorMap residual_map, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap fnp_map, const __grid_constant__ CUtensorMap post_map, const __grid_constant__ CUtensorMap comb_map, const __grid_constant__ CUtensorMap new_residual_map, float* __restrict__ mix_scales, float* __restrict__ mix_bases, float* __restrict__ new_post_mix, float* __restrict__ new_comb_res_mix, __nv_bfloat16* __restrict__ rmsnorm_weight, __nv_bfloat16* __restrict__ new_residual, __nv_bfloat16* __restrict__ y_bf16, float* __restrict__ scratch, unsigned long long* __restrict__ split_barriers, unsigned long long* __restrict__ launch_epochs, unsigned int num_tokens, unsigned int num_splits, CakeFastDivmod split_div, unsigned int num_ctas, float hc_norm_eps, float hc_pre_eps, float hc_post_scale, float sinkhorn_eps, unsigned int num_sinkhorn_iters, float rmsnorm_eps, float rmsnorm_scale, uint8_t* __restrict__ y_fp4, uint8_t* __restrict__ y_fp4_sf, float* __restrict__ fp4_global_scale)
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
    unsigned int* queue = reinterpret_cast<unsigned int*>(smem_raw + 207616);
    const int queue_addr = smem + 207616;
    unsigned int* weight_words = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int weight_words_addr = smem + 207872;
    unsigned int* exhausted = reinterpret_cast<unsigned int*>(smem_raw + 207680);
    const int exhausted_addr = smem + 207680;
    unsigned int lane_in_warp = threadIdx.x % 32;
    uint32_t _fast_div_q_0 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(64), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(64);
    uint32_t _fast_div_r_0 = (uint32_t)(64) - _fast_div_q_0 * (uint32_t)(split_div.divisor);
    if (warp == 0) {
        if (elect_sync()) {
            queue[0] = 0;
            exhausted[0] = 0;
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

    // TMEM alloc (256 columns, 224 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216360);
    if (warp == 3) {
        int _tmem_hold = smem + 216360;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
                        int _mma_b_lo_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_0, 0x40004040, 67503248, ((kb_1 == 0) ? 0 : 1));
                        int _mma_b_lo_48_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_48_0, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_49_0, 0x40004040, 67503248, 0);
                        int _mma_b_lo_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_1, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_48_1, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_49_1, 0x40004040, 67503248, 1);
                        int _mma_b_lo_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_2, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_48_2, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_49_2, 0x40004040, 67503248, 1);
                        int _mma_b_lo_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_3, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_48_3, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_49_3, 0x40004040, 67503248, 1);
                        elect_commit(empty_a_addr + (astage_1_0) * 8);
                        aseq += 1;
                        unsigned int astage_1_1 = aseq % 4;
                        unsigned int _stage_full_a_1 = aseq % 4;
                        unsigned int _parity_full_a_1 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_1) * 8, _parity_full_a_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_4, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_48_4, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_49_4, 0x40004040, 67503248, 1);
                        int _mma_b_lo_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_48_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_49_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_6, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_48_6, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_49_6, 0x40004040, 67503248, 1);
                        int _mma_b_lo_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_7, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_48_7, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_49_7, 0x40004040, 67503248, 1);
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
                        int _mma_b_lo_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_8, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_48_8, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_49_8, 0x40004040, 67503248, 0);
                        int _mma_b_lo_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_9, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_48_9, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_49_9, 0x40004040, 67503248, 1);
                        int _mma_b_lo_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_10, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_48_10, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_49_10, 0x40004040, 67503248, 1);
                        int _mma_b_lo_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_11, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_48_11, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_49_11, 0x40004040, 67503248, 1);
                        elect_commit(empty_a_addr + (astage_1_2) * 8);
                        aseq += 1;
                        unsigned int astage_1_3 = aseq % 4;
                        unsigned int _stage_full_a_3 = aseq % 4;
                        unsigned int _parity_full_a_3 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_3) * 8, _parity_full_a_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_12, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_48_12, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_49_12, 0x40004040, 67503248, 1);
                        int _mma_b_lo_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_13, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_48_13, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_49_13, 0x40004040, 67503248, 1);
                        int _mma_b_lo_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_14, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_48_14, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_49_14, 0x40004040, 67503248, 1);
                        int _mma_b_lo_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_15, 0x40004040, 67503248, 1);
                        int _mma_b_lo_48_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_48_15, 0x40004040, 67503248, 1);
                        int _mma_b_lo_49_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_49_15, 0x40004040, 67503248, 1);
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
                    float tmem_load[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7]))
                        : "r"((unsigned int)tmem_acc_big + cstage * 24));
                    float tmem_load_1[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3]))
                        : "r"((unsigned int)tmem_acc_big + (cstage * 24 + 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(big_empty_addr + (cstage) * 8);
                    }
                    float sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8) : "f"(tmem_load[0]), "f"(comp[0]));
                    float add_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(acc[0]), "f"(sub_8));
                    float _sub_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(add_8), "f"(acc[0]));
                    float _sub_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_sub_1), "f"(sub_8));
                    comp[0] = _sub_0;
                    acc[0] = add_8;
                    float sub_8_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8_1) : "f"(tmem_load[1]), "f"(comp[1]));
                    float add_8_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8_1) : "f"(acc[1]), "f"(sub_8_1));
                    float _sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(add_8_1), "f"(acc[1]));
                    float _sub_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_sub_3), "f"(sub_8_1));
                    comp[1] = _sub_2;
                    acc[1] = add_8_1;
                    float sub_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_9) : "f"(tmem_load[2]), "f"(comp[2]));
                    float add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(acc[2]), "f"(sub_9));
                    float _sub_5_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_1) : "f"(add_9), "f"(acc[2]));
                    float _sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(_sub_5_1), "f"(sub_9));
                    comp[2] = _sub_4;
                    acc[2] = add_9;
                    float sub_10;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_10) : "f"(tmem_load[3]), "f"(comp[3]));
                    float add_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(acc[3]), "f"(sub_10));
                    float _sub_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(add_10), "f"(acc[3]));
                    float _sub_6_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_1) : "f"(_sub_7), "f"(sub_10));
                    comp[3] = _sub_6_1;
                    acc[3] = add_10;
                    float sub_11;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_11) : "f"(tmem_load[4]), "f"(comp[4]));
                    float add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(acc[4]), "f"(sub_11));
                    float _sub_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(add_11), "f"(acc[4]));
                    float _sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(_sub_9), "f"(sub_11));
                    comp[4] = _sub_8;
                    acc[4] = add_11;
                    float sub_12;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_12) : "f"(tmem_load[5]), "f"(comp[5]));
                    float add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(acc[5]), "f"(sub_12));
                    float _sub_11;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(add_12), "f"(acc[5]));
                    float _sub_10;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(_sub_11), "f"(sub_12));
                    comp[5] = _sub_10;
                    acc[5] = add_12;
                    float sub_13;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_13) : "f"(tmem_load[6]), "f"(comp[6]));
                    float add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(acc[6]), "f"(sub_13));
                    float _sub_13_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_1) : "f"(add_13), "f"(acc[6]));
                    float _sub_12_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_1) : "f"(_sub_13_1), "f"(sub_13));
                    comp[6] = _sub_12_1;
                    acc[6] = add_13;
                    float sub_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_14) : "f"(tmem_load[7]), "f"(comp[7]));
                    float add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(acc[7]), "f"(sub_14));
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(add_14), "f"(acc[7]));
                    float _sub_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(_sub_15), "f"(sub_14));
                    comp[7] = _sub_14;
                    acc[7] = add_14;
                    float sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_15) : "f"(tmem_load_1[0]), "f"(comp[8]));
                    float add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(acc[8]), "f"(sub_15));
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(add_15), "f"(acc[8]));
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(_sub_17), "f"(sub_15));
                    comp[8] = _sub_16;
                    acc[8] = add_15;
                    float sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_16) : "f"(tmem_load_1[1]), "f"(comp[9]));
                    float add_16;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(acc[9]), "f"(sub_16));
                    float _sub_19_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_1) : "f"(add_16), "f"(acc[9]));
                    float _sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(_sub_19_1), "f"(sub_16));
                    comp[9] = _sub_18;
                    acc[9] = add_16;
                    float sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_17) : "f"(tmem_load_1[2]), "f"(comp[10]));
                    float add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(acc[10]), "f"(sub_17));
                    float _sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(add_17), "f"(acc[10]));
                    float _sub_20_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_1) : "f"(_sub_21), "f"(sub_17));
                    comp[10] = _sub_20_1;
                    acc[10] = add_17;
                    float sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_18) : "f"(tmem_load_1[3]), "f"(comp[11]));
                    float add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(acc[11]), "f"(sub_18));
                    float _sub_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(add_18), "f"(acc[11]));
                    float _sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(_sub_23), "f"(sub_18));
                    comp[11] = _sub_22;
                    acc[11] = add_18;
                    cseq_1 += 1;
                }
                unsigned int ustage_1 = seq_1 % 2;
                unsigned int _stage_small_full = seq_1 % 2;
                unsigned int _parity_small_full = seq_1 / 2 % 2;
                mbarrier_wait(small_full_addr + (_stage_small_full) * 8, _parity_small_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float tmem_load_2[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7]))
                    : "r"((unsigned int)tmem_acc_small + ustage_1 * 24));
                float tmem_load_1_1[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[3]))
                    : "r"((unsigned int)tmem_acc_small + (ustage_1 * 24 + 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (elect_sync()) {
                    mbarrier_arrive(small_empty_addr + (ustage_1) * 8);
                }
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
                    float _add_48;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(hc_sums[row_0_1]), "f"(hc_sums[64 + row_0_1]));
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)task_4 * 64 + (unsigned long long)row_0_1] = _add_48;
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
            unsigned int first_3 = (((num_tokens + 63) / 64 * num_splits <= num_ctas) ? (unsigned int)bid + (unsigned int)(warp - 16) * num_ctas : first_0);
            #pragma unroll 1
            for (unsigned int token = first_3; token < stop_2; token += step_1) {
                unsigned int mb_1 = token / 64;
                float sc = ((lane_in_warp < 4) ? mix_scales[0] : ((lane_in_warp < 8) ? mix_scales[1] : mix_scales[2]));
                float bs = mix_bases[((lane_in_warp < 24) ? lane_in_warp : 23)];
                unsigned long long target = epoch_base + (unsigned long long)num_splits;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_1) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target)) >= static_cast<unsigned long long>(1));
                }
                float s = 0.0f;
                float q = 0.0f;
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
                    }
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
                    }
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
                    float _vec_load_12[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_1 + 0);
                        _vec_load_12[0 + 0] = _v4.x;
                        _vec_load_12[0 + 1] = _v4.y;
                        _vec_load_12[0 + 2] = _v4.z;
                        _vec_load_12[0 + 3] = _v4.w;
                    }
                    float value_23_48 = ((split_1 < num_splits) ? _vec_load_12[0] : 0.0f);
                    sums_1[0] = value_23_48;
                    float value_23_49 = ((split_1 < num_splits) ? _vec_load_12[1] : 0.0f);
                    sums_1[1] = value_23_49;
                    float value_23_50 = ((split_1 < num_splits) ? _vec_load_12[2] : 0.0f);
                    sums_1[2] = value_23_50;
                    float value_23_51 = ((split_1 < num_splits) ? _vec_load_12[3] : 0.0f);
                    sums_1[3] = value_23_51;
                    float _vec_load_13[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 4) + 0);
                        _vec_load_13[0 + 0] = _v4.x;
                        _vec_load_13[0 + 1] = _v4.y;
                        _vec_load_13[0 + 2] = _v4.z;
                        _vec_load_13[0 + 3] = _v4.w;
                    }
                    float value_23_52 = ((split_1 < num_splits) ? _vec_load_13[0] : 0.0f);
                    sums_1[4] = value_23_52;
                    float value_23_53 = ((split_1 < num_splits) ? _vec_load_13[1] : 0.0f);
                    sums_1[5] = value_23_53;
                    float value_23_54 = ((split_1 < num_splits) ? _vec_load_13[2] : 0.0f);
                    sums_1[6] = value_23_54;
                    float value_23_55 = ((split_1 < num_splits) ? _vec_load_13[3] : 0.0f);
                    sums_1[7] = value_23_55;
                    float _vec_load_14[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 8) + 0);
                        _vec_load_14[0 + 0] = _v4.x;
                        _vec_load_14[0 + 1] = _v4.y;
                        _vec_load_14[0 + 2] = _v4.z;
                        _vec_load_14[0 + 3] = _v4.w;
                    }
                    float value_23_56 = ((split_1 < num_splits) ? _vec_load_14[0] : 0.0f);
                    sums_1[8] = value_23_56;
                    float value_23_57 = ((split_1 < num_splits) ? _vec_load_14[1] : 0.0f);
                    sums_1[9] = value_23_57;
                    float value_23_58 = ((split_1 < num_splits) ? _vec_load_14[2] : 0.0f);
                    sums_1[10] = value_23_58;
                    float value_23_59 = ((split_1 < num_splits) ? _vec_load_14[3] : 0.0f);
                    sums_1[11] = value_23_59;
                    float _vec_load_15[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 12) + 0);
                        _vec_load_15[0 + 0] = _v4.x;
                        _vec_load_15[0 + 1] = _v4.y;
                        _vec_load_15[0 + 2] = _v4.z;
                        _vec_load_15[0 + 3] = _v4.w;
                    }
                    float value_23_60 = ((split_1 < num_splits) ? _vec_load_15[0] : 0.0f);
                    sums_1[12] = value_23_60;
                    float value_23_61 = ((split_1 < num_splits) ? _vec_load_15[1] : 0.0f);
                    sums_1[13] = value_23_61;
                    float value_23_62 = ((split_1 < num_splits) ? _vec_load_15[2] : 0.0f);
                    sums_1[14] = value_23_62;
                    float value_23_63 = ((split_1 < num_splits) ? _vec_load_15[3] : 0.0f);
                    sums_1[15] = value_23_63;
                    float _vec_load_16[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 16) + 0);
                        _vec_load_16[0 + 0] = _v4.x;
                        _vec_load_16[0 + 1] = _v4.y;
                        _vec_load_16[0 + 2] = _v4.z;
                        _vec_load_16[0 + 3] = _v4.w;
                    }
                    float value_23_64 = ((split_1 < num_splits) ? _vec_load_16[0] : 0.0f);
                    sums_1[16] = value_23_64;
                    float value_23_65 = ((split_1 < num_splits) ? _vec_load_16[1] : 0.0f);
                    sums_1[17] = value_23_65;
                    float value_23_66 = ((split_1 < num_splits) ? _vec_load_16[2] : 0.0f);
                    sums_1[18] = value_23_66;
                    float value_23_67 = ((split_1 < num_splits) ? _vec_load_16[3] : 0.0f);
                    sums_1[19] = value_23_67;
                    float _vec_load_17[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 20) + 0);
                        _vec_load_17[0 + 0] = _v4.x;
                        _vec_load_17[0 + 1] = _v4.y;
                        _vec_load_17[0 + 2] = _v4.z;
                        _vec_load_17[0 + 3] = _v4.w;
                    }
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
                    float _vec_load_18[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25_1 + 0);
                        _vec_load_18[0 + 0] = _v4.x;
                        _vec_load_18[0 + 1] = _v4.y;
                        _vec_load_18[0 + 2] = _v4.z;
                        _vec_load_18[0 + 3] = _v4.w;
                    }
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
                    float _vec_load_19[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 4) + 0);
                        _vec_load_19[0 + 0] = _v4.x;
                        _vec_load_19[0 + 1] = _v4.y;
                        _vec_load_19[0 + 2] = _v4.z;
                        _vec_load_19[0 + 3] = _v4.w;
                    }
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
                    float _vec_load_20[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 8) + 0);
                        _vec_load_20[0 + 0] = _v4.x;
                        _vec_load_20[0 + 1] = _v4.y;
                        _vec_load_20[0 + 2] = _v4.z;
                        _vec_load_20[0 + 3] = _v4.w;
                    }
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
                    float _vec_load_21[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 12) + 0);
                        _vec_load_21[0 + 0] = _v4.x;
                        _vec_load_21[0 + 1] = _v4.y;
                        _vec_load_21[0 + 2] = _v4.z;
                        _vec_load_21[0 + 3] = _v4.w;
                    }
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
                    float _vec_load_22[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 16) + 0);
                        _vec_load_22[0 + 0] = _v4.x;
                        _vec_load_22[0 + 1] = _v4.y;
                        _vec_load_22[0 + 2] = _v4.z;
                        _vec_load_22[0 + 3] = _v4.w;
                    }
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
                    float _vec_load_23[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 20) + 0);
                        _vec_load_23[0 + 0] = _v4.x;
                        _vec_load_23[0 + 1] = _v4.y;
                        _vec_load_23[0 + 2] = _v4.z;
                        _vec_load_23[0 + 3] = _v4.w;
                    }
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
                    float _add_206;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(sums_1[10]), "f"(_shfl_xor_sync_207));
                    sums_1[10] = _add_206;
                    float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 2);
                    float _add_208;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(sums_1[11]), "f"(_shfl_xor_sync_209));
                    sums_1[11] = _add_208;
                    float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 2);
                    float _add_210;
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
                    }
                } else if (lane_in_warp >= 4) {
                    new_post_mix[(unsigned long long)token * 4 + (unsigned long long)(lane_in_warp - 4)] = _mul_0;
                }
                __syncwarp();
            }
        }
    }

    // Kernel teardown ops
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 80;");
    mbarrier_wait(weight_full_addr, 0);
    float gs = fp4_global_scale[0];
    float rcp_gs = approx_rcp(gs);
    float r6 = approx_rcp(6.0f);
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
                    unsigned int _atomic_fetch_add_1 = atomicAdd(&exhausted[shard], 0);
                    if (_atomic_fetch_add_1 == 0) {
                        unsigned int _atomic_fetch_add_2 = atomicAdd(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)(unsigned int)(launch_epochs[bid] % 2) * 16 + (unsigned long long)shard) * 16)), 1);
                        ticket = local_tokens + _atomic_fetch_add_2 * 16 + shard;
                        if (ticket >= num_tokens) {
                            atomicMax(&exhausted[shard], 1);
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
                }
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
                float z_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z_1) : "f"(r_1), "f"(s_1));
                float a_map_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map_1) : "f"(z_1), "f"(sc_1), "f"(bs_1));
                float a_1 = ((lane_in_warp < 24) ? a_map_1 : 0.0f);
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(0.0f), "f"(fabsf(a_1)));
                float exp_1 = __nv_expf(_sub_71);
                float add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31) : "f"(1.0f), "f"(exp_1));
                float fdiv_rn_2 = __fdiv_rn(1.0f, add_31);
                float fdiv_rn_1_1 = __fdiv_rn(exp_1, add_31);
                float sig_1 = ((a_1 >= 0.0f) ? fdiv_rn_2 : fdiv_rn_1_1);
                float sneg_1 = ((a_1 >= 0.0f) ? fdiv_rn_1_1 : fdiv_rn_2);
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
                    float2 _f32x2_mul_a_72 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_73 = make_float2(lo, hi);
                    float2 pair_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                    values[0] = pair_2.x;
                    values[1] = pair_2.y;
                    unsigned int word_0_1 = early[pack * 16 + 1];
                    float lo_1 = __uint_as_float(word_0_1 << 16);
                    float hi_2 = __uint_as_float(word_0_1 & 4294901760u);
                    float2 _f32x2_mul_a_74 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_75 = make_float2(lo_1, hi_2);
                    float2 pair_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                    values[2] = pair_3.x;
                    values[3] = pair_3.y;
                    unsigned int word_4 = early[pack * 16 + 2];
                    float lo_5 = __uint_as_float(word_4 << 16);
                    float hi_6 = __uint_as_float(word_4 & 4294901760u);
                    float2 _f32x2_mul_a_76 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_77 = make_float2(lo_5, hi_6);
                    float2 pair_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                    values[4] = pair_7.x;
                    values[5] = pair_7.y;
                    unsigned int word_8 = early[pack * 16 + 3];
                    float lo_9 = __uint_as_float(word_8 << 16);
                    float hi_10 = __uint_as_float(word_8 & 4294901760u);
                    float2 _f32x2_mul_a_78 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_79 = make_float2(lo_9, hi_10);
                    float2 pair_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
                    values[6] = pair_11.x;
                    values[7] = pair_11.y;
                } else {
                    float _vec_load_26[8];
                    {
                        const uint4* _vptr_80 = reinterpret_cast<const uint4*>(new_residual + off + 0);
                        uint4 _vld_80[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vld_80[_blk].x), "=r"(_vld_80[_blk].y), "=r"(_vld_80[_blk].z), "=r"(_vld_80[_blk].w) : "l"((const void*)(_vptr_80 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                                (&_vec_load_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float2 _f32x2_mul_a_81 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_82 = make_float2(_vec_load_26[0], _vec_load_26[1]);
                    float2 pair_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
                    values[0] = pair_4.x;
                    values[1] = pair_4.y;
                    float2 _f32x2_mul_a_83 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_84 = make_float2(_vec_load_26[2], _vec_load_26[3]);
                    float2 pair_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
                    values[2] = pair_0.x;
                    values[3] = pair_0.y;
                    float2 _f32x2_mul_a_85 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_86 = make_float2(_vec_load_26[4], _vec_load_26[5]);
                    float2 pair_1_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                    values[4] = pair_1_1.x;
                    values[5] = pair_1_1.y;
                    float2 _f32x2_mul_a_87 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_88 = make_float2(_vec_load_26[6], _vec_load_26[7]);
                    float2 pair_2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
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
                    float _vec_load_27[8];
                    {
                        const uint4* _vptr_89 = reinterpret_cast<const uint4*>(new_residual + (off + 4096) + 0);
                        uint4 _vld_89[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vld_89[_blk].x), "=r"(_vld_89[_blk].y), "=r"(_vld_89[_blk].z), "=r"(_vld_89[_blk].w) : "l"((const void*)(_vptr_89 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                                (&_vec_load_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float2 pair_9 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[0], _vec_load_27[1]), make_float2(values[0], values[1]));
                    values[0] = pair_9.x;
                    values[1] = pair_9.y;
                    float2 pair_0_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[2], _vec_load_27[3]), make_float2(values[2], values[3]));
                    values[2] = pair_0_1.x;
                    values[3] = pair_0_1.y;
                    float2 pair_1_2 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[4], _vec_load_27[5]), make_float2(values[4], values[5]));
                    values[4] = pair_1_2.x;
                    values[5] = pair_1_2.y;
                    float2 pair_2_2 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[6], _vec_load_27[7]), make_float2(values[6], values[7]));
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
                    float _vec_load_28[8];
                    {
                        const uint4* _vptr_90 = reinterpret_cast<const uint4*>(new_residual + (off + 8192) + 0);
                        uint4 _vld_90[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vld_90[_blk].x), "=r"(_vld_90[_blk].y), "=r"(_vld_90[_blk].z), "=r"(_vld_90[_blk].w) : "l"((const void*)(_vptr_90 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_90 = reinterpret_cast<uint32_t*>(&_vld_90[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) << 16);
                                (&_vec_load_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float2 pair_12 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[0], _vec_load_28[1]), make_float2(values[0], values[1]));
                    values[0] = pair_12.x;
                    values[1] = pair_12.y;
                    float2 pair_0_2 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[2], _vec_load_28[3]), make_float2(values[2], values[3]));
                    values[2] = pair_0_2.x;
                    values[3] = pair_0_2.y;
                    float2 pair_1_3 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[4], _vec_load_28[5]), make_float2(values[4], values[5]));
                    values[4] = pair_1_3.x;
                    values[5] = pair_1_3.y;
                    float2 pair_2_3 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[6], _vec_load_28[7]), make_float2(values[6], values[7]));
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
                    float _vec_load_29[8];
                    {
                        const uint4* _vptr_91 = reinterpret_cast<const uint4*>(new_residual + (off + 12288) + 0);
                        uint4 _vld_91[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vld_91[_blk].x), "=r"(_vld_91[_blk].y), "=r"(_vld_91[_blk].z), "=r"(_vld_91[_blk].w) : "l"((const void*)(_vptr_91 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_91 = reinterpret_cast<uint32_t*>(&_vld_91[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) << 16);
                                (&_vec_load_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float2 pair_14 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[0], _vec_load_29[1]), make_float2(values[0], values[1]));
                    values[0] = pair_14.x;
                    values[1] = pair_14.y;
                    float2 pair_0_3 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[2], _vec_load_29[3]), make_float2(values[2], values[3]));
                    values[2] = pair_0_3.x;
                    values[3] = pair_0_3.y;
                    float2 pair_1_4 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[4], _vec_load_29[5]), make_float2(values[4], values[5]));
                    values[4] = pair_1_4.x;
                    values[5] = pair_1_4.y;
                    float2 pair_2_4 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[6], _vec_load_29[7]), make_float2(values[6], values[7]));
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
            float _add_92;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(sum0), "f"(sum1));
            float _warp_reduce_0 = _add_92;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            if (lane_in_warp == 0) {
                tail_scratch[warp] = _warp_reduce_0;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
            float row_sum = ((lane_in_warp < 4) ? tail_scratch[warp / 4 * 4 + lane_in_warp] : 0.0f);
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
            float rstd;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rstd) : "f"(_rcp_97), "f"(rmsnorm_scale));
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
                    float2 _f32x2_mul_a_100 = make_float2(lo_6, hi_5);
                    float2 _f32x2_mul_b_101 = make_float2(rstd, rstd);
                    float2 scaled;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
                    float2 _f32x2_mul_b_102 = make_float2(weight_lo, weight_hi);
                    float2 weighted;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
                    normalized_words[j] = __as_u32(__float22bfloat162_rn(make_float2(weighted.x, weighted.y)));
                }
                unsigned long long off_1 = (unsigned long long)token_1 * 4096 + (unsigned long long)(hidden_pack_1 * 256) + (unsigned long long)(lane_in_warp * 8);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %6, 0; @p_ st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5; }" :: "l"(y_bf16 + off_1), "r"(normalized_words[(0) + 0]), "r"(normalized_words[(0) + 1]), "r"(normalized_words[(0) + 2]), "r"(normalized_words[(0) + 3]), "l"(0x12F0000000000000ULL), "r"((unsigned int)(y_fit)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(y_bf16 + off_1), "r"(normalized_words[(0) + 0]), "r"(normalized_words[(0) + 1]), "r"(normalized_words[(0) + 2]), "r"(normalized_words[(0) + 3]), "r"((unsigned int)(!y_fit)) : "memory");
                uint32_t _bf16x2_abs_0;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(normalized_words[0]));
                uint32_t _bf16x2_abs_1;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(normalized_words[2]));
                uint32_t _bf16x2_max_0;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
                uint32_t _bf16x2_abs_2;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(normalized_words[1]));
                uint32_t _bf16x2_abs_3;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(normalized_words[3]));
                uint32_t _bf16x2_max_1;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_abs_2), "r"(_bf16x2_abs_3));
                uint32_t _bf16x2_max_2;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_0), "r"(_bf16x2_max_1));
                float amax;
                asm("max.f32 %0, %1, %2;" : "=f"(amax) : "f"(__uint_as_float(_bf16x2_max_2 << 16)), "f"(__uint_as_float(_bf16x2_max_2 & 4294901760u)));
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
                float amax1;
                asm("max.f32 %0, %1, %2;" : "=f"(amax1) : "f"(amax), "f"(_shfl_xor_sync_103));
                uint16_t _e4m3x2_104;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_104) : "f"(0.0f), "f"(mul_f32_rn_noftz(gs, mul_f32_rn_noftz(amax1, r6))));
                uint8_t _e4m3_scalar_0 = (uint8_t)_e4m3x2_104;
                uint32_t _f16x2_decode_105;
                float _fp8_decode_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_105) : "h"(_e4m3x2_104));
                uint16_t _f16_decode_105 = (uint16_t)_f16x2_decode_105;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_105));
                uint32_t _setp_106_bits;
                asm("{ .reg .pred p; setp.neu.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_106_bits) : "f"(_fp8_decode_0), "f"(0.0f));
                bool _setp_106 = _setp_106_bits != 0;
                float _rcp_107 = approx_rcp(mul_f32_rn_noftz(_fp8_decode_0, rcp_gs));
                float inv = ((_setp_106) ? _rcp_107 : 0.0f);
                float q_2[8];
                #pragma unroll
                for (int j_1 = 0; j_1 < 4; j_1++) {
                    q_2[2 * j_1] = mul_f32_rn_noftz(__uint_as_float(normalized_words[j_1] << 16), inv);
                    q_2[2 * j_1 + 1] = mul_f32_rn_noftz(__uint_as_float(normalized_words[j_1] & 4294901760u), inv);
                }
                unsigned int packed[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(q_2[0]), "f"(q_2[1]), "f"(q_2[2]), "f"(q_2[3]), "f"(q_2[4]), "f"(q_2[5]), "f"(q_2[6]), "f"(q_2[7]));
                unsigned long long off4 = (unsigned long long)token_1 * 2048 + (unsigned long long)(hidden_pack_1 * 128) + (unsigned long long)(lane_in_warp * 4);
                *(reinterpret_cast<unsigned int*>(y_fp4 + off4) + (0)) = packed[0];
                unsigned long long sf_idx = (unsigned long long)token_1 * 256 + (unsigned long long)(hidden_pack_1 * 16 + lane_in_warp / 2);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b8 [%0], %1; }" :: "l"(y_fp4_sf + sf_idx), "r"((unsigned int)(unsigned char)((unsigned int)_e4m3_scalar_0)), "r"((unsigned int)(lane_in_warp % 2 == 0)) : "memory");
                if (pack_1 + 1 < 4) {
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 4; j_2++) {
                        weight_cache[j_2] = next_weight[j_2];
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
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
