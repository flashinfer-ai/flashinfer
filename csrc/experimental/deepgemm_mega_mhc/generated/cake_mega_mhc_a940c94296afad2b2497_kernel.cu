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
#define SMEM_EXHAUSTED_OFF 207680
#define SMEM_EXHAUSTED_STAGE_BYTES 64
#define SMEM_EXHAUSTED_STRIDE 64
#else
#define SMEM_PAIR_SUMS_OFF 207616
#define SMEM_PAIR_SUMS_STAGE_BYTES 256
#define SMEM_PAIR_SUMS_STRIDE 256
#define SMEM_EXHAUSTED_OFF 207580
#define SMEM_EXHAUSTED_STAGE_BYTES 4
#define SMEM_EXHAUSTED_STRIDE 4
#endif
#define SMEM_TOTAL 208896
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



__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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
kernel_cake_mega_mhc_a940c94296afad2b2497(const __grid_constant__ CUtensorMap residual_map, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap fnp_map, const __grid_constant__ CUtensorMap post_map, const __grid_constant__ CUtensorMap comb_map, const __grid_constant__ CUtensorMap new_residual_map, float* __restrict__ mix_scales, float* __restrict__ mix_bases, float* __restrict__ new_post_mix, float* __restrict__ new_comb_res_mix, __nv_bfloat16* __restrict__ rmsnorm_weight, __nv_bfloat16* __restrict__ new_residual, __nv_bfloat16* __restrict__ y_bf16, float* __restrict__ scratch, unsigned long long* __restrict__ split_barriers, unsigned long long* __restrict__ launch_epochs, unsigned int num_tokens, unsigned int num_splits, CakeFastDivmod split_div, unsigned int num_ctas, float hc_norm_eps, float hc_pre_eps, float hc_post_scale, float sinkhorn_eps, unsigned int num_sinkhorn_iters, float rmsnorm_eps, float rmsnorm_scale, uint8_t* __restrict__ y_fp8, float* __restrict__ y_fp8_sf, unsigned int fp8_sf_ue8m0)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 207872;
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
    unsigned int* exhausted = reinterpret_cast<unsigned int*>(smem_raw + 207680);
    const int exhausted_addr = smem + 207680;
#else
    unsigned int* queue = reinterpret_cast<unsigned int*>(smem_raw + 207576);
    const int queue_addr = smem + 207576;
    float* pair_sums = reinterpret_cast<float*>(smem_raw + 207616);
    const int pair_sums_addr = smem + 207616;
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

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[207872..208160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_io: 3 barriers, init_count=1
            mbarrier_init(smem + 207872, 1);
            mbarrier_init(smem + 207880, 1);
            mbarrier_init(smem + 207888, 1);
            // empty_io: 3 barriers, init_count=1
            mbarrier_init(smem + 207896, 1);
            mbarrier_init(smem + 207904, 1);
            mbarrier_init(smem + 207912, 1);
            // store_ready: 3 barriers, init_count=8
            mbarrier_init(smem + 207920, 8);
            mbarrier_init(smem + 207928, 8);
            mbarrier_init(smem + 207936, 8);
            // full_coeff: 2 barriers, init_count=1
            mbarrier_init(smem + 208104, 1);
            mbarrier_init(smem + 208112, 1);
            // empty_coeff: 2 barriers, init_count=8
            mbarrier_init(smem + 208120, 8);
            mbarrier_init(smem + 208128, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 207944, 1);
            mbarrier_init(smem + 207952, 1);
            // empty_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 207960, 1);
            mbarrier_init(smem + 207968, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_a: 4 barriers, init_count=128
            mbarrier_init(smem + 207976, 128);
            mbarrier_init(smem + 207984, 128);
            mbarrier_init(smem + 207992, 128);
            mbarrier_init(smem + 208000, 128);
            // empty_a: 4 barriers, init_count=1
            mbarrier_init(smem + 208008, 1);
            mbarrier_init(smem + 208016, 1);
            mbarrier_init(smem + 208024, 1);
            mbarrier_init(smem + 208032, 1);
            // big_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208040, 1);
            mbarrier_init(smem + 208048, 1);
            // big_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 208056, 4);
            mbarrier_init(smem + 208064, 4);
            // small_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208072, 1);
            mbarrier_init(smem + 208080, 1);
            // small_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 208088, 4);
            mbarrier_init(smem + 208096, 4);
            // full_stats: 1 barriers, init_count=8
            mbarrier_init(smem + 208136, 8);
            // empty_stats: 1 barriers, init_count=4
            mbarrier_init(smem + 208144, 4);
            // mix_arrival: 1 barriers, init_count=5
            mbarrier_init(smem + 208152, 5);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

#if __CUDA_ARCH__ == 1070
    // TMEM alloc (256 columns, 224 used)
#else
    // TMEM alloc (512 columns, 272 used)
#endif
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208160);
    if (warp == 3) {
        int _tmem_hold = smem + 208160;
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
                        uint32_t _fast_div_q_1 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task);
                        uint32_t _fast_div_r_1 = (uint32_t)(task) - _fast_div_q_1 * (uint32_t)(split_div.divisor);
                        #pragma unroll 1
                        for (unsigned int kb = 0; kb < _fast_div_q_0 + ((_fast_div_r_1 < _fast_div_r_0) ? 1 : 0); kb++) {
                            unsigned int hidden = (_fast_div_r_1 * _fast_div_q_0 + ((_fast_div_r_1 < _fast_div_r_0) ? _fast_div_r_1 : _fast_div_r_0)) * 64 + kb * 64;
                            if (warp == 0) {
                                unsigned int stage = seq % 3;
                                unsigned int _stage_empty_io = seq % 3;
                                unsigned int _parity_empty_io = seq / 3 % 2 ^ 1;
                                mbarrier_wait(empty_io_addr + (_stage_empty_io) * 8, _parity_empty_io);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(residual_addr + stage * 32768), "l"((&residual_map)), "r"(hidden), "r"(_fast_div_q_1 * 64), "r"(0),
                                       "r"(full_io_addr + (stage) * 8), "l"(io_policy) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(x_addr + stage * 8192), "l"((&x_map)), "r"(hidden), "r"(_fast_div_q_1 * 64),
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
                    uint32_t _fast_div_q_2 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_1), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_1);
                    uint32_t _fast_div_r_2 = (uint32_t)(task_1) - _fast_div_q_2 * (uint32_t)(split_div.divisor);
                    unsigned int ustage = useq % 2;
                    unsigned int _stage_small_empty = useq % 2;
                    unsigned int _parity_small_empty = useq / 2 % 2 ^ 1;
                    mbarrier_wait(small_empty_addr + (_stage_small_empty) * 8, _parity_small_empty);
                    #pragma unroll 1
                    for (unsigned int kb_1 = 0; kb_1 < _fast_div_q_0 + ((_fast_div_r_2 < _fast_div_r_0) ? 1 : 0); kb_1++) {
                        unsigned int fstage = fseq % 2;
                        unsigned int _stage_full_fn = fseq % 2;
                        unsigned int _parity_full_fn = fseq / 2 % 2;
                        mbarrier_wait(full_fn_addr + (_stage_full_fn) * 8, _parity_full_fn);
                        unsigned int cstage = cseq % 2;
                        unsigned int _stage_big_empty = cseq % 2;
                        unsigned int _parity_big_empty = cseq / 2 % 2 ^ 1;
                        mbarrier_wait(big_empty_addr + (_stage_big_empty) * 8, _parity_big_empty);
                        unsigned int astage = aseq % 4;
                        unsigned int _stage_full_a = aseq % 4;
                        unsigned int _parity_full_a = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a) * 8, _parity_full_a);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _mma_b_lo_0, 0x40004040, 67503248, ((kb_1 == 0) ? 0 : 1));
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _mma_b_lo_1, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _mma_b_lo_2, 0x40004040, 67503248, 0);
                        int _mma_b_lo_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48)), (unsigned int)tmem_a_tmem + astage * 32, _mma_b_lo_2, 0x40004040, 67503248, 0);
                        int _mma_b_lo_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _mma_b_lo_3, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_4 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_4 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _mma_b_lo_4, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_5 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _mma_b_lo_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_6 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_5 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _mma_b_lo_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_6 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _mma_b_lo_6, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_7 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_7 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _mma_b_lo_7, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_8 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _mma_b_lo_8, 0x40004040, 67503248, 1);
                        int _mma_b_lo_9 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_8 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _mma_b_lo_8, 0x40004040, 67503248, 1);
                        int _mma_b_lo_9 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _mma_b_lo_9, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_10 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_10 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _mma_b_lo_10, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_11 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _mma_b_lo_11, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_11 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _mma_b_lo_11, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage) * 8);
                        aseq += 1;
                        unsigned int astage_0 = aseq % 4;
                        unsigned int _stage_full_a_1 = aseq % 4;
                        unsigned int _parity_full_a_1 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_1) * 8, _parity_full_a_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_12 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_12 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _mma_b_lo_12, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_13 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_13 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _mma_b_lo_13, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_14 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _mma_b_lo_14, 0x40004040, 67503248, 1);
                        int _mma_b_lo_15 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_14 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48 + 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _mma_b_lo_14, 0x40004040, 67503248, 0);
                        int _mma_b_lo_15 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _mma_b_lo_15, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_16 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_16 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _mma_b_lo_16, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_17 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _mma_b_lo_17, 0x40004040, 67503248, 1);
                        int _mma_b_lo_18 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_17 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _mma_b_lo_17, 0x40004040, 67503248, 1);
                        int _mma_b_lo_18 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _mma_b_lo_18, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_19 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_19 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _mma_b_lo_19, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_20 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _mma_b_lo_20, 0x40004040, 67503248, 1);
                        int _mma_b_lo_21 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_20 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _mma_b_lo_20, 0x40004040, 67503248, 1);
                        int _mma_b_lo_21 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _mma_b_lo_21, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_22 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_22 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _mma_b_lo_22, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_23 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _mma_b_lo_23, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_23 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _mma_b_lo_23, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_0) * 8);
                        aseq += 1;
                        elect_commit(big_full_addr + (cstage) * 8);
                        cseq += 1;
                        unsigned int cstage_1 = cseq % 2;
                        unsigned int _stage_big_empty_1 = cseq % 2;
                        unsigned int _parity_big_empty_1 = cseq / 2 % 2 ^ 1;
                        mbarrier_wait(big_empty_addr + (_stage_big_empty_1) * 8, _parity_big_empty_1);
                        unsigned int astage_2 = aseq % 4;
                        unsigned int _stage_full_a_2 = aseq % 4;
                        unsigned int _parity_full_a_2 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_2) * 8, _parity_full_a_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_24 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_24 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_2 * 32, _mma_b_lo_24, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_25 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_25 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_2 * 32, _mma_b_lo_25, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_26 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + astage_2 * 32, _mma_b_lo_26, 0x40004040, 67503248, 0);
                        int _mma_b_lo_27 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_26 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48)), (unsigned int)tmem_a_tmem + astage_2 * 32, _mma_b_lo_26, 0x40004040, 67503248, 0);
                        int _mma_b_lo_27 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 8), _mma_b_lo_27, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_28 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_28 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 8), _mma_b_lo_28, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_29 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 8), _mma_b_lo_29, 0x40004040, 67503248, 1);
                        int _mma_b_lo_30 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_29 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 8), _mma_b_lo_29, 0x40004040, 67503248, 1);
                        int _mma_b_lo_30 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 16), _mma_b_lo_30, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_31 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_31 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 16), _mma_b_lo_31, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_32 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 16), _mma_b_lo_32, 0x40004040, 67503248, 1);
                        int _mma_b_lo_33 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_32 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 16), _mma_b_lo_32, 0x40004040, 67503248, 1);
                        int _mma_b_lo_33 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 24), _mma_b_lo_33, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_34 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_34 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 24), _mma_b_lo_34, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_35 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 24), _mma_b_lo_35, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_35 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 24), _mma_b_lo_35, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_2) * 8);
                        aseq += 1;
                        unsigned int astage_3 = aseq % 4;
                        unsigned int _stage_full_a_3 = aseq % 4;
                        unsigned int _parity_full_a_3 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_3) * 8, _parity_full_a_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_36 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_36 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_3 * 32, _mma_b_lo_36, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_37 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_37 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_3 * 32, _mma_b_lo_37, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_38 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + astage_3 * 32, _mma_b_lo_38, 0x40004040, 67503248, 1);
                        int _mma_b_lo_39 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_38 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48 + 24)), (unsigned int)tmem_a_tmem + astage_3 * 32, _mma_b_lo_38, 0x40004040, 67503248, 0);
                        int _mma_b_lo_39 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 8), _mma_b_lo_39, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_40 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_40 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 8), _mma_b_lo_40, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_41 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 8), _mma_b_lo_41, 0x40004040, 67503248, 1);
                        int _mma_b_lo_42 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_41 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 8), _mma_b_lo_41, 0x40004040, 67503248, 1);
                        int _mma_b_lo_42 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 16), _mma_b_lo_42, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_43 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_43 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 16), _mma_b_lo_43, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_44 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 16), _mma_b_lo_44, 0x40004040, 67503248, 1);
                        int _mma_b_lo_45 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_44 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 16), _mma_b_lo_44, 0x40004040, 67503248, 1);
                        int _mma_b_lo_45 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 24), _mma_b_lo_45, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_46 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_46 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 24), _mma_b_lo_46, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_47 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 24), _mma_b_lo_47, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_47 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_1 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 24), _mma_b_lo_47, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_3) * 8);
                        aseq += 1;
                        elect_commit(big_full_addr + (cstage_1) * 8);
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
                        uint32_t _fast_div_q_3 = (split_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)bid), split_div.multiplier) >> split_div.shift_right) : (uint32_t)((unsigned int)bid);
                        uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)bid) - _fast_div_q_3 * (uint32_t)(split_div.divisor);
                        tma_2d_gmem2smem(post_addr, (&post_map), 0, _fast_div_q_3 * 64, full_coeff_addr);
                        tma_2d_gmem2smem(comb_addr, (&comb_map), 0, _fast_div_q_3 * 64, full_coeff_addr);
                        mbarrier_arrive_expect_tx(full_coeff_addr, 5120);
                    }
                    __syncwarp();
                    coeff_seq += 1;
                }
                #pragma unroll 1
                for (unsigned int task_2 = (unsigned int)bid; task_2 < (num_tokens + 63) / 64 * num_splits; task_2 += num_ctas) {
                    uint32_t _fast_div_q_4 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_2), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_2);
                    uint32_t _fast_div_r_4 = (uint32_t)(task_2) - _fast_div_q_4 * (uint32_t)(split_div.divisor);
                    unsigned int blocks = _fast_div_q_0 + ((_fast_div_r_4 < _fast_div_r_0) ? 1 : 0);
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
                                :: "l"((&new_residual_map)), "r"((_fast_div_r_4 * _fast_div_q_0 + ((_fast_div_r_4 < _fast_div_r_0) ? _fast_div_r_4 : _fast_div_r_0)) * 64 + kb_2 * 64), "r"(_fast_div_q_4 * 64), "r"(0), "r"(residual_addr + stage_2 * 32768), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                            if (prefetch != 0) {
                                uint32_t _fast_div_q_5 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_2 + num_ctas), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_2 + num_ctas);
                                uint32_t _fast_div_r_5 = (uint32_t)(task_2 + num_ctas) - _fast_div_q_5 * (uint32_t)(split_div.divisor);
                                tma_2d_gmem2smem(post_addr + coeff_seq % 2 * 1024, (&post_map), 0, _fast_div_q_5 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                tma_2d_gmem2smem(comb_addr + coeff_seq % 2 * 4096, (&comb_map), 0, _fast_div_q_5 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
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
                uint32_t _fast_div_q_6 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_3), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_3);
                uint32_t _fast_div_r_6 = (uint32_t)(task_3) - _fast_div_q_6 * (uint32_t)(split_div.divisor);
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
                for (unsigned int kb_3 = 0; kb_3 < _fast_div_q_0 + ((_fast_div_r_6 < _fast_div_r_0) ? 1 : 0); kb_3++) {
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
                    float cc_3 = comb_coeff[4];
                    value = fma_f32x2_rn_noftz(make_float2(cc_3, cc_3), make_float2(rv[2], rv[3]), value);
                    float cc_4 = comb_coeff[8];
                    value = fma_f32x2_rn_noftz(make_float2(cc_4, cc_4), make_float2(rv[4], rv[5]), value);
                    float cc_5 = comb_coeff[12];
                    value = fma_f32x2_rn_noftz(make_float2(cc_5, cc_5), make_float2(rv[6], rv[7]), value);
                    unsigned int word_6 = __as_u32(__float22bfloat162_rn(make_float2(value.x, value.y)));
                    residual_pairs[io_stage * 8192 + offset] = word_6;
                    words[0] = word_6;
                    float2 rounded_pair = make_float2(__uint_as_float(word_6 << 16), __uint_as_float(word_6 & 4294901760u));
                    float2 squares = fma_f32x2_rn_noftz(rounded_pair, rounded_pair, make_float2(sq[0], sq[1]));
                    sq[0] = squares.x;
                    sq[1] = squares.y;
                    float coeff_7 = post_coeff[1];
                    float2 _f32x2_mul_a_1 = make_float2(coeff_7, coeff_7);
                    float2 value_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&xv));
                    float cc_9 = comb_coeff[1];
                    value_8 = fma_f32x2_rn_noftz(make_float2(cc_9, cc_9), make_float2(rv[0], rv[1]), value_8);
                    float cc_10 = comb_coeff[5];
                    value_8 = fma_f32x2_rn_noftz(make_float2(cc_10, cc_10), make_float2(rv[2], rv[3]), value_8);
                    float cc_11 = comb_coeff[9];
                    value_8 = fma_f32x2_rn_noftz(make_float2(cc_11, cc_11), make_float2(rv[4], rv[5]), value_8);
                    float cc_12 = comb_coeff[13];
                    value_8 = fma_f32x2_rn_noftz(make_float2(cc_12, cc_12), make_float2(rv[6], rv[7]), value_8);
                    unsigned int word_13 = __as_u32(__float22bfloat162_rn(make_float2(value_8.x, value_8.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset] = word_13;
                    words[4] = word_13;
                    float2 rounded_pair_14 = make_float2(__uint_as_float(word_13 << 16), __uint_as_float(word_13 & 4294901760u));
                    float2 squares_15 = fma_f32x2_rn_noftz(rounded_pair_14, rounded_pair_14, make_float2(sq[0], sq[1]));
                    sq[0] = squares_15.x;
                    sq[1] = squares_15.y;
                    float coeff_16 = post_coeff[2];
                    float2 _f32x2_mul_a_2 = make_float2(coeff_16, coeff_16);
                    float2 value_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&xv));
                    float cc_18 = comb_coeff[2];
                    value_17 = fma_f32x2_rn_noftz(make_float2(cc_18, cc_18), make_float2(rv[0], rv[1]), value_17);
                    float cc_19 = comb_coeff[6];
                    value_17 = fma_f32x2_rn_noftz(make_float2(cc_19, cc_19), make_float2(rv[2], rv[3]), value_17);
                    float cc_20 = comb_coeff[10];
                    value_17 = fma_f32x2_rn_noftz(make_float2(cc_20, cc_20), make_float2(rv[4], rv[5]), value_17);
                    float cc_21 = comb_coeff[14];
                    value_17 = fma_f32x2_rn_noftz(make_float2(cc_21, cc_21), make_float2(rv[6], rv[7]), value_17);
                    unsigned int word_22 = __as_u32(__float22bfloat162_rn(make_float2(value_17.x, value_17.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset] = word_22;
                    words[8] = word_22;
                    float2 rounded_pair_23 = make_float2(__uint_as_float(word_22 << 16), __uint_as_float(word_22 & 4294901760u));
                    float2 squares_24 = fma_f32x2_rn_noftz(rounded_pair_23, rounded_pair_23, make_float2(sq[0], sq[1]));
                    sq[0] = squares_24.x;
                    sq[1] = squares_24.y;
                    float coeff_25 = post_coeff[3];
                    float2 _f32x2_mul_a_3 = make_float2(coeff_25, coeff_25);
                    float2 value_26;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&xv));
                    float cc_27 = comb_coeff[3];
                    value_26 = fma_f32x2_rn_noftz(make_float2(cc_27, cc_27), make_float2(rv[0], rv[1]), value_26);
                    float cc_28 = comb_coeff[7];
                    value_26 = fma_f32x2_rn_noftz(make_float2(cc_28, cc_28), make_float2(rv[2], rv[3]), value_26);
                    float cc_29 = comb_coeff[11];
                    value_26 = fma_f32x2_rn_noftz(make_float2(cc_29, cc_29), make_float2(rv[4], rv[5]), value_26);
                    float cc_30 = comb_coeff[15];
                    value_26 = fma_f32x2_rn_noftz(make_float2(cc_30, cc_30), make_float2(rv[6], rv[7]), value_26);
                    unsigned int word_31 = __as_u32(__float22bfloat162_rn(make_float2(value_26.x, value_26.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset] = word_31;
                    words[12] = word_31;
                    float2 rounded_pair_32 = make_float2(__uint_as_float(word_31 << 16), __uint_as_float(word_31 & 4294901760u));
                    float2 squares_33 = fma_f32x2_rn_noftz(rounded_pair_32, rounded_pair_32, make_float2(sq[0], sq[1]));
                    sq[0] = squares_33.x;
                    sq[1] = squares_33.y;
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(sq[0]), "f"(sq[1]));
                    float _sub_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_add_4), "f"(hc_comp[0]));
                    float _add_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(hc_acc[0]), "f"(_sub_0));
                    float _sub_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(_add_0), "f"(hc_acc[0]));
                    float _sub_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(_sub_6), "f"(_sub_0));
                    hc_comp[0] = _sub_5;
                    hc_acc[0] = _add_0;
                    unsigned int offset_34 = offs[1];
                    unsigned int x_word_35 = x_pairs[io_stage * 2048 + offset_34];
                    float2 xv_36 = make_float2(__uint_as_float(x_word_35 << 16), __uint_as_float(x_word_35 & 4294901760u));
                    float rv_37[8];
                    unsigned int word_38 = residual_pairs[io_stage * 8192 + offset_34];
                    rv_37[0] = __uint_as_float(word_38 << 16);
                    rv_37[1] = __uint_as_float(word_38 & 4294901760u);
                    unsigned int word_39 = residual_pairs[io_stage * 8192 + 2048 + offset_34];
                    rv_37[2] = __uint_as_float(word_39 << 16);
                    rv_37[3] = __uint_as_float(word_39 & 4294901760u);
                    unsigned int word_40 = residual_pairs[io_stage * 8192 + 4096 + offset_34];
                    rv_37[4] = __uint_as_float(word_40 << 16);
                    rv_37[5] = __uint_as_float(word_40 & 4294901760u);
                    unsigned int word_41 = residual_pairs[io_stage * 8192 + 6144 + offset_34];
                    rv_37[6] = __uint_as_float(word_41 << 16);
                    rv_37[7] = __uint_as_float(word_41 & 4294901760u);
                    float sq_42[2];
                    sq_42[0] = 0.0f;
                    sq_42[1] = 0.0f;
                    float coeff_43 = post_coeff[4];
                    float2 _f32x2_mul_a_7 = make_float2(coeff_43, coeff_43);
                    float2 value_44;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&xv_36));
                    float cc_45 = comb_coeff[16];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_45, cc_45), make_float2(rv_37[0], rv_37[1]), value_44);
                    float cc_46 = comb_coeff[20];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_46, cc_46), make_float2(rv_37[2], rv_37[3]), value_44);
                    float cc_47 = comb_coeff[24];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_47, cc_47), make_float2(rv_37[4], rv_37[5]), value_44);
                    float cc_48 = comb_coeff[28];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_48, cc_48), make_float2(rv_37[6], rv_37[7]), value_44);
                    unsigned int word_49 = __as_u32(__float22bfloat162_rn(make_float2(value_44.x, value_44.y)));
                    residual_pairs[io_stage * 8192 + offset_34] = word_49;
                    words[2] = word_49;
                    float2 rounded_pair_50 = make_float2(__uint_as_float(word_49 << 16), __uint_as_float(word_49 & 4294901760u));
                    float2 squares_51 = fma_f32x2_rn_noftz(rounded_pair_50, rounded_pair_50, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_51.x;
                    sq_42[1] = squares_51.y;
                    float coeff_52 = post_coeff[5];
                    float2 _f32x2_mul_a_8 = make_float2(coeff_52, coeff_52);
                    float2 value_53;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&xv_36));
                    float cc_54 = comb_coeff[17];
                    value_53 = fma_f32x2_rn_noftz(make_float2(cc_54, cc_54), make_float2(rv_37[0], rv_37[1]), value_53);
                    float cc_55 = comb_coeff[21];
                    value_53 = fma_f32x2_rn_noftz(make_float2(cc_55, cc_55), make_float2(rv_37[2], rv_37[3]), value_53);
                    float cc_56 = comb_coeff[25];
                    value_53 = fma_f32x2_rn_noftz(make_float2(cc_56, cc_56), make_float2(rv_37[4], rv_37[5]), value_53);
                    float cc_57 = comb_coeff[29];
                    value_53 = fma_f32x2_rn_noftz(make_float2(cc_57, cc_57), make_float2(rv_37[6], rv_37[7]), value_53);
                    unsigned int word_58 = __as_u32(__float22bfloat162_rn(make_float2(value_53.x, value_53.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_34] = word_58;
                    words[6] = word_58;
                    float2 rounded_pair_59 = make_float2(__uint_as_float(word_58 << 16), __uint_as_float(word_58 & 4294901760u));
                    float2 squares_60 = fma_f32x2_rn_noftz(rounded_pair_59, rounded_pair_59, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_60.x;
                    sq_42[1] = squares_60.y;
                    float coeff_61 = post_coeff[6];
                    float2 _f32x2_mul_a_9 = make_float2(coeff_61, coeff_61);
                    float2 value_62;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&xv_36));
                    float cc_63 = comb_coeff[18];
                    value_62 = fma_f32x2_rn_noftz(make_float2(cc_63, cc_63), make_float2(rv_37[0], rv_37[1]), value_62);
                    float cc_64 = comb_coeff[22];
                    value_62 = fma_f32x2_rn_noftz(make_float2(cc_64, cc_64), make_float2(rv_37[2], rv_37[3]), value_62);
                    float cc_65 = comb_coeff[26];
                    value_62 = fma_f32x2_rn_noftz(make_float2(cc_65, cc_65), make_float2(rv_37[4], rv_37[5]), value_62);
                    float cc_66 = comb_coeff[30];
                    value_62 = fma_f32x2_rn_noftz(make_float2(cc_66, cc_66), make_float2(rv_37[6], rv_37[7]), value_62);
                    unsigned int word_67 = __as_u32(__float22bfloat162_rn(make_float2(value_62.x, value_62.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_34] = word_67;
                    words[10] = word_67;
                    float2 rounded_pair_68 = make_float2(__uint_as_float(word_67 << 16), __uint_as_float(word_67 & 4294901760u));
                    float2 squares_69 = fma_f32x2_rn_noftz(rounded_pair_68, rounded_pair_68, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_69.x;
                    sq_42[1] = squares_69.y;
                    float coeff_70 = post_coeff[7];
                    float2 _f32x2_mul_a_10 = make_float2(coeff_70, coeff_70);
                    float2 value_71;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_71) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&xv_36));
                    float cc_72 = comb_coeff[19];
                    value_71 = fma_f32x2_rn_noftz(make_float2(cc_72, cc_72), make_float2(rv_37[0], rv_37[1]), value_71);
                    float cc_73 = comb_coeff[23];
                    value_71 = fma_f32x2_rn_noftz(make_float2(cc_73, cc_73), make_float2(rv_37[2], rv_37[3]), value_71);
                    float cc_74 = comb_coeff[27];
                    value_71 = fma_f32x2_rn_noftz(make_float2(cc_74, cc_74), make_float2(rv_37[4], rv_37[5]), value_71);
                    float cc_75 = comb_coeff[31];
                    value_71 = fma_f32x2_rn_noftz(make_float2(cc_75, cc_75), make_float2(rv_37[6], rv_37[7]), value_71);
                    unsigned int word_76 = __as_u32(__float22bfloat162_rn(make_float2(value_71.x, value_71.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_34] = word_76;
                    words[14] = word_76;
                    float2 rounded_pair_77 = make_float2(__uint_as_float(word_76 << 16), __uint_as_float(word_76 & 4294901760u));
                    float2 squares_78 = fma_f32x2_rn_noftz(rounded_pair_77, rounded_pair_77, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_78.x;
                    sq_42[1] = squares_78.y;
                    float _add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(sq_42[0]), "f"(sq_42[1]));
                    float _sub_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(_add_11), "f"(hc_comp[1]));
                    float _add_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(hc_acc[1]), "f"(_sub_1));
                    float _sub_13;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(_add_1), "f"(hc_acc[1]));
                    float _sub_12;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(_sub_13), "f"(_sub_1));
                    hc_comp[1] = _sub_12;
                    hc_acc[1] = _add_1;
                    unsigned int offset_79 = offs[2];
                    unsigned int x_word_80 = x_pairs[io_stage * 2048 + offset_79];
                    float2 xv_81 = make_float2(__uint_as_float(x_word_80 << 16), __uint_as_float(x_word_80 & 4294901760u));
                    float rv_82[8];
                    unsigned int word_83 = residual_pairs[io_stage * 8192 + offset_79];
                    rv_82[0] = __uint_as_float(word_83 << 16);
                    rv_82[1] = __uint_as_float(word_83 & 4294901760u);
                    unsigned int word_84 = residual_pairs[io_stage * 8192 + 2048 + offset_79];
                    rv_82[2] = __uint_as_float(word_84 << 16);
                    rv_82[3] = __uint_as_float(word_84 & 4294901760u);
                    unsigned int word_85 = residual_pairs[io_stage * 8192 + 4096 + offset_79];
                    rv_82[4] = __uint_as_float(word_85 << 16);
                    rv_82[5] = __uint_as_float(word_85 & 4294901760u);
                    unsigned int word_86 = residual_pairs[io_stage * 8192 + 6144 + offset_79];
                    rv_82[6] = __uint_as_float(word_86 << 16);
                    rv_82[7] = __uint_as_float(word_86 & 4294901760u);
                    float sq_87[2];
                    sq_87[0] = 0.0f;
                    sq_87[1] = 0.0f;
                    float coeff_88 = post_coeff[0];
                    float2 _f32x2_mul_a_14 = make_float2(coeff_88, coeff_88);
                    float2 value_89;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&xv_81));
                    float cc_90 = comb_coeff[0];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_90, cc_90), make_float2(rv_82[0], rv_82[1]), value_89);
                    float cc_91 = comb_coeff[4];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_91, cc_91), make_float2(rv_82[2], rv_82[3]), value_89);
                    float cc_92 = comb_coeff[8];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_92, cc_92), make_float2(rv_82[4], rv_82[5]), value_89);
                    float cc_93 = comb_coeff[12];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_93, cc_93), make_float2(rv_82[6], rv_82[7]), value_89);
                    unsigned int word_94 = __as_u32(__float22bfloat162_rn(make_float2(value_89.x, value_89.y)));
                    residual_pairs[io_stage * 8192 + offset_79] = word_94;
                    words[1] = word_94;
                    float2 rounded_pair_95 = make_float2(__uint_as_float(word_94 << 16), __uint_as_float(word_94 & 4294901760u));
                    float2 squares_96 = fma_f32x2_rn_noftz(rounded_pair_95, rounded_pair_95, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_96.x;
                    sq_87[1] = squares_96.y;
                    float coeff_97 = post_coeff[1];
                    float2 _f32x2_mul_a_15 = make_float2(coeff_97, coeff_97);
                    float2 value_98;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&xv_81));
                    float cc_99 = comb_coeff[1];
                    value_98 = fma_f32x2_rn_noftz(make_float2(cc_99, cc_99), make_float2(rv_82[0], rv_82[1]), value_98);
                    float cc_100 = comb_coeff[5];
                    value_98 = fma_f32x2_rn_noftz(make_float2(cc_100, cc_100), make_float2(rv_82[2], rv_82[3]), value_98);
                    float cc_101 = comb_coeff[9];
                    value_98 = fma_f32x2_rn_noftz(make_float2(cc_101, cc_101), make_float2(rv_82[4], rv_82[5]), value_98);
                    float cc_102 = comb_coeff[13];
                    value_98 = fma_f32x2_rn_noftz(make_float2(cc_102, cc_102), make_float2(rv_82[6], rv_82[7]), value_98);
                    unsigned int word_103 = __as_u32(__float22bfloat162_rn(make_float2(value_98.x, value_98.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_79] = word_103;
                    words[5] = word_103;
                    float2 rounded_pair_104 = make_float2(__uint_as_float(word_103 << 16), __uint_as_float(word_103 & 4294901760u));
                    float2 squares_105 = fma_f32x2_rn_noftz(rounded_pair_104, rounded_pair_104, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_105.x;
                    sq_87[1] = squares_105.y;
                    float coeff_106 = post_coeff[2];
                    float2 _f32x2_mul_a_16 = make_float2(coeff_106, coeff_106);
                    float2 value_107;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_107) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&xv_81));
                    float cc_108 = comb_coeff[2];
                    value_107 = fma_f32x2_rn_noftz(make_float2(cc_108, cc_108), make_float2(rv_82[0], rv_82[1]), value_107);
                    float cc_109 = comb_coeff[6];
                    value_107 = fma_f32x2_rn_noftz(make_float2(cc_109, cc_109), make_float2(rv_82[2], rv_82[3]), value_107);
                    float cc_110 = comb_coeff[10];
                    value_107 = fma_f32x2_rn_noftz(make_float2(cc_110, cc_110), make_float2(rv_82[4], rv_82[5]), value_107);
                    float cc_111 = comb_coeff[14];
                    value_107 = fma_f32x2_rn_noftz(make_float2(cc_111, cc_111), make_float2(rv_82[6], rv_82[7]), value_107);
                    unsigned int word_112 = __as_u32(__float22bfloat162_rn(make_float2(value_107.x, value_107.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_79] = word_112;
                    words[9] = word_112;
                    float2 rounded_pair_113 = make_float2(__uint_as_float(word_112 << 16), __uint_as_float(word_112 & 4294901760u));
                    float2 squares_114 = fma_f32x2_rn_noftz(rounded_pair_113, rounded_pair_113, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_114.x;
                    sq_87[1] = squares_114.y;
                    float coeff_115 = post_coeff[3];
                    float2 _f32x2_mul_a_17 = make_float2(coeff_115, coeff_115);
                    float2 value_116;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&xv_81));
                    float cc_117 = comb_coeff[3];
                    value_116 = fma_f32x2_rn_noftz(make_float2(cc_117, cc_117), make_float2(rv_82[0], rv_82[1]), value_116);
                    float cc_118 = comb_coeff[7];
                    value_116 = fma_f32x2_rn_noftz(make_float2(cc_118, cc_118), make_float2(rv_82[2], rv_82[3]), value_116);
                    float cc_119 = comb_coeff[11];
                    value_116 = fma_f32x2_rn_noftz(make_float2(cc_119, cc_119), make_float2(rv_82[4], rv_82[5]), value_116);
                    float cc_120 = comb_coeff[15];
                    value_116 = fma_f32x2_rn_noftz(make_float2(cc_120, cc_120), make_float2(rv_82[6], rv_82[7]), value_116);
                    unsigned int word_121 = __as_u32(__float22bfloat162_rn(make_float2(value_116.x, value_116.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_79] = word_121;
                    words[13] = word_121;
                    float2 rounded_pair_122 = make_float2(__uint_as_float(word_121 << 16), __uint_as_float(word_121 & 4294901760u));
                    float2 squares_123 = fma_f32x2_rn_noftz(rounded_pair_122, rounded_pair_122, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_123.x;
                    sq_87[1] = squares_123.y;
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(sq_87[0]), "f"(sq_87[1]));
                    float _sub_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_add_18), "f"(hc_comp[0]));
                    float _add_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(hc_acc[0]), "f"(_sub_2));
                    float _sub_20;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(_add_2), "f"(hc_acc[0]));
                    float _sub_19;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(_sub_20), "f"(_sub_2));
                    hc_comp[0] = _sub_19;
                    hc_acc[0] = _add_2;
                    unsigned int offset_124 = offs[3];
                    unsigned int x_word_125 = x_pairs[io_stage * 2048 + offset_124];
                    float2 xv_126 = make_float2(__uint_as_float(x_word_125 << 16), __uint_as_float(x_word_125 & 4294901760u));
                    float rv_127[8];
                    unsigned int word_128 = residual_pairs[io_stage * 8192 + offset_124];
                    rv_127[0] = __uint_as_float(word_128 << 16);
                    rv_127[1] = __uint_as_float(word_128 & 4294901760u);
                    unsigned int word_129 = residual_pairs[io_stage * 8192 + 2048 + offset_124];
                    rv_127[2] = __uint_as_float(word_129 << 16);
                    rv_127[3] = __uint_as_float(word_129 & 4294901760u);
                    unsigned int word_130 = residual_pairs[io_stage * 8192 + 4096 + offset_124];
                    rv_127[4] = __uint_as_float(word_130 << 16);
                    rv_127[5] = __uint_as_float(word_130 & 4294901760u);
                    unsigned int word_131 = residual_pairs[io_stage * 8192 + 6144 + offset_124];
                    rv_127[6] = __uint_as_float(word_131 << 16);
                    rv_127[7] = __uint_as_float(word_131 & 4294901760u);
                    float sq_132[2];
                    sq_132[0] = 0.0f;
                    sq_132[1] = 0.0f;
                    float coeff_133 = post_coeff[4];
                    float2 _f32x2_mul_a_21 = make_float2(coeff_133, coeff_133);
                    float2 value_134;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_134) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&xv_126));
                    float cc_135 = comb_coeff[16];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_135, cc_135), make_float2(rv_127[0], rv_127[1]), value_134);
                    float cc_136 = comb_coeff[20];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_136, cc_136), make_float2(rv_127[2], rv_127[3]), value_134);
                    float cc_137 = comb_coeff[24];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_137, cc_137), make_float2(rv_127[4], rv_127[5]), value_134);
                    float cc_138 = comb_coeff[28];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_138, cc_138), make_float2(rv_127[6], rv_127[7]), value_134);
                    unsigned int word_139 = __as_u32(__float22bfloat162_rn(make_float2(value_134.x, value_134.y)));
                    residual_pairs[io_stage * 8192 + offset_124] = word_139;
                    words[3] = word_139;
                    float2 rounded_pair_140 = make_float2(__uint_as_float(word_139 << 16), __uint_as_float(word_139 & 4294901760u));
                    float2 squares_141 = fma_f32x2_rn_noftz(rounded_pair_140, rounded_pair_140, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_141.x;
                    sq_132[1] = squares_141.y;
                    float coeff_142 = post_coeff[5];
                    float2 _f32x2_mul_a_22 = make_float2(coeff_142, coeff_142);
                    float2 value_143;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_143) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&xv_126));
                    float cc_144 = comb_coeff[17];
                    value_143 = fma_f32x2_rn_noftz(make_float2(cc_144, cc_144), make_float2(rv_127[0], rv_127[1]), value_143);
                    float cc_145 = comb_coeff[21];
                    value_143 = fma_f32x2_rn_noftz(make_float2(cc_145, cc_145), make_float2(rv_127[2], rv_127[3]), value_143);
                    float cc_146 = comb_coeff[25];
                    value_143 = fma_f32x2_rn_noftz(make_float2(cc_146, cc_146), make_float2(rv_127[4], rv_127[5]), value_143);
                    float cc_147 = comb_coeff[29];
                    value_143 = fma_f32x2_rn_noftz(make_float2(cc_147, cc_147), make_float2(rv_127[6], rv_127[7]), value_143);
                    unsigned int word_148 = __as_u32(__float22bfloat162_rn(make_float2(value_143.x, value_143.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_124] = word_148;
                    words[7] = word_148;
                    float2 rounded_pair_149 = make_float2(__uint_as_float(word_148 << 16), __uint_as_float(word_148 & 4294901760u));
                    float2 squares_150 = fma_f32x2_rn_noftz(rounded_pair_149, rounded_pair_149, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_150.x;
                    sq_132[1] = squares_150.y;
                    float coeff_151 = post_coeff[6];
                    float2 _f32x2_mul_a_23 = make_float2(coeff_151, coeff_151);
                    float2 value_152;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_152) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&xv_126));
                    float cc_153 = comb_coeff[18];
                    value_152 = fma_f32x2_rn_noftz(make_float2(cc_153, cc_153), make_float2(rv_127[0], rv_127[1]), value_152);
                    float cc_154 = comb_coeff[22];
                    value_152 = fma_f32x2_rn_noftz(make_float2(cc_154, cc_154), make_float2(rv_127[2], rv_127[3]), value_152);
                    float cc_155 = comb_coeff[26];
                    value_152 = fma_f32x2_rn_noftz(make_float2(cc_155, cc_155), make_float2(rv_127[4], rv_127[5]), value_152);
                    float cc_156 = comb_coeff[30];
                    value_152 = fma_f32x2_rn_noftz(make_float2(cc_156, cc_156), make_float2(rv_127[6], rv_127[7]), value_152);
                    unsigned int word_157 = __as_u32(__float22bfloat162_rn(make_float2(value_152.x, value_152.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_124] = word_157;
                    words[11] = word_157;
                    float2 rounded_pair_158 = make_float2(__uint_as_float(word_157 << 16), __uint_as_float(word_157 & 4294901760u));
                    float2 squares_159 = fma_f32x2_rn_noftz(rounded_pair_158, rounded_pair_158, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_159.x;
                    sq_132[1] = squares_159.y;
                    float coeff_160 = post_coeff[7];
                    float2 _f32x2_mul_a_24 = make_float2(coeff_160, coeff_160);
                    float2 value_161;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_161) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&xv_126));
                    float cc_162 = comb_coeff[19];
                    value_161 = fma_f32x2_rn_noftz(make_float2(cc_162, cc_162), make_float2(rv_127[0], rv_127[1]), value_161);
                    float cc_163 = comb_coeff[23];
                    value_161 = fma_f32x2_rn_noftz(make_float2(cc_163, cc_163), make_float2(rv_127[2], rv_127[3]), value_161);
                    float cc_164 = comb_coeff[27];
                    value_161 = fma_f32x2_rn_noftz(make_float2(cc_164, cc_164), make_float2(rv_127[4], rv_127[5]), value_161);
                    float cc_165 = comb_coeff[31];
                    value_161 = fma_f32x2_rn_noftz(make_float2(cc_165, cc_165), make_float2(rv_127[6], rv_127[7]), value_161);
                    unsigned int word_166 = __as_u32(__float22bfloat162_rn(make_float2(value_161.x, value_161.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_124] = word_166;
                    words[15] = word_166;
                    float2 rounded_pair_167 = make_float2(__uint_as_float(word_166 << 16), __uint_as_float(word_166 & 4294901760u));
                    float2 squares_168 = fma_f32x2_rn_noftz(rounded_pair_167, rounded_pair_167, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_168.x;
                    sq_132[1] = squares_168.y;
                    float _add_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(sq_132[0]), "f"(sq_132[1]));
                    float _sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_add_25), "f"(hc_comp[1]));
                    float _add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(hc_acc[1]), "f"(_sub_3));
                    float _sub_27;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(_add_3), "f"(hc_acc[1]));
                    float _sub_26;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(_sub_27), "f"(_sub_3));
                    hc_comp[1] = _sub_26;
                    hc_acc[1] = _add_3;
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
                    unsigned int word_175 = residual_pairs[io_stage * 8192 + 2048 + offset_170];
                    rv_173[2] = __uint_as_float(word_175 << 16);
                    rv_173[3] = __uint_as_float(word_175 & 4294901760u);
                    unsigned int word_176 = residual_pairs[io_stage * 8192 + 4096 + offset_170];
                    rv_173[4] = __uint_as_float(word_176 << 16);
                    rv_173[5] = __uint_as_float(word_176 & 4294901760u);
                    unsigned int word_177 = residual_pairs[io_stage * 8192 + 6144 + offset_170];
                    rv_173[6] = __uint_as_float(word_177 << 16);
                    rv_173[7] = __uint_as_float(word_177 & 4294901760u);
                    float sq_178[2];
                    sq_178[0] = 0.0f;
                    sq_178[1] = 0.0f;
                    float coeff_179 = post_coeff[0];
                    float2 _f32x2_mul_a_28 = make_float2(coeff_179, coeff_179);
                    float2 value_180;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_180) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&xv_172));
                    float cc_181 = comb_coeff[0];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_181, cc_181), make_float2(rv_173[0], rv_173[1]), value_180);
                    float cc_182 = comb_coeff[4];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_182, cc_182), make_float2(rv_173[2], rv_173[3]), value_180);
                    float cc_183 = comb_coeff[8];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_183, cc_183), make_float2(rv_173[4], rv_173[5]), value_180);
                    float cc_184 = comb_coeff[12];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_184, cc_184), make_float2(rv_173[6], rv_173[7]), value_180);
                    unsigned int word_185 = __as_u32(__float22bfloat162_rn(make_float2(value_180.x, value_180.y)));
                    residual_pairs[io_stage * 8192 + offset_170] = word_185;
                    words_169[0] = word_185;
                    float2 rounded_pair_186 = make_float2(__uint_as_float(word_185 << 16), __uint_as_float(word_185 & 4294901760u));
                    float2 squares_187 = fma_f32x2_rn_noftz(rounded_pair_186, rounded_pair_186, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_187.x;
                    sq_178[1] = squares_187.y;
                    float coeff_188 = post_coeff[1];
                    float2 _f32x2_mul_a_29 = make_float2(coeff_188, coeff_188);
                    float2 value_189;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_189) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&xv_172));
                    float cc_190 = comb_coeff[1];
                    value_189 = fma_f32x2_rn_noftz(make_float2(cc_190, cc_190), make_float2(rv_173[0], rv_173[1]), value_189);
                    float cc_191 = comb_coeff[5];
                    value_189 = fma_f32x2_rn_noftz(make_float2(cc_191, cc_191), make_float2(rv_173[2], rv_173[3]), value_189);
                    float cc_192 = comb_coeff[9];
                    value_189 = fma_f32x2_rn_noftz(make_float2(cc_192, cc_192), make_float2(rv_173[4], rv_173[5]), value_189);
                    float cc_193 = comb_coeff[13];
                    value_189 = fma_f32x2_rn_noftz(make_float2(cc_193, cc_193), make_float2(rv_173[6], rv_173[7]), value_189);
                    unsigned int word_194 = __as_u32(__float22bfloat162_rn(make_float2(value_189.x, value_189.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_170] = word_194;
                    words_169[4] = word_194;
                    float2 rounded_pair_195 = make_float2(__uint_as_float(word_194 << 16), __uint_as_float(word_194 & 4294901760u));
                    float2 squares_196 = fma_f32x2_rn_noftz(rounded_pair_195, rounded_pair_195, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_196.x;
                    sq_178[1] = squares_196.y;
                    float coeff_197 = post_coeff[2];
                    float2 _f32x2_mul_a_30 = make_float2(coeff_197, coeff_197);
                    float2 value_198;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_198) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&xv_172));
                    float cc_199 = comb_coeff[2];
                    value_198 = fma_f32x2_rn_noftz(make_float2(cc_199, cc_199), make_float2(rv_173[0], rv_173[1]), value_198);
                    float cc_200 = comb_coeff[6];
                    value_198 = fma_f32x2_rn_noftz(make_float2(cc_200, cc_200), make_float2(rv_173[2], rv_173[3]), value_198);
                    float cc_201 = comb_coeff[10];
                    value_198 = fma_f32x2_rn_noftz(make_float2(cc_201, cc_201), make_float2(rv_173[4], rv_173[5]), value_198);
                    float cc_202 = comb_coeff[14];
                    value_198 = fma_f32x2_rn_noftz(make_float2(cc_202, cc_202), make_float2(rv_173[6], rv_173[7]), value_198);
                    unsigned int word_203 = __as_u32(__float22bfloat162_rn(make_float2(value_198.x, value_198.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_170] = word_203;
                    words_169[8] = word_203;
                    float2 rounded_pair_204 = make_float2(__uint_as_float(word_203 << 16), __uint_as_float(word_203 & 4294901760u));
                    float2 squares_205 = fma_f32x2_rn_noftz(rounded_pair_204, rounded_pair_204, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_205.x;
                    sq_178[1] = squares_205.y;
                    float coeff_206 = post_coeff[3];
                    float2 _f32x2_mul_a_31 = make_float2(coeff_206, coeff_206);
                    float2 value_207;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_207) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&xv_172));
                    float cc_208 = comb_coeff[3];
                    value_207 = fma_f32x2_rn_noftz(make_float2(cc_208, cc_208), make_float2(rv_173[0], rv_173[1]), value_207);
                    float cc_209 = comb_coeff[7];
                    value_207 = fma_f32x2_rn_noftz(make_float2(cc_209, cc_209), make_float2(rv_173[2], rv_173[3]), value_207);
                    float cc_210 = comb_coeff[11];
                    value_207 = fma_f32x2_rn_noftz(make_float2(cc_210, cc_210), make_float2(rv_173[4], rv_173[5]), value_207);
                    float cc_211 = comb_coeff[15];
                    value_207 = fma_f32x2_rn_noftz(make_float2(cc_211, cc_211), make_float2(rv_173[6], rv_173[7]), value_207);
                    unsigned int word_212 = __as_u32(__float22bfloat162_rn(make_float2(value_207.x, value_207.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_170] = word_212;
                    words_169[12] = word_212;
                    float2 rounded_pair_213 = make_float2(__uint_as_float(word_212 << 16), __uint_as_float(word_212 & 4294901760u));
                    float2 squares_214 = fma_f32x2_rn_noftz(rounded_pair_213, rounded_pair_213, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_214.x;
                    sq_178[1] = squares_214.y;
                    float _add_32;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(sq_178[0]), "f"(sq_178[1]));
                    float _sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(_add_32), "f"(hc_comp[0]));
                    float _add_4_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4_1) : "f"(hc_acc[0]), "f"(_sub_4));
                    float _sub_34;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(_add_4_1), "f"(hc_acc[0]));
                    float _sub_33;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(_sub_34), "f"(_sub_4));
                    hc_comp[0] = _sub_33;
                    hc_acc[0] = _add_4_1;
                    unsigned int offset_215 = offs[5];
                    unsigned int x_word_216 = x_pairs[io_stage * 2048 + offset_215];
                    float2 xv_217 = make_float2(__uint_as_float(x_word_216 << 16), __uint_as_float(x_word_216 & 4294901760u));
                    float rv_218[8];
                    unsigned int word_219 = residual_pairs[io_stage * 8192 + offset_215];
                    rv_218[0] = __uint_as_float(word_219 << 16);
                    rv_218[1] = __uint_as_float(word_219 & 4294901760u);
                    unsigned int word_220 = residual_pairs[io_stage * 8192 + 2048 + offset_215];
                    rv_218[2] = __uint_as_float(word_220 << 16);
                    rv_218[3] = __uint_as_float(word_220 & 4294901760u);
                    unsigned int word_221 = residual_pairs[io_stage * 8192 + 4096 + offset_215];
                    rv_218[4] = __uint_as_float(word_221 << 16);
                    rv_218[5] = __uint_as_float(word_221 & 4294901760u);
                    unsigned int word_222 = residual_pairs[io_stage * 8192 + 6144 + offset_215];
                    rv_218[6] = __uint_as_float(word_222 << 16);
                    rv_218[7] = __uint_as_float(word_222 & 4294901760u);
                    float sq_223[2];
                    sq_223[0] = 0.0f;
                    sq_223[1] = 0.0f;
                    float coeff_224 = post_coeff[4];
                    float2 _f32x2_mul_a_35 = make_float2(coeff_224, coeff_224);
                    float2 value_225;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_225) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&xv_217));
                    float cc_226 = comb_coeff[16];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_226, cc_226), make_float2(rv_218[0], rv_218[1]), value_225);
                    float cc_227 = comb_coeff[20];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_227, cc_227), make_float2(rv_218[2], rv_218[3]), value_225);
                    float cc_228 = comb_coeff[24];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_228, cc_228), make_float2(rv_218[4], rv_218[5]), value_225);
                    float cc_229 = comb_coeff[28];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_229, cc_229), make_float2(rv_218[6], rv_218[7]), value_225);
                    unsigned int word_230 = __as_u32(__float22bfloat162_rn(make_float2(value_225.x, value_225.y)));
                    residual_pairs[io_stage * 8192 + offset_215] = word_230;
                    words_169[2] = word_230;
                    float2 rounded_pair_231 = make_float2(__uint_as_float(word_230 << 16), __uint_as_float(word_230 & 4294901760u));
                    float2 squares_232 = fma_f32x2_rn_noftz(rounded_pair_231, rounded_pair_231, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_232.x;
                    sq_223[1] = squares_232.y;
                    float coeff_233 = post_coeff[5];
                    float2 _f32x2_mul_a_36 = make_float2(coeff_233, coeff_233);
                    float2 value_234;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_234) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&xv_217));
                    float cc_235 = comb_coeff[17];
                    value_234 = fma_f32x2_rn_noftz(make_float2(cc_235, cc_235), make_float2(rv_218[0], rv_218[1]), value_234);
                    float cc_236 = comb_coeff[21];
                    value_234 = fma_f32x2_rn_noftz(make_float2(cc_236, cc_236), make_float2(rv_218[2], rv_218[3]), value_234);
                    float cc_237 = comb_coeff[25];
                    value_234 = fma_f32x2_rn_noftz(make_float2(cc_237, cc_237), make_float2(rv_218[4], rv_218[5]), value_234);
                    float cc_238 = comb_coeff[29];
                    value_234 = fma_f32x2_rn_noftz(make_float2(cc_238, cc_238), make_float2(rv_218[6], rv_218[7]), value_234);
                    unsigned int word_239 = __as_u32(__float22bfloat162_rn(make_float2(value_234.x, value_234.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_215] = word_239;
                    words_169[6] = word_239;
                    float2 rounded_pair_240 = make_float2(__uint_as_float(word_239 << 16), __uint_as_float(word_239 & 4294901760u));
                    float2 squares_241 = fma_f32x2_rn_noftz(rounded_pair_240, rounded_pair_240, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_241.x;
                    sq_223[1] = squares_241.y;
                    float coeff_242 = post_coeff[6];
                    float2 _f32x2_mul_a_37 = make_float2(coeff_242, coeff_242);
                    float2 value_243;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_243) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&xv_217));
                    float cc_244 = comb_coeff[18];
                    value_243 = fma_f32x2_rn_noftz(make_float2(cc_244, cc_244), make_float2(rv_218[0], rv_218[1]), value_243);
                    float cc_245 = comb_coeff[22];
                    value_243 = fma_f32x2_rn_noftz(make_float2(cc_245, cc_245), make_float2(rv_218[2], rv_218[3]), value_243);
                    float cc_246 = comb_coeff[26];
                    value_243 = fma_f32x2_rn_noftz(make_float2(cc_246, cc_246), make_float2(rv_218[4], rv_218[5]), value_243);
                    float cc_247 = comb_coeff[30];
                    value_243 = fma_f32x2_rn_noftz(make_float2(cc_247, cc_247), make_float2(rv_218[6], rv_218[7]), value_243);
                    unsigned int word_248 = __as_u32(__float22bfloat162_rn(make_float2(value_243.x, value_243.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_215] = word_248;
                    words_169[10] = word_248;
                    float2 rounded_pair_249 = make_float2(__uint_as_float(word_248 << 16), __uint_as_float(word_248 & 4294901760u));
                    float2 squares_250 = fma_f32x2_rn_noftz(rounded_pair_249, rounded_pair_249, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_250.x;
                    sq_223[1] = squares_250.y;
                    float coeff_251 = post_coeff[7];
                    float2 _f32x2_mul_a_38 = make_float2(coeff_251, coeff_251);
                    float2 value_252;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_252) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&xv_217));
                    float cc_253 = comb_coeff[19];
                    value_252 = fma_f32x2_rn_noftz(make_float2(cc_253, cc_253), make_float2(rv_218[0], rv_218[1]), value_252);
                    float cc_254 = comb_coeff[23];
                    value_252 = fma_f32x2_rn_noftz(make_float2(cc_254, cc_254), make_float2(rv_218[2], rv_218[3]), value_252);
                    float cc_255 = comb_coeff[27];
                    value_252 = fma_f32x2_rn_noftz(make_float2(cc_255, cc_255), make_float2(rv_218[4], rv_218[5]), value_252);
                    float cc_256 = comb_coeff[31];
                    value_252 = fma_f32x2_rn_noftz(make_float2(cc_256, cc_256), make_float2(rv_218[6], rv_218[7]), value_252);
                    unsigned int word_257 = __as_u32(__float22bfloat162_rn(make_float2(value_252.x, value_252.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_215] = word_257;
                    words_169[14] = word_257;
                    float2 rounded_pair_258 = make_float2(__uint_as_float(word_257 << 16), __uint_as_float(word_257 & 4294901760u));
                    float2 squares_259 = fma_f32x2_rn_noftz(rounded_pair_258, rounded_pair_258, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_259.x;
                    sq_223[1] = squares_259.y;
                    float _add_39;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(sq_223[0]), "f"(sq_223[1]));
                    float _sub_5_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_1) : "f"(_add_39), "f"(hc_comp[1]));
                    float _add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(hc_acc[1]), "f"(_sub_5_1));
                    float _sub_41;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(_add_5), "f"(hc_acc[1]));
                    float _sub_40;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(_sub_41), "f"(_sub_5_1));
                    hc_comp[1] = _sub_40;
                    hc_acc[1] = _add_5;
                    unsigned int offset_260 = offs[6];
                    unsigned int x_word_261 = x_pairs[io_stage * 2048 + offset_260];
                    float2 xv_262 = make_float2(__uint_as_float(x_word_261 << 16), __uint_as_float(x_word_261 & 4294901760u));
                    float rv_263[8];
                    unsigned int word_264 = residual_pairs[io_stage * 8192 + offset_260];
                    rv_263[0] = __uint_as_float(word_264 << 16);
                    rv_263[1] = __uint_as_float(word_264 & 4294901760u);
                    unsigned int word_265 = residual_pairs[io_stage * 8192 + 2048 + offset_260];
                    rv_263[2] = __uint_as_float(word_265 << 16);
                    rv_263[3] = __uint_as_float(word_265 & 4294901760u);
                    unsigned int word_266 = residual_pairs[io_stage * 8192 + 4096 + offset_260];
                    rv_263[4] = __uint_as_float(word_266 << 16);
                    rv_263[5] = __uint_as_float(word_266 & 4294901760u);
                    unsigned int word_267 = residual_pairs[io_stage * 8192 + 6144 + offset_260];
                    rv_263[6] = __uint_as_float(word_267 << 16);
                    rv_263[7] = __uint_as_float(word_267 & 4294901760u);
                    float sq_268[2];
                    sq_268[0] = 0.0f;
                    sq_268[1] = 0.0f;
                    float coeff_269 = post_coeff[0];
                    float2 _f32x2_mul_a_42 = make_float2(coeff_269, coeff_269);
                    float2 value_270;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_270) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&xv_262));
                    float cc_271 = comb_coeff[0];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_271, cc_271), make_float2(rv_263[0], rv_263[1]), value_270);
                    float cc_272 = comb_coeff[4];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_272, cc_272), make_float2(rv_263[2], rv_263[3]), value_270);
                    float cc_273 = comb_coeff[8];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_273, cc_273), make_float2(rv_263[4], rv_263[5]), value_270);
                    float cc_274 = comb_coeff[12];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_274, cc_274), make_float2(rv_263[6], rv_263[7]), value_270);
                    unsigned int word_275 = __as_u32(__float22bfloat162_rn(make_float2(value_270.x, value_270.y)));
                    residual_pairs[io_stage * 8192 + offset_260] = word_275;
                    words_169[1] = word_275;
                    float2 rounded_pair_276 = make_float2(__uint_as_float(word_275 << 16), __uint_as_float(word_275 & 4294901760u));
                    float2 squares_277 = fma_f32x2_rn_noftz(rounded_pair_276, rounded_pair_276, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_277.x;
                    sq_268[1] = squares_277.y;
                    float coeff_278 = post_coeff[1];
                    float2 _f32x2_mul_a_43 = make_float2(coeff_278, coeff_278);
                    float2 value_279;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_279) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&xv_262));
                    float cc_280 = comb_coeff[1];
                    value_279 = fma_f32x2_rn_noftz(make_float2(cc_280, cc_280), make_float2(rv_263[0], rv_263[1]), value_279);
                    float cc_281 = comb_coeff[5];
                    value_279 = fma_f32x2_rn_noftz(make_float2(cc_281, cc_281), make_float2(rv_263[2], rv_263[3]), value_279);
                    float cc_282 = comb_coeff[9];
                    value_279 = fma_f32x2_rn_noftz(make_float2(cc_282, cc_282), make_float2(rv_263[4], rv_263[5]), value_279);
                    float cc_283 = comb_coeff[13];
                    value_279 = fma_f32x2_rn_noftz(make_float2(cc_283, cc_283), make_float2(rv_263[6], rv_263[7]), value_279);
                    unsigned int word_284 = __as_u32(__float22bfloat162_rn(make_float2(value_279.x, value_279.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_260] = word_284;
                    words_169[5] = word_284;
                    float2 rounded_pair_285 = make_float2(__uint_as_float(word_284 << 16), __uint_as_float(word_284 & 4294901760u));
                    float2 squares_286 = fma_f32x2_rn_noftz(rounded_pair_285, rounded_pair_285, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_286.x;
                    sq_268[1] = squares_286.y;
                    float coeff_287 = post_coeff[2];
                    float2 _f32x2_mul_a_44 = make_float2(coeff_287, coeff_287);
                    float2 value_288;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_288) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&xv_262));
                    float cc_289 = comb_coeff[2];
                    value_288 = fma_f32x2_rn_noftz(make_float2(cc_289, cc_289), make_float2(rv_263[0], rv_263[1]), value_288);
                    float cc_290 = comb_coeff[6];
                    value_288 = fma_f32x2_rn_noftz(make_float2(cc_290, cc_290), make_float2(rv_263[2], rv_263[3]), value_288);
                    float cc_291 = comb_coeff[10];
                    value_288 = fma_f32x2_rn_noftz(make_float2(cc_291, cc_291), make_float2(rv_263[4], rv_263[5]), value_288);
                    float cc_292 = comb_coeff[14];
                    value_288 = fma_f32x2_rn_noftz(make_float2(cc_292, cc_292), make_float2(rv_263[6], rv_263[7]), value_288);
                    unsigned int word_293 = __as_u32(__float22bfloat162_rn(make_float2(value_288.x, value_288.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_260] = word_293;
                    words_169[9] = word_293;
                    float2 rounded_pair_294 = make_float2(__uint_as_float(word_293 << 16), __uint_as_float(word_293 & 4294901760u));
                    float2 squares_295 = fma_f32x2_rn_noftz(rounded_pair_294, rounded_pair_294, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_295.x;
                    sq_268[1] = squares_295.y;
                    float coeff_296 = post_coeff[3];
                    float2 _f32x2_mul_a_45 = make_float2(coeff_296, coeff_296);
                    float2 value_297;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_297) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&xv_262));
                    float cc_298 = comb_coeff[3];
                    value_297 = fma_f32x2_rn_noftz(make_float2(cc_298, cc_298), make_float2(rv_263[0], rv_263[1]), value_297);
                    float cc_299 = comb_coeff[7];
                    value_297 = fma_f32x2_rn_noftz(make_float2(cc_299, cc_299), make_float2(rv_263[2], rv_263[3]), value_297);
                    float cc_300 = comb_coeff[11];
                    value_297 = fma_f32x2_rn_noftz(make_float2(cc_300, cc_300), make_float2(rv_263[4], rv_263[5]), value_297);
                    float cc_301 = comb_coeff[15];
                    value_297 = fma_f32x2_rn_noftz(make_float2(cc_301, cc_301), make_float2(rv_263[6], rv_263[7]), value_297);
                    unsigned int word_302 = __as_u32(__float22bfloat162_rn(make_float2(value_297.x, value_297.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_260] = word_302;
                    words_169[13] = word_302;
                    float2 rounded_pair_303 = make_float2(__uint_as_float(word_302 << 16), __uint_as_float(word_302 & 4294901760u));
                    float2 squares_304 = fma_f32x2_rn_noftz(rounded_pair_303, rounded_pair_303, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_304.x;
                    sq_268[1] = squares_304.y;
                    float _add_46;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(sq_268[0]), "f"(sq_268[1]));
                    float _sub_6_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_1) : "f"(_add_46), "f"(hc_comp[0]));
                    float _add_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(hc_acc[0]), "f"(_sub_6_1));
                    float _sub_48;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(_add_6), "f"(hc_acc[0]));
                    float _sub_47;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(_sub_48), "f"(_sub_6_1));
                    hc_comp[0] = _sub_47;
                    hc_acc[0] = _add_6;
                    unsigned int offset_305 = offs[7];
                    unsigned int x_word_306 = x_pairs[io_stage * 2048 + offset_305];
                    float2 xv_307 = make_float2(__uint_as_float(x_word_306 << 16), __uint_as_float(x_word_306 & 4294901760u));
                    float rv_308[8];
                    unsigned int word_309 = residual_pairs[io_stage * 8192 + offset_305];
                    rv_308[0] = __uint_as_float(word_309 << 16);
                    rv_308[1] = __uint_as_float(word_309 & 4294901760u);
                    unsigned int word_310 = residual_pairs[io_stage * 8192 + 2048 + offset_305];
                    rv_308[2] = __uint_as_float(word_310 << 16);
                    rv_308[3] = __uint_as_float(word_310 & 4294901760u);
                    unsigned int word_311 = residual_pairs[io_stage * 8192 + 4096 + offset_305];
                    rv_308[4] = __uint_as_float(word_311 << 16);
                    rv_308[5] = __uint_as_float(word_311 & 4294901760u);
                    unsigned int word_312 = residual_pairs[io_stage * 8192 + 6144 + offset_305];
                    rv_308[6] = __uint_as_float(word_312 << 16);
                    rv_308[7] = __uint_as_float(word_312 & 4294901760u);
                    float sq_313[2];
                    sq_313[0] = 0.0f;
                    sq_313[1] = 0.0f;
                    float coeff_314 = post_coeff[4];
                    float2 _f32x2_mul_a_49 = make_float2(coeff_314, coeff_314);
                    float2 value_315;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_315) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&xv_307));
                    float cc_316 = comb_coeff[16];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_316, cc_316), make_float2(rv_308[0], rv_308[1]), value_315);
                    float cc_317 = comb_coeff[20];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_317, cc_317), make_float2(rv_308[2], rv_308[3]), value_315);
                    float cc_318 = comb_coeff[24];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_318, cc_318), make_float2(rv_308[4], rv_308[5]), value_315);
                    float cc_319 = comb_coeff[28];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_319, cc_319), make_float2(rv_308[6], rv_308[7]), value_315);
                    unsigned int word_320 = __as_u32(__float22bfloat162_rn(make_float2(value_315.x, value_315.y)));
                    residual_pairs[io_stage * 8192 + offset_305] = word_320;
                    words_169[3] = word_320;
                    float2 rounded_pair_321 = make_float2(__uint_as_float(word_320 << 16), __uint_as_float(word_320 & 4294901760u));
                    float2 squares_322 = fma_f32x2_rn_noftz(rounded_pair_321, rounded_pair_321, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_322.x;
                    sq_313[1] = squares_322.y;
                    float coeff_323 = post_coeff[5];
                    float2 _f32x2_mul_a_50 = make_float2(coeff_323, coeff_323);
                    float2 value_324;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_324) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&xv_307));
                    float cc_325 = comb_coeff[17];
                    value_324 = fma_f32x2_rn_noftz(make_float2(cc_325, cc_325), make_float2(rv_308[0], rv_308[1]), value_324);
                    float cc_326 = comb_coeff[21];
                    value_324 = fma_f32x2_rn_noftz(make_float2(cc_326, cc_326), make_float2(rv_308[2], rv_308[3]), value_324);
                    float cc_327 = comb_coeff[25];
                    value_324 = fma_f32x2_rn_noftz(make_float2(cc_327, cc_327), make_float2(rv_308[4], rv_308[5]), value_324);
                    float cc_328 = comb_coeff[29];
                    value_324 = fma_f32x2_rn_noftz(make_float2(cc_328, cc_328), make_float2(rv_308[6], rv_308[7]), value_324);
                    unsigned int word_329 = __as_u32(__float22bfloat162_rn(make_float2(value_324.x, value_324.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_305] = word_329;
                    words_169[7] = word_329;
                    float2 rounded_pair_330 = make_float2(__uint_as_float(word_329 << 16), __uint_as_float(word_329 & 4294901760u));
                    float2 squares_331 = fma_f32x2_rn_noftz(rounded_pair_330, rounded_pair_330, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_331.x;
                    sq_313[1] = squares_331.y;
                    float coeff_332 = post_coeff[6];
                    float2 _f32x2_mul_a_51 = make_float2(coeff_332, coeff_332);
                    float2 value_333;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_333) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&xv_307));
                    float cc_334 = comb_coeff[18];
                    value_333 = fma_f32x2_rn_noftz(make_float2(cc_334, cc_334), make_float2(rv_308[0], rv_308[1]), value_333);
                    float cc_335 = comb_coeff[22];
                    value_333 = fma_f32x2_rn_noftz(make_float2(cc_335, cc_335), make_float2(rv_308[2], rv_308[3]), value_333);
                    float cc_336 = comb_coeff[26];
                    value_333 = fma_f32x2_rn_noftz(make_float2(cc_336, cc_336), make_float2(rv_308[4], rv_308[5]), value_333);
                    float cc_337 = comb_coeff[30];
                    value_333 = fma_f32x2_rn_noftz(make_float2(cc_337, cc_337), make_float2(rv_308[6], rv_308[7]), value_333);
                    unsigned int word_338 = __as_u32(__float22bfloat162_rn(make_float2(value_333.x, value_333.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_305] = word_338;
                    words_169[11] = word_338;
                    float2 rounded_pair_339 = make_float2(__uint_as_float(word_338 << 16), __uint_as_float(word_338 & 4294901760u));
                    float2 squares_340 = fma_f32x2_rn_noftz(rounded_pair_339, rounded_pair_339, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_340.x;
                    sq_313[1] = squares_340.y;
                    float coeff_341 = post_coeff[7];
                    float2 _f32x2_mul_a_52 = make_float2(coeff_341, coeff_341);
                    float2 value_342;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_342) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&xv_307));
                    float cc_343 = comb_coeff[19];
                    value_342 = fma_f32x2_rn_noftz(make_float2(cc_343, cc_343), make_float2(rv_308[0], rv_308[1]), value_342);
                    float cc_344 = comb_coeff[23];
                    value_342 = fma_f32x2_rn_noftz(make_float2(cc_344, cc_344), make_float2(rv_308[2], rv_308[3]), value_342);
                    float cc_345 = comb_coeff[27];
                    value_342 = fma_f32x2_rn_noftz(make_float2(cc_345, cc_345), make_float2(rv_308[4], rv_308[5]), value_342);
                    float cc_346 = comb_coeff[31];
                    value_342 = fma_f32x2_rn_noftz(make_float2(cc_346, cc_346), make_float2(rv_308[6], rv_308[7]), value_342);
                    unsigned int word_347 = __as_u32(__float22bfloat162_rn(make_float2(value_342.x, value_342.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_305] = word_347;
                    words_169[15] = word_347;
                    float2 rounded_pair_348 = make_float2(__uint_as_float(word_347 << 16), __uint_as_float(word_347 & 4294901760u));
                    float2 squares_349 = fma_f32x2_rn_noftz(rounded_pair_348, rounded_pair_348, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_349.x;
                    sq_313[1] = squares_349.y;
                    float _add_53;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(sq_313[0]), "f"(sq_313[1]));
                    float _sub_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(_add_53), "f"(hc_comp[1]));
                    float _add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(hc_acc[1]), "f"(_sub_7));
                    float _sub_55;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(_add_7), "f"(hc_acc[1]));
                    float _sub_54;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(_sub_55), "f"(_sub_7));
                    hc_comp[1] = _sub_54;
                    hc_acc[1] = _add_7;
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
                float _sub_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(hc_acc[0]), "f"(hc_comp[0]));
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, _sub_8, 2);
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(_sub_8), "f"(_shfl_xor_sync_56));
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, _add_8, 1);
                float _add_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(_add_8), "f"(_shfl_xor_sync_57));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[worker * 64 + row_1_1] = _add_9;
                }
                unsigned int row_4 = first_row + 8;
                float _sub_9;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(hc_acc[1]), "f"(hc_comp[1]));
                float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, _sub_9, 2);
                float _add_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_sub_9), "f"(_shfl_xor_sync_58));
                float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, _add_10, 1);
                float _add_11_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11_1) : "f"(_add_10), "f"(_shfl_xor_sync_59));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[worker * 64 + row_4] = _add_11_1;
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
                uint32_t _fast_div_q_7 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_4), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_4);
                uint32_t _fast_div_r_7 = (uint32_t)(task_4) - _fast_div_q_7 * (uint32_t)(split_div.divisor);
                if (warp % 4 == 0) {
                    unsigned long long base = (launch_epochs[bid] + 1) * 128;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + _fast_div_q_7) * 16))) : "memory");
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
                for (unsigned int chunk = 0; chunk < (_fast_div_q_0 + ((_fast_div_r_7 < _fast_div_r_0) ? 1 : 0)) * 2; chunk++) {
                    unsigned int cstage_2 = cseq_1 % 2;
                    unsigned int _stage_big_full = cseq_1 % 2;
                    unsigned int _parity_big_full = cseq_1 / 2 % 2;
                    mbarrier_wait(big_full_addr + (_stage_big_full) * 8, _parity_big_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
#if __CUDA_ARCH__ == 1070
                        : "r"((unsigned int)tmem_acc_big + cstage_2 * 24));
#else
                        : "r"((unsigned int)tmem_acc_big + cstage_2 * 48));
#endif
                    float _tmem_load_1[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
#if __CUDA_ARCH__ == 1070
                        : "r"((unsigned int)tmem_acc_big + (cstage_2 * 24 + 16)));
#else
                        : "r"((unsigned int)tmem_acc_big + (cstage_2 * 48 + 16)));
                    float _tmem_load_2[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
                        : "r"((unsigned int)tmem_acc_big + (cstage_2 * 48 + 24)));
                    float _tmem_load_3[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
                        : "r"((unsigned int)tmem_acc_big + (cstage_2 * 48 + 24 + 16)));
#endif
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(big_empty_addr + (cstage_2) * 8);
                    }
                    float _sub_10;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(_tmem_load_0[0]), "f"(comp[0]));
                    float _add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(acc[0]), "f"(_sub_10));
                    float _sub_1_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1_1) : "f"(_add_12), "f"(acc[0]));
                    float _sub_0_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0_1) : "f"(_sub_1_1), "f"(_sub_10));
                    comp[0] = _sub_0_1;
                    acc[0] = _add_12;
                    float _sub_11;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(_tmem_load_0[1]), "f"(comp[1]));
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(acc[1]), "f"(_sub_11));
                    float _sub_3_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3_1) : "f"(_add_13), "f"(acc[1]));
                    float _sub_2_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2_1) : "f"(_sub_3_1), "f"(_sub_11));
                    comp[1] = _sub_2_1;
                    acc[1] = _add_13;
                    float _sub_12_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_1) : "f"(_tmem_load_0[2]), "f"(comp[2]));
                    float _add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(acc[2]), "f"(_sub_12_1));
                    float _sub_5_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_2) : "f"(_add_14), "f"(acc[2]));
                    float _sub_4_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4_1) : "f"(_sub_5_2), "f"(_sub_12_1));
                    comp[2] = _sub_4_1;
                    acc[2] = _add_14;
                    float _sub_13_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_1) : "f"(_tmem_load_0[3]), "f"(comp[3]));
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(acc[3]), "f"(_sub_13_1));
                    float _sub_7_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7_1) : "f"(_add_15), "f"(acc[3]));
                    float _sub_6_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_2) : "f"(_sub_7_1), "f"(_sub_13_1));
                    comp[3] = _sub_6_2;
                    acc[3] = _add_15;
                    float _sub_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(_tmem_load_0[4]), "f"(comp[4]));
                    float _add_16;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(acc[4]), "f"(_sub_14));
                    float _sub_9_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9_1) : "f"(_add_16), "f"(acc[4]));
                    float _sub_8_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8_1) : "f"(_sub_9_1), "f"(_sub_14));
                    comp[4] = _sub_8_1;
                    acc[4] = _add_16;
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(_tmem_load_0[5]), "f"(comp[5]));
                    float _add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(acc[5]), "f"(_sub_15));
                    float _sub_11_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11_1) : "f"(_add_17), "f"(acc[5]));
                    float _sub_10_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10_1) : "f"(_sub_11_1), "f"(_sub_15));
                    comp[5] = _sub_10_1;
                    acc[5] = _add_17;
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(_tmem_load_0[6]), "f"(comp[6]));
                    float _add_18_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_1) : "f"(acc[6]), "f"(_sub_16));
                    float _sub_13_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_2) : "f"(_add_18_1), "f"(acc[6]));
                    float _sub_12_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_2) : "f"(_sub_13_2), "f"(_sub_16));
                    comp[6] = _sub_12_2;
                    acc[6] = _add_18_1;
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(_tmem_load_0[7]), "f"(comp[7]));
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(acc[7]), "f"(_sub_17));
                    float _sub_15_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15_1) : "f"(_add_19), "f"(acc[7]));
                    float _sub_14_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14_1) : "f"(_sub_15_1), "f"(_sub_17));
                    comp[7] = _sub_14_1;
                    acc[7] = _add_19;
                    float _sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(_tmem_load_1[0]), "f"(comp[8]));
                    float _add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(acc[8]), "f"(_sub_18));
                    float _sub_17_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17_1) : "f"(_add_20), "f"(acc[8]));
                    float _sub_16_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16_1) : "f"(_sub_17_1), "f"(_sub_18));
                    comp[8] = _sub_16_1;
                    acc[8] = _add_20;
                    float _sub_19_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_1) : "f"(_tmem_load_1[1]), "f"(comp[9]));
                    float _add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(acc[9]), "f"(_sub_19_1));
                    float _sub_19_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_2) : "f"(_add_21), "f"(acc[9]));
                    float _sub_18_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18_1) : "f"(_sub_19_2), "f"(_sub_19_1));
                    comp[9] = _sub_18_1;
                    acc[9] = _add_21;
                    float _sub_20_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_1) : "f"(_tmem_load_1[2]), "f"(comp[10]));
                    float _add_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(acc[10]), "f"(_sub_20_1));
                    float _sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(_add_22), "f"(acc[10]));
                    float _sub_20_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_2) : "f"(_sub_21), "f"(_sub_20_1));
                    comp[10] = _sub_20_2;
                    acc[10] = _add_22;
                    float _sub_21_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21_1) : "f"(_tmem_load_1[3]), "f"(comp[11]));
                    float _add_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(acc[11]), "f"(_sub_21_1));
                    float _sub_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(_add_23), "f"(acc[11]));
                    float _sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(_sub_23), "f"(_sub_21_1));
                    comp[11] = _sub_22;
                    acc[11] = _add_23;
#if __CUDA_ARCH__ == 1070
#else
                    float _sub_22_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_1) : "f"(_tmem_load_2[0]), "f"(comp[0]));
                    float _add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(acc[0]), "f"(_sub_22_1));
                    float _sub_25;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(_add_24), "f"(acc[0]));
                    float _sub_24;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(_sub_25), "f"(_sub_22_1));
                    comp[0] = _sub_24;
                    acc[0] = _add_24;
                    float _sub_23_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_1) : "f"(_tmem_load_2[1]), "f"(comp[1]));
                    float _add_25_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_1) : "f"(acc[1]), "f"(_sub_23_1));
                    float _sub_27_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_1) : "f"(_add_25_1), "f"(acc[1]));
                    float _sub_26_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_1) : "f"(_sub_27_1), "f"(_sub_23_1));
                    comp[1] = _sub_26_1;
                    acc[1] = _add_25_1;
                    float _sub_24_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24_1) : "f"(_tmem_load_2[2]), "f"(comp[2]));
                    float _add_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(acc[2]), "f"(_sub_24_1));
                    float _sub_29;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(_add_26), "f"(acc[2]));
                    float _sub_28;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(_sub_29), "f"(_sub_24_1));
                    comp[2] = _sub_28;
                    acc[2] = _add_26;
                    float _sub_25_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25_1) : "f"(_tmem_load_2[3]), "f"(comp[3]));
                    float _add_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(acc[3]), "f"(_sub_25_1));
                    float _sub_31;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(_add_27), "f"(acc[3]));
                    float _sub_30;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(_sub_31), "f"(_sub_25_1));
                    comp[3] = _sub_30;
                    acc[3] = _add_27;
                    float _sub_26_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_2) : "f"(_tmem_load_2[4]), "f"(comp[4]));
                    float _add_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(acc[4]), "f"(_sub_26_2));
                    float _sub_33_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_1) : "f"(_add_28), "f"(acc[4]));
                    float _sub_32;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_sub_33_1), "f"(_sub_26_2));
                    comp[4] = _sub_32;
                    acc[4] = _add_28;
                    float _sub_27_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_2) : "f"(_tmem_load_2[5]), "f"(comp[5]));
                    float _add_29;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(acc[5]), "f"(_sub_27_2));
                    float _sub_35;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(_add_29), "f"(acc[5]));
                    float _sub_34_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_1) : "f"(_sub_35), "f"(_sub_27_2));
                    comp[5] = _sub_34_1;
                    acc[5] = _add_29;
                    float _sub_28_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_1) : "f"(_tmem_load_2[6]), "f"(comp[6]));
                    float _add_30;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(acc[6]), "f"(_sub_28_1));
                    float _sub_37;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(_add_30), "f"(acc[6]));
                    float _sub_36;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(_sub_37), "f"(_sub_28_1));
                    comp[6] = _sub_36;
                    acc[6] = _add_30;
                    float _sub_29_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29_1) : "f"(_tmem_load_2[7]), "f"(comp[7]));
                    float _add_31;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(acc[7]), "f"(_sub_29_1));
                    float _sub_39;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(_add_31), "f"(acc[7]));
                    float _sub_38;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(_sub_39), "f"(_sub_29_1));
                    comp[7] = _sub_38;
                    acc[7] = _add_31;
                    float _sub_30_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30_1) : "f"(_tmem_load_3[0]), "f"(comp[8]));
                    float _add_32_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_1) : "f"(acc[8]), "f"(_sub_30_1));
                    float _sub_41_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_1) : "f"(_add_32_1), "f"(acc[8]));
                    float _sub_40_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_1) : "f"(_sub_41_1), "f"(_sub_30_1));
                    comp[8] = _sub_40_1;
                    acc[8] = _add_32_1;
                    float _sub_31_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_1) : "f"(_tmem_load_3[1]), "f"(comp[9]));
                    float _add_33;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(acc[9]), "f"(_sub_31_1));
                    float _sub_43;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(_add_33), "f"(acc[9]));
                    float _sub_42;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(_sub_43), "f"(_sub_31_1));
                    comp[9] = _sub_42;
                    acc[9] = _add_33;
                    float _sub_32_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32_1) : "f"(_tmem_load_3[2]), "f"(comp[10]));
                    float _add_34;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(acc[10]), "f"(_sub_32_1));
                    float _sub_45;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(_add_34), "f"(acc[10]));
                    float _sub_44;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(_sub_45), "f"(_sub_32_1));
                    comp[10] = _sub_44;
                    acc[10] = _add_34;
                    float _sub_33_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_2) : "f"(_tmem_load_3[3]), "f"(comp[11]));
                    float _add_35;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(acc[11]), "f"(_sub_33_2));
                    float _sub_47_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_1) : "f"(_add_35), "f"(acc[11]));
                    float _sub_46;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(_sub_47_1), "f"(_sub_33_2));
                    comp[11] = _sub_46;
                    acc[11] = _add_35;
#endif
                    cseq_1 += 1;
                }
                unsigned int ustage_1 = seq_1 % 2;
                unsigned int _stage_small_full = seq_1 % 2;
                unsigned int _parity_small_full = seq_1 / 2 % 2;
                mbarrier_wait(small_full_addr + (_stage_small_full) * 8, _parity_small_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                float _tmem_load_2[8];
#else
                float _tmem_load_4[8];
#endif
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
#else
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
#endif
                    : "r"((unsigned int)tmem_acc_small + ustage_1 * 24));
#if __CUDA_ARCH__ == 1070
                float _tmem_load_3[4];
#else
                float _tmem_load_5[4];
#endif
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
#else
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3]))
#endif
                    : "r"((unsigned int)tmem_acc_small + (ustage_1 * 24 + 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (elect_sync()) {
                    mbarrier_arrive(small_empty_addr + (ustage_1) * 8);
                }
#if __CUDA_ARCH__ == 1070
                float _sub_22_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_1) : "f"(_tmem_load_2[0]), "f"(comp[0]));
                float _add_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(acc[0]), "f"(_sub_22_1));
                float _sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(_add_24), "f"(acc[0]));
                float _sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(_sub_25), "f"(_sub_22_1));
                comp[0] = _sub_24;
                acc[0] = _add_24;
                float _sub_23_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_1) : "f"(_tmem_load_2[1]), "f"(comp[1]));
                float _add_25_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_1) : "f"(acc[1]), "f"(_sub_23_1));
                float _sub_27_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_1) : "f"(_add_25_1), "f"(acc[1]));
                float _sub_26_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_1) : "f"(_sub_27_1), "f"(_sub_23_1));
                comp[1] = _sub_26_1;
                acc[1] = _add_25_1;
                float _sub_24_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24_1) : "f"(_tmem_load_2[2]), "f"(comp[2]));
                float _add_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(acc[2]), "f"(_sub_24_1));
                float _sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(_add_26), "f"(acc[2]));
                float _sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(_sub_29), "f"(_sub_24_1));
                comp[2] = _sub_28;
                acc[2] = _add_26;
                float _sub_25_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25_1) : "f"(_tmem_load_2[3]), "f"(comp[3]));
                float _add_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(acc[3]), "f"(_sub_25_1));
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(_add_27), "f"(acc[3]));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(_sub_31), "f"(_sub_25_1));
                comp[3] = _sub_30;
                acc[3] = _add_27;
                float _sub_26_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_2) : "f"(_tmem_load_2[4]), "f"(comp[4]));
                float _add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(acc[4]), "f"(_sub_26_2));
                float _sub_33_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_1) : "f"(_add_28), "f"(acc[4]));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_sub_33_1), "f"(_sub_26_2));
                comp[4] = _sub_32;
                acc[4] = _add_28;
                float _sub_27_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_2) : "f"(_tmem_load_2[5]), "f"(comp[5]));
                float _add_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(acc[5]), "f"(_sub_27_2));
                float _sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(_add_29), "f"(acc[5]));
                float _sub_34_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_1) : "f"(_sub_35), "f"(_sub_27_2));
                comp[5] = _sub_34_1;
                acc[5] = _add_29;
                float _sub_28_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_1) : "f"(_tmem_load_2[6]), "f"(comp[6]));
                float _add_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(acc[6]), "f"(_sub_28_1));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(_add_30), "f"(acc[6]));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(_sub_37), "f"(_sub_28_1));
                comp[6] = _sub_36;
                acc[6] = _add_30;
                float _sub_29_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29_1) : "f"(_tmem_load_2[7]), "f"(comp[7]));
                float _add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(acc[7]), "f"(_sub_29_1));
                float _sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(_add_31), "f"(acc[7]));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(_sub_39), "f"(_sub_29_1));
                comp[7] = _sub_38;
                acc[7] = _add_31;
                float _sub_30_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30_1) : "f"(_tmem_load_3[0]), "f"(comp[8]));
                float _add_32_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_1) : "f"(acc[8]), "f"(_sub_30_1));
                float _sub_41_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_1) : "f"(_add_32_1), "f"(acc[8]));
                float _sub_40_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_1) : "f"(_sub_41_1), "f"(_sub_30_1));
                comp[8] = _sub_40_1;
                acc[8] = _add_32_1;
                float _sub_31_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_1) : "f"(_tmem_load_3[1]), "f"(comp[9]));
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(acc[9]), "f"(_sub_31_1));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(_add_33), "f"(acc[9]));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(_sub_43), "f"(_sub_31_1));
                comp[9] = _sub_42;
                acc[9] = _add_33;
                float _sub_32_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32_1) : "f"(_tmem_load_3[2]), "f"(comp[10]));
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(acc[10]), "f"(_sub_32_1));
                float _sub_45;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(_add_34), "f"(acc[10]));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(_sub_45), "f"(_sub_32_1));
                comp[10] = _sub_44;
                acc[10] = _add_34;
                float _sub_33_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_2) : "f"(_tmem_load_3[3]), "f"(comp[11]));
                float _add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(acc[11]), "f"(_sub_33_2));
                float _sub_47_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_1) : "f"(_add_35), "f"(acc[11]));
                float _sub_46;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(_sub_47_1), "f"(_sub_33_2));
                comp[11] = _sub_46;
                acc[11] = _add_35;
#else
                float _sub_34_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_2) : "f"(_tmem_load_4[0]), "f"(comp[0]));
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(acc[0]), "f"(_sub_34_2));
                float _sub_49;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(_add_36), "f"(acc[0]));
                float _sub_48_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48_1) : "f"(_sub_49), "f"(_sub_34_2));
                comp[0] = _sub_48_1;
                acc[0] = _add_36;
                float _sub_35_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35_1) : "f"(_tmem_load_4[1]), "f"(comp[1]));
                float _add_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(acc[1]), "f"(_sub_35_1));
                float _sub_51;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(_add_37), "f"(acc[1]));
                float _sub_50;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(_sub_51), "f"(_sub_35_1));
                comp[1] = _sub_50;
                acc[1] = _add_37;
                float _sub_36_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36_1) : "f"(_tmem_load_4[2]), "f"(comp[2]));
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(acc[2]), "f"(_sub_36_1));
                float _sub_53;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(_add_38), "f"(acc[2]));
                float _sub_52;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(_sub_53), "f"(_sub_36_1));
                comp[2] = _sub_52;
                acc[2] = _add_38;
                float _sub_37_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37_1) : "f"(_tmem_load_4[3]), "f"(comp[3]));
                float _add_39_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_1) : "f"(acc[3]), "f"(_sub_37_1));
                float _sub_55_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55_1) : "f"(_add_39_1), "f"(acc[3]));
                float _sub_54_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54_1) : "f"(_sub_55_1), "f"(_sub_37_1));
                comp[3] = _sub_54_1;
                acc[3] = _add_39_1;
                float _sub_38_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38_1) : "f"(_tmem_load_4[4]), "f"(comp[4]));
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(acc[4]), "f"(_sub_38_1));
                float _sub_57;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(_add_40), "f"(acc[4]));
                float _sub_56;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(_sub_57), "f"(_sub_38_1));
                comp[4] = _sub_56;
                acc[4] = _add_40;
                float _sub_39_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39_1) : "f"(_tmem_load_4[5]), "f"(comp[5]));
                float _add_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(acc[5]), "f"(_sub_39_1));
                float _sub_59;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(_add_41), "f"(acc[5]));
                float _sub_58;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(_sub_59), "f"(_sub_39_1));
                comp[5] = _sub_58;
                acc[5] = _add_41;
                float _sub_40_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_2) : "f"(_tmem_load_4[6]), "f"(comp[6]));
                float _add_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(acc[6]), "f"(_sub_40_2));
                float _sub_61;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(_add_42), "f"(acc[6]));
                float _sub_60;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(_sub_61), "f"(_sub_40_2));
                comp[6] = _sub_60;
                acc[6] = _add_42;
                float _sub_41_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_2) : "f"(_tmem_load_4[7]), "f"(comp[7]));
                float _add_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(acc[7]), "f"(_sub_41_2));
                float _sub_63;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(_add_43), "f"(acc[7]));
                float _sub_62;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(_sub_63), "f"(_sub_41_2));
                comp[7] = _sub_62;
                acc[7] = _add_43;
                float _sub_42_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42_1) : "f"(_tmem_load_5[0]), "f"(comp[8]));
                float _add_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(acc[8]), "f"(_sub_42_1));
                float _sub_65;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(_add_44), "f"(acc[8]));
                float _sub_64;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(_sub_65), "f"(_sub_42_1));
                comp[8] = _sub_64;
                acc[8] = _add_44;
                float _sub_43_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43_1) : "f"(_tmem_load_5[1]), "f"(comp[9]));
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(acc[9]), "f"(_sub_43_1));
                float _sub_67;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(_add_45), "f"(acc[9]));
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(_sub_67), "f"(_sub_43_1));
                comp[9] = _sub_66;
                acc[9] = _add_45;
                float _sub_44_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44_1) : "f"(_tmem_load_5[2]), "f"(comp[10]));
                float _add_46_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46_1) : "f"(acc[10]), "f"(_sub_44_1));
                float _sub_69;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(_add_46_1), "f"(acc[10]));
                float _sub_68;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(_sub_69), "f"(_sub_44_1));
                comp[10] = _sub_68;
                acc[10] = _add_46_1;
                float _sub_45_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45_1) : "f"(_tmem_load_5[3]), "f"(comp[11]));
                float _add_47;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(acc[11]), "f"(_sub_45_1));
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(_add_47), "f"(acc[11]));
                float _sub_70;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(_sub_71), "f"(_sub_45_1));
                comp[11] = _sub_70;
                acc[11] = _add_47;
#endif
                unsigned int row_5 = first_row_1;
                float values[2];
                float total;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total) : "f"(acc[0]), "f"(comp[0]));
                values[0] = ((_fast_div_q_7 * 64 + row_5 < num_tokens) ? total : 0.0f);
                float total_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_0) : "f"(acc[1]), "f"(comp[1]));
                values[1] = ((_fast_div_q_7 * 64 + row_5 < num_tokens) ? total_0 : 0.0f);
                {
                    float2 _v2 = make_float2(values[0 + 0], values[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_5 * 24) + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_1_2 = first_row_1 + 8;
                float values_2[2];
                float total_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_3) : "f"(acc[2]), "f"(comp[2]));
                values_2[0] = ((_fast_div_q_7 * 64 + row_1_2 < num_tokens) ? total_3 : 0.0f);
                float total_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_4) : "f"(acc[3]), "f"(comp[3]));
                values_2[1] = ((_fast_div_q_7 * 64 + row_1_2 < num_tokens) ? total_4 : 0.0f);
                {
                    float2 _v2 = make_float2(values_2[0 + 0], values_2[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_1_2 * 24) + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_5_1 = first_row_1;
                float values_6[2];
                float total_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_7) : "f"(acc[4]), "f"(comp[4]));
                values_6[0] = ((_fast_div_q_7 * 64 + row_5_1 < num_tokens) ? total_7 : 0.0f);
                float total_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_8) : "f"(acc[5]), "f"(comp[5]));
                values_6[1] = ((_fast_div_q_7 * 64 + row_5_1 < num_tokens) ? total_8 : 0.0f);
                {
                    float2 _v2 = make_float2(values_6[0 + 0], values_6[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_5_1 * 24) + 8 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_9_1 = first_row_1 + 8;
                float values_10[2];
                float total_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_11) : "f"(acc[6]), "f"(comp[6]));
                values_10[0] = ((_fast_div_q_7 * 64 + row_9_1 < num_tokens) ? total_11 : 0.0f);
                float total_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_12) : "f"(acc[7]), "f"(comp[7]));
                values_10[1] = ((_fast_div_q_7 * 64 + row_9_1 < num_tokens) ? total_12 : 0.0f);
                {
                    float2 _v2 = make_float2(values_10[0 + 0], values_10[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_9_1 * 24) + 8 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_13 = first_row_1;
                float values_14[2];
                float total_15;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_15) : "f"(acc[8]), "f"(comp[8]));
                values_14[0] = ((_fast_div_q_7 * 64 + row_13 < num_tokens) ? total_15 : 0.0f);
                float total_16;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_16) : "f"(acc[9]), "f"(comp[9]));
                values_14[1] = ((_fast_div_q_7 * 64 + row_13 < num_tokens) ? total_16 : 0.0f);
                {
                    float2 _v2 = make_float2(values_14[0 + 0], values_14[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_13 * 24) + 16 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_17 = first_row_1 + 8;
                float values_18[2];
                float total_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_19) : "f"(acc[10]), "f"(comp[10]));
                values_18[0] = ((_fast_div_q_7 * 64 + row_17 < num_tokens) ? total_19 : 0.0f);
                float total_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_20) : "f"(acc[11]), "f"(comp[11]));
                values_18[1] = ((_fast_div_q_7 * 64 + row_17 < num_tokens) ? total_20 : 0.0f);
                {
                    float2 _v2 = make_float2(values_18[0 + 0], values_18[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_17 * 24) + 16 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
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
                        asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + _fast_div_q_7) * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
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
#if __CUDA_ARCH__ == 1070
            unsigned int step_1 = (((num_tokens + 63) / 64 * num_splits <= 3 * (unsigned int)num_ctas) ? num_ctas * 8 : num_ctas * 4);
            unsigned int stop_2 = (((num_tokens + 63) / 64 * num_splits <= 3 * (unsigned int)num_ctas || warp < 20) ? num_tokens : 0);
            unsigned int first_3 = (((num_tokens + 63) / 64 * num_splits <= num_ctas) ? (unsigned int)bid + (unsigned int)(warp - 16) * num_ctas : first_0);
#else
            unsigned int step_1 = (((num_tokens + 63) / 64 * num_splits <= 3 * (unsigned int)num_ctas && (num_tokens + 63) / 64 * num_splits <= num_ctas) ? num_ctas * 8 : num_ctas * 4);
            unsigned int stop_2 = ((num_tokens > 0 || warp < 20) ? num_tokens : 0);
            unsigned int first_3 = (((num_tokens + 63) / 64 * num_splits <= num_ctas) ? (unsigned int)bid + ((warp < 20) ? (unsigned int)(warp - 16) : (((num_tokens + 63) / 64 * num_splits > num_ctas || num_tokens <= num_ctas * 4) ? (unsigned int)(warp - 18) % 4 : (unsigned int)(warp - 16))) * num_ctas : (unsigned int)bid * 4 + ((warp < 20) ? (unsigned int)(warp - 16) : (((num_tokens + 63) / 64 * num_splits > num_ctas || num_tokens <= num_ctas * 4) ? (unsigned int)(warp - 18) % 4 : (unsigned int)(warp - 16))));
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
                unsigned int flag = ((warp >= 20 && ((num_tokens + 63) / 64 * num_splits > num_ctas || num_tokens <= num_ctas * 4)) ? 1 : 0);
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
                    unsigned int row_8 = token % 64;
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
                    unsigned long long record = ((unsigned long long)first_task + (unsigned long long)safe_split) * 1536 + (unsigned long long)(row_8 * 24);
                    float _vec_load_0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record + 0);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
#else
                if ((num_tokens + 63) / 64 * num_splits > num_ctas || num_tokens <= num_ctas * 4) {
                    unsigned int w = ((flag == 1) ? (unsigned int)(warp - 18) : (unsigned int)(warp - 16)) % 4;
                    unsigned int col0 = flag * 12;
                    asm volatile("barrier.sync %0, 64;" :: "r"(4 + w) : "memory");
                    float total_1 = 0.0f;
                    if (num_splits > 22) {
                        unsigned int row_8 = token % 64;
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
                        unsigned long long record = ((unsigned long long)first_task + (unsigned long long)safe_split) * 1536 + (unsigned long long)(row_8 * 24);
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + (unsigned long long)col0) + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        float value_1 = ((split < num_splits) ? _vec_load_0[0] : 0.0f);
                        sums[0] = value_1;
                        float value_0 = ((split < num_splits) ? _vec_load_0[1] : 0.0f);
                        sums[8] = value_0;
                        float value_1_1 = ((split < num_splits) ? _vec_load_0[2] : 0.0f);
                        sums[4] = value_1_1;
                        float value_2 = ((split < num_splits) ? _vec_load_0[3] : 0.0f);
                        sums[12] = value_2;
                        float _vec_load_1[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_1[0 + 0] = _v4.x;
                            _vec_load_1[0 + 1] = _v4.y;
                            _vec_load_1[0 + 2] = _v4.z;
                            _vec_load_1[0 + 3] = _v4.w;
                        }
                        float value_3 = ((split < num_splits) ? _vec_load_1[0] : 0.0f);
                        sums[2] = value_3;
                        float value_4 = ((split < num_splits) ? _vec_load_1[1] : 0.0f);
                        sums[10] = value_4;
                        float value_5 = ((split < num_splits) ? _vec_load_1[2] : 0.0f);
                        sums[6] = value_5;
                        float value_6 = ((split < num_splits) ? _vec_load_1[3] : 0.0f);
                        sums[14] = value_6;
                        float _vec_load_2[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_2[0 + 0] = _v4.x;
                            _vec_load_2[0 + 1] = _v4.y;
                            _vec_load_2[0 + 2] = _v4.z;
                            _vec_load_2[0 + 3] = _v4.w;
                        }
                        float value_7 = ((split < num_splits) ? _vec_load_2[0] : 0.0f);
                        sums[1] = value_7;
                        float value_8_1 = ((split < num_splits) ? _vec_load_2[1] : 0.0f);
                        sums[9] = value_8_1;
                        float value_9 = ((split < num_splits) ? _vec_load_2[2] : 0.0f);
                        sums[5] = value_9;
                        float value_10 = ((split < num_splits) ? _vec_load_2[3] : 0.0f);
                        sums[13] = value_10;
                        float q_word = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split * 64) + (unsigned long long)row_8];
                        float q_value = ((split < num_splits) ? q_word : 0.0f);
                        sums[3] = q_value;
                        unsigned int split_11 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_12 = ((split_11 < num_splits) ? split_11 : 0);
                        unsigned long long record_13 = ((unsigned long long)first_task + (unsigned long long)safe_split_12) * 1536 + (unsigned long long)(row_8 * 24);
                        float _vec_load_3[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13 + (unsigned long long)col0) + 0);
                            _vec_load_3[0 + 0] = _v4.x;
                            _vec_load_3[0 + 1] = _v4.y;
                            _vec_load_3[0 + 2] = _v4.z;
                            _vec_load_3[0 + 3] = _v4.w;
                        }
                        float value_14 = ((split_11 < num_splits) ? _vec_load_3[0] : 0.0f);
                        float s_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(sums[0]), "f"(value_14));
                        float bb;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb) : "f"(s_15), "f"(sums[0]));
                        float _sub_5_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_3) : "f"(s_15), "f"(bb));
                        float _sub_4_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4_2) : "f"(sums[0]), "f"(_sub_5_3));
                        float _sub_6_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_3) : "f"(value_14), "f"(bb));
                        float e;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_4_2), "f"(_sub_6_3));
                        sums[0] = s_15;
                        errs[0] = e;
                        float value_16 = ((split_11 < num_splits) ? _vec_load_3[1] : 0.0f);
                        float s_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(sums[8]), "f"(value_16));
                        float bb_18;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_18) : "f"(s_17), "f"(sums[8]));
                        float _sub_8_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8_2) : "f"(s_17), "f"(bb_18));
                        float _sub_7_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7_2) : "f"(sums[8]), "f"(_sub_8_2));
                        float _sub_9_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9_2) : "f"(value_16), "f"(bb_18));
                        float e_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_7_2), "f"(_sub_9_2));
                        sums[8] = s_17;
                        errs[8] = e_19;
                        float value_20 = ((split_11 < num_splits) ? _vec_load_3[2] : 0.0f);
                        float s_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(sums[4]), "f"(value_20));
                        float bb_22;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_22) : "f"(s_21), "f"(sums[4]));
                        float _sub_11_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11_2) : "f"(s_21), "f"(bb_22));
                        float _sub_10_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10_2) : "f"(sums[4]), "f"(_sub_11_2));
                        float _sub_12_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_3) : "f"(value_20), "f"(bb_22));
                        float e_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_23) : "f"(_sub_10_2), "f"(_sub_12_3));
                        sums[4] = s_21;
                        errs[4] = e_23;
                        float value_24 = ((split_11 < num_splits) ? _vec_load_3[3] : 0.0f);
                        float s_25;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(sums[12]), "f"(value_24));
                        float bb_26;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_26) : "f"(s_25), "f"(sums[12]));
                        float _sub_14_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14_2) : "f"(s_25), "f"(bb_26));
                        float _sub_13_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_3) : "f"(sums[12]), "f"(_sub_14_2));
                        float _sub_15_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15_2) : "f"(value_24), "f"(bb_26));
                        float e_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_27) : "f"(_sub_13_3), "f"(_sub_15_2));
                        sums[12] = s_25;
                        errs[12] = e_27;
                        float _vec_load_4[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13 + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_4[0 + 0] = _v4.x;
                            _vec_load_4[0 + 1] = _v4.y;
                            _vec_load_4[0 + 2] = _v4.z;
                            _vec_load_4[0 + 3] = _v4.w;
                        }
                        float value_28 = ((split_11 < num_splits) ? _vec_load_4[0] : 0.0f);
                        float s_29;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(sums[2]), "f"(value_28));
                        float bb_30;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_30) : "f"(s_29), "f"(sums[2]));
                        float _sub_18_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18_2) : "f"(s_29), "f"(bb_30));
                        float _sub_17_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17_2) : "f"(sums[2]), "f"(_sub_18_2));
                        float _sub_19_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_3) : "f"(value_28), "f"(bb_30));
                        float e_31;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_31) : "f"(_sub_17_2), "f"(_sub_19_3));
                        sums[2] = s_29;
                        errs[2] = e_31;
                        float value_32 = ((split_11 < num_splits) ? _vec_load_4[1] : 0.0f);
                        float s_33;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(sums[10]), "f"(value_32));
                        float bb_34;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_34) : "f"(s_33), "f"(sums[10]));
                        float _sub_21_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21_2) : "f"(s_33), "f"(bb_34));
                        float _sub_20_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_3) : "f"(sums[10]), "f"(_sub_21_2));
                        float _sub_22_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_2) : "f"(value_32), "f"(bb_34));
                        float e_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_35) : "f"(_sub_20_3), "f"(_sub_22_2));
                        sums[10] = s_33;
                        errs[10] = e_35;
                        float value_36 = ((split_11 < num_splits) ? _vec_load_4[2] : 0.0f);
                        float s_37;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(sums[6]), "f"(value_36));
                        float bb_38;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_38) : "f"(s_37), "f"(sums[6]));
                        float _sub_24_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24_2) : "f"(s_37), "f"(bb_38));
                        float _sub_23_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_2) : "f"(sums[6]), "f"(_sub_24_2));
                        float _sub_25_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25_2) : "f"(value_36), "f"(bb_38));
                        float e_39;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_39) : "f"(_sub_23_2), "f"(_sub_25_2));
                        sums[6] = s_37;
                        errs[6] = e_39;
                        float value_40 = ((split_11 < num_splits) ? _vec_load_4[3] : 0.0f);
                        float s_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(sums[14]), "f"(value_40));
                        float bb_42;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_42) : "f"(s_41), "f"(sums[14]));
                        float _sub_27_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_3) : "f"(s_41), "f"(bb_42));
                        float _sub_26_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_3) : "f"(sums[14]), "f"(_sub_27_3));
                        float _sub_28_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_2) : "f"(value_40), "f"(bb_42));
                        float e_43;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_43) : "f"(_sub_26_3), "f"(_sub_28_2));
                        sums[14] = s_41;
                        errs[14] = e_43;
                        float _vec_load_5[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13 + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_5[0 + 0] = _v4.x;
                            _vec_load_5[0 + 1] = _v4.y;
                            _vec_load_5[0 + 2] = _v4.z;
                            _vec_load_5[0 + 3] = _v4.w;
                        }
                        float value_44_1 = ((split_11 < num_splits) ? _vec_load_5[0] : 0.0f);
                        float s_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(sums[1]), "f"(value_44_1));
                        float bb_46;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_46) : "f"(s_45), "f"(sums[1]));
                        float _sub_31_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_2) : "f"(s_45), "f"(bb_46));
                        float _sub_30_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30_2) : "f"(sums[1]), "f"(_sub_31_2));
                        float _sub_32_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32_2) : "f"(value_44_1), "f"(bb_46));
                        float e_47;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_47) : "f"(_sub_30_2), "f"(_sub_32_2));
                        sums[1] = s_45;
                        errs[1] = e_47;
                        float value_48 = ((split_11 < num_splits) ? _vec_load_5[1] : 0.0f);
                        float s_49;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(sums[9]), "f"(value_48));
                        float bb_50;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_50) : "f"(s_49), "f"(sums[9]));
                        float _sub_34_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_3) : "f"(s_49), "f"(bb_50));
                        float _sub_33_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_3) : "f"(sums[9]), "f"(_sub_34_3));
                        float _sub_35_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35_2) : "f"(value_48), "f"(bb_50));
                        float e_51;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_51) : "f"(_sub_33_3), "f"(_sub_35_2));
                        sums[9] = s_49;
                        errs[9] = e_51;
                        float value_52 = ((split_11 < num_splits) ? _vec_load_5[2] : 0.0f);
                        float s_53;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(sums[5]), "f"(value_52));
                        float bb_54;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_54) : "f"(s_53), "f"(sums[5]));
                        float _sub_37_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37_2) : "f"(s_53), "f"(bb_54));
                        float _sub_36_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36_2) : "f"(sums[5]), "f"(_sub_37_2));
                        float _sub_38_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38_2) : "f"(value_52), "f"(bb_54));
                        float e_55;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_55) : "f"(_sub_36_2), "f"(_sub_38_2));
                        sums[5] = s_53;
                        errs[5] = e_55;
                        float value_56 = ((split_11 < num_splits) ? _vec_load_5[3] : 0.0f);
                        float s_57;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(sums[13]), "f"(value_56));
                        float bb_58;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_58) : "f"(s_57), "f"(sums[13]));
                        float _sub_40_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_3) : "f"(s_57), "f"(bb_58));
                        float _sub_39_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39_2) : "f"(sums[13]), "f"(_sub_40_3));
                        float _sub_41_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_3) : "f"(value_56), "f"(bb_58));
                        float e_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_59) : "f"(_sub_39_2), "f"(_sub_41_3));
                        sums[13] = s_57;
                        errs[13] = e_59;
                        float q_word_60 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split_12 * 64) + (unsigned long long)row_8];
                        float q_value_61 = ((split_11 < num_splits) ? q_word_60 : 0.0f);
                        float s_62;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(sums[3]), "f"(q_value_61));
                        float bb_63;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_63) : "f"(s_62), "f"(sums[3]));
                        float _sub_43_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43_2) : "f"(s_62), "f"(bb_63));
                        float _sub_42_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42_2) : "f"(sums[3]), "f"(_sub_43_2));
                        float _sub_44_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44_2) : "f"(q_value_61), "f"(bb_63));
                        float e_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_64) : "f"(_sub_42_2), "f"(_sub_44_2));
                        sums[3] = s_62;
                        errs[3] = e_64;
                        float half_s1[8];
                        float half_e1[8];
                        float keep = ((lane_in_warp % 2 == 1) ? sums[8] : sums[0]);
                        float send = ((lane_in_warp % 2 == 1) ? sums[0] : sums[8]);
                        float keep_err = ((lane_in_warp % 2 == 1) ? errs[8] : errs[0]);
                        float send_err = ((lane_in_warp % 2 == 1) ? errs[0] : errs[8]);
                        float received = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                        float received_err = __shfl_xor_sync(0xFFFFFFFF, send_err, 1);
                        float s_65;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(keep), "f"(received));
                        float bb_66;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_66) : "f"(s_65), "f"(keep));
                        float _sub_46_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46_1) : "f"(s_65), "f"(bb_66));
                        float _sub_45_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45_2) : "f"(keep), "f"(_sub_46_1));
                        float _sub_47_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_2) : "f"(received), "f"(bb_66));
                        float e_67;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_67) : "f"(_sub_45_2), "f"(_sub_47_2));
                        half_s1[0] = s_65;
                        float _add_49;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(keep_err), "f"(received_err));
                        float _add_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_49), "f"(e_67));
                        half_e1[0] = _add_48;
                        float keep_68 = ((lane_in_warp % 2 == 1) ? sums[9] : sums[1]);
                        float send_69 = ((lane_in_warp % 2 == 1) ? sums[1] : sums[9]);
                        float keep_err_70 = ((lane_in_warp % 2 == 1) ? errs[9] : errs[1]);
                        float send_err_71 = ((lane_in_warp % 2 == 1) ? errs[1] : errs[9]);
                        float received_72 = __shfl_xor_sync(0xFFFFFFFF, send_69, 1);
                        float received_err_73 = __shfl_xor_sync(0xFFFFFFFF, send_err_71, 1);
                        float s_74;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_74) : "f"(keep_68), "f"(received_72));
                        float bb_75;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_75) : "f"(s_74), "f"(keep_68));
                        float _sub_51_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51_1) : "f"(s_74), "f"(bb_75));
                        float _sub_50_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50_1) : "f"(keep_68), "f"(_sub_51_1));
                        float _sub_52_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52_1) : "f"(received_72), "f"(bb_75));
                        float e_76;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_76) : "f"(_sub_50_1), "f"(_sub_52_1));
                        half_s1[1] = s_74;
                        float _add_54;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(keep_err_70), "f"(received_err_73));
                        float _add_53_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_1) : "f"(_add_54), "f"(e_76));
                        half_e1[1] = _add_53_1;
                        float keep_77 = ((lane_in_warp % 2 == 1) ? sums[10] : sums[2]);
                        float send_78 = ((lane_in_warp % 2 == 1) ? sums[2] : sums[10]);
                        float keep_err_79 = ((lane_in_warp % 2 == 1) ? errs[10] : errs[2]);
                        float send_err_80 = ((lane_in_warp % 2 == 1) ? errs[2] : errs[10]);
                        float received_81 = __shfl_xor_sync(0xFFFFFFFF, send_78, 1);
                        float received_err_82 = __shfl_xor_sync(0xFFFFFFFF, send_err_80, 1);
                        float s_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_83) : "f"(keep_77), "f"(received_81));
                        float bb_84;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_84) : "f"(s_83), "f"(keep_77));
                        float _sub_56_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56_1) : "f"(s_83), "f"(bb_84));
                        float _sub_55_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55_2) : "f"(keep_77), "f"(_sub_56_1));
                        float _sub_57_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57_1) : "f"(received_81), "f"(bb_84));
                        float e_85;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_85) : "f"(_sub_55_2), "f"(_sub_57_1));
                        half_s1[2] = s_83;
                        float _add_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(keep_err_79), "f"(received_err_82));
                        float _add_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(_add_59), "f"(e_85));
                        half_e1[2] = _add_58;
                        float keep_86 = ((lane_in_warp % 2 == 1) ? sums[11] : sums[3]);
                        float send_87 = ((lane_in_warp % 2 == 1) ? sums[3] : sums[11]);
                        float keep_err_88 = ((lane_in_warp % 2 == 1) ? errs[11] : errs[3]);
                        float send_err_89 = ((lane_in_warp % 2 == 1) ? errs[3] : errs[11]);
                        float received_90 = __shfl_xor_sync(0xFFFFFFFF, send_87, 1);
                        float received_err_91 = __shfl_xor_sync(0xFFFFFFFF, send_err_89, 1);
                        float s_92;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_92) : "f"(keep_86), "f"(received_90));
                        float bb_93;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_93) : "f"(s_92), "f"(keep_86));
                        float _sub_61_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61_1) : "f"(s_92), "f"(bb_93));
                        float _sub_60_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60_1) : "f"(keep_86), "f"(_sub_61_1));
                        float _sub_62_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62_1) : "f"(received_90), "f"(bb_93));
                        float e_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_94) : "f"(_sub_60_1), "f"(_sub_62_1));
                        half_s1[3] = s_92;
                        float _add_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(keep_err_88), "f"(received_err_91));
                        float _add_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_add_64), "f"(e_94));
                        half_e1[3] = _add_63;
                        float keep_95 = ((lane_in_warp % 2 == 1) ? sums[12] : sums[4]);
                        float send_96 = ((lane_in_warp % 2 == 1) ? sums[4] : sums[12]);
                        float keep_err_97 = ((lane_in_warp % 2 == 1) ? errs[12] : errs[4]);
                        float send_err_98 = ((lane_in_warp % 2 == 1) ? errs[4] : errs[12]);
                        float received_99 = __shfl_xor_sync(0xFFFFFFFF, send_96, 1);
                        float received_err_100 = __shfl_xor_sync(0xFFFFFFFF, send_err_98, 1);
                        float s_101;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_101) : "f"(keep_95), "f"(received_99));
                        float bb_102;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_102) : "f"(s_101), "f"(keep_95));
                        float _sub_66_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66_1) : "f"(s_101), "f"(bb_102));
                        float _sub_65_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65_1) : "f"(keep_95), "f"(_sub_66_1));
                        float _sub_67_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67_1) : "f"(received_99), "f"(bb_102));
                        float e_103;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_103) : "f"(_sub_65_1), "f"(_sub_67_1));
                        half_s1[4] = s_101;
                        float _add_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(keep_err_97), "f"(received_err_100));
                        float _add_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(e_103));
                        half_e1[4] = _add_68;
                        float keep_104 = ((lane_in_warp % 2 == 1) ? sums[13] : sums[5]);
                        float send_105 = ((lane_in_warp % 2 == 1) ? sums[5] : sums[13]);
                        float keep_err_106 = ((lane_in_warp % 2 == 1) ? errs[13] : errs[5]);
                        float send_err_107 = ((lane_in_warp % 2 == 1) ? errs[5] : errs[13]);
                        float received_108 = __shfl_xor_sync(0xFFFFFFFF, send_105, 1);
                        float received_err_109 = __shfl_xor_sync(0xFFFFFFFF, send_err_107, 1);
                        float s_110;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_110) : "f"(keep_104), "f"(received_108));
                        float bb_111;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_111) : "f"(s_110), "f"(keep_104));
                        float _sub_71_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71_1) : "f"(s_110), "f"(bb_111));
                        float _sub_70_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70_1) : "f"(keep_104), "f"(_sub_71_1));
                        float _sub_72;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(received_108), "f"(bb_111));
                        float e_112;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_112) : "f"(_sub_70_1), "f"(_sub_72));
                        half_s1[5] = s_110;
                        float _add_74;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(keep_err_106), "f"(received_err_109));
                        float _add_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(e_112));
                        half_e1[5] = _add_73;
                        float keep_113 = ((lane_in_warp % 2 == 1) ? sums[14] : sums[6]);
                        float send_114 = ((lane_in_warp % 2 == 1) ? sums[6] : sums[14]);
                        float keep_err_115 = ((lane_in_warp % 2 == 1) ? errs[14] : errs[6]);
                        float send_err_116 = ((lane_in_warp % 2 == 1) ? errs[6] : errs[14]);
                        float received_117 = __shfl_xor_sync(0xFFFFFFFF, send_114, 1);
                        float received_err_118 = __shfl_xor_sync(0xFFFFFFFF, send_err_116, 1);
                        float s_119;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_119) : "f"(keep_113), "f"(received_117));
                        float bb_120;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_120) : "f"(s_119), "f"(keep_113));
                        float _sub_76;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(s_119), "f"(bb_120));
                        float _sub_75;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(keep_113), "f"(_sub_76));
                        float _sub_77;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(received_117), "f"(bb_120));
                        float e_121;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_121) : "f"(_sub_75), "f"(_sub_77));
                        half_s1[6] = s_119;
                        float _add_79;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(keep_err_115), "f"(received_err_118));
                        float _add_78;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_79), "f"(e_121));
                        half_e1[6] = _add_78;
                        float keep_122 = ((lane_in_warp % 2 == 1) ? sums[15] : sums[7]);
                        float send_123 = ((lane_in_warp % 2 == 1) ? sums[7] : sums[15]);
                        float keep_err_124 = ((lane_in_warp % 2 == 1) ? errs[15] : errs[7]);
                        float send_err_125 = ((lane_in_warp % 2 == 1) ? errs[7] : errs[15]);
                        float received_126 = __shfl_xor_sync(0xFFFFFFFF, send_123, 1);
                        float received_err_127 = __shfl_xor_sync(0xFFFFFFFF, send_err_125, 1);
                        float s_128;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_128) : "f"(keep_122), "f"(received_126));
                        float bb_129;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_129) : "f"(s_128), "f"(keep_122));
                        float _sub_81;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(s_128), "f"(bb_129));
                        float _sub_80;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(keep_122), "f"(_sub_81));
                        float _sub_82;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(received_126), "f"(bb_129));
                        float e_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_130) : "f"(_sub_80), "f"(_sub_82));
                        half_s1[7] = s_128;
                        float _add_84;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(keep_err_124), "f"(received_err_127));
                        float _add_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_84), "f"(e_130));
                        half_e1[7] = _add_83;
                        float half_s2[4];
                        float half_e2[4];
                        float keep_131 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[4] : half_s1[0]);
                        float send_132 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[0] : half_s1[4]);
                        float keep_err_133 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[4] : half_e1[0]);
                        float send_err_134 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[0] : half_e1[4]);
                        float received_135 = __shfl_xor_sync(0xFFFFFFFF, send_132, 2);
                        float received_err_136 = __shfl_xor_sync(0xFFFFFFFF, send_err_134, 2);
                        float s_137;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_137) : "f"(keep_131), "f"(received_135));
                        float bb_138;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_138) : "f"(s_137), "f"(keep_131));
                        float _sub_86;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(s_137), "f"(bb_138));
                        float _sub_85;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(keep_131), "f"(_sub_86));
                        float _sub_87;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(received_135), "f"(bb_138));
                        float e_139;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_139) : "f"(_sub_85), "f"(_sub_87));
                        half_s2[0] = s_137;
                        float _add_89;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(keep_err_133), "f"(received_err_136));
                        float _add_88;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(e_139));
                        half_e2[0] = _add_88;
                        float keep_140 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[5] : half_s1[1]);
                        float send_141 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[1] : half_s1[5]);
                        float keep_err_142 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[5] : half_e1[1]);
                        float send_err_143 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[1] : half_e1[5]);
                        float received_144 = __shfl_xor_sync(0xFFFFFFFF, send_141, 2);
                        float received_err_145 = __shfl_xor_sync(0xFFFFFFFF, send_err_143, 2);
                        float s_146;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_146) : "f"(keep_140), "f"(received_144));
                        float bb_147;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_147) : "f"(s_146), "f"(keep_140));
                        float _sub_91;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(s_146), "f"(bb_147));
                        float _sub_90;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(keep_140), "f"(_sub_91));
                        float _sub_92;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(received_144), "f"(bb_147));
                        float e_148;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_148) : "f"(_sub_90), "f"(_sub_92));
                        half_s2[1] = s_146;
                        float _add_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(keep_err_142), "f"(received_err_145));
                        float _add_93;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_94), "f"(e_148));
                        half_e2[1] = _add_93;
                        float keep_149 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[6] : half_s1[2]);
                        float send_150 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[2] : half_s1[6]);
                        float keep_err_151 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[6] : half_e1[2]);
                        float send_err_152 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[2] : half_e1[6]);
                        float received_153 = __shfl_xor_sync(0xFFFFFFFF, send_150, 2);
                        float received_err_154 = __shfl_xor_sync(0xFFFFFFFF, send_err_152, 2);
                        float s_155;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_155) : "f"(keep_149), "f"(received_153));
                        float bb_156;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_156) : "f"(s_155), "f"(keep_149));
                        float _sub_96;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(s_155), "f"(bb_156));
                        float _sub_95;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(keep_149), "f"(_sub_96));
                        float _sub_97;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(received_153), "f"(bb_156));
                        float e_157;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_157) : "f"(_sub_95), "f"(_sub_97));
                        half_s2[2] = s_155;
                        float _add_99;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(keep_err_151), "f"(received_err_154));
                        float _add_98;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_add_99), "f"(e_157));
                        half_e2[2] = _add_98;
                        float keep_158 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[7] : half_s1[3]);
                        float send_159 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[3] : half_s1[7]);
                        float keep_err_160 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[7] : half_e1[3]);
                        float send_err_161 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[3] : half_e1[7]);
                        float received_162 = __shfl_xor_sync(0xFFFFFFFF, send_159, 2);
                        float received_err_163 = __shfl_xor_sync(0xFFFFFFFF, send_err_161, 2);
                        float s_164;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_164) : "f"(keep_158), "f"(received_162));
                        float bb_165;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_165) : "f"(s_164), "f"(keep_158));
                        float _sub_101;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(s_164), "f"(bb_165));
                        float _sub_100;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(keep_158), "f"(_sub_101));
                        float _sub_102;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(received_162), "f"(bb_165));
                        float e_166;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_166) : "f"(_sub_100), "f"(_sub_102));
                        half_s2[3] = s_164;
                        float _add_104;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(keep_err_160), "f"(received_err_163));
                        float _add_103;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_104), "f"(e_166));
                        half_e2[3] = _add_103;
                        float half_s3[2];
                        float half_e3[2];
                        float keep_167 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[2] : half_s2[0]);
                        float send_168 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[0] : half_s2[2]);
                        float keep_err_169 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[2] : half_e2[0]);
                        float send_err_170 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[0] : half_e2[2]);
                        float received_171 = __shfl_xor_sync(0xFFFFFFFF, send_168, 4);
                        float received_err_172 = __shfl_xor_sync(0xFFFFFFFF, send_err_170, 4);
                        float s_173;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_173) : "f"(keep_167), "f"(received_171));
                        float bb_174;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_174) : "f"(s_173), "f"(keep_167));
                        float _sub_106;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106) : "f"(s_173), "f"(bb_174));
                        float _sub_105;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(keep_167), "f"(_sub_106));
                        float _sub_107;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_107) : "f"(received_171), "f"(bb_174));
                        float e_175;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_175) : "f"(_sub_105), "f"(_sub_107));
                        half_s3[0] = s_173;
                        float _add_109;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(keep_err_169), "f"(received_err_172));
                        float _add_108;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_add_109), "f"(e_175));
                        half_e3[0] = _add_108;
                        float keep_176 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[3] : half_s2[1]);
                        float send_177 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[1] : half_s2[3]);
                        float keep_err_178 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[3] : half_e2[1]);
                        float send_err_179 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[1] : half_e2[3]);
                        float received_180 = __shfl_xor_sync(0xFFFFFFFF, send_177, 4);
                        float received_err_181 = __shfl_xor_sync(0xFFFFFFFF, send_err_179, 4);
                        float s_182;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_182) : "f"(keep_176), "f"(received_180));
                        float bb_183;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_183) : "f"(s_182), "f"(keep_176));
                        float _sub_111;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(s_182), "f"(bb_183));
                        float _sub_110;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_110) : "f"(keep_176), "f"(_sub_111));
                        float _sub_112;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(received_180), "f"(bb_183));
                        float e_184;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_184) : "f"(_sub_110), "f"(_sub_112));
                        half_s3[1] = s_182;
                        float _add_114;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(keep_err_178), "f"(received_err_181));
                        float _add_113;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_add_114), "f"(e_184));
                        half_e3[1] = _add_113;
                        float half_s4[1];
                        float half_e4[1];
                        float keep_185 = ((lane_in_warp / 8 % 2 == 1) ? half_s3[1] : half_s3[0]);
                        float send_186 = ((lane_in_warp / 8 % 2 == 1) ? half_s3[0] : half_s3[1]);
                        float keep_err_187 = ((lane_in_warp / 8 % 2 == 1) ? half_e3[1] : half_e3[0]);
                        float send_err_188 = ((lane_in_warp / 8 % 2 == 1) ? half_e3[0] : half_e3[1]);
                        float received_189 = __shfl_xor_sync(0xFFFFFFFF, send_186, 8);
                        float received_err_190 = __shfl_xor_sync(0xFFFFFFFF, send_err_188, 8);
                        float s_191;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_191) : "f"(keep_185), "f"(received_189));
                        float bb_192;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_192) : "f"(s_191), "f"(keep_185));
                        float _sub_116;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_116) : "f"(s_191), "f"(bb_192));
                        float _sub_115;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_115) : "f"(keep_185), "f"(_sub_116));
                        float _sub_117;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_117) : "f"(received_189), "f"(bb_192));
                        float e_193;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_193) : "f"(_sub_115), "f"(_sub_117));
                        half_s4[0] = s_191;
                        float _add_119;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(keep_err_187), "f"(received_err_190));
                        float _add_118;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_add_119), "f"(e_193));
                        half_e4[0] = _add_118;
                        float received5 = __shfl_xor_sync(0xFFFFFFFF, half_s4[0], 16);
                        float received5_err = __shfl_xor_sync(0xFFFFFFFF, half_e4[0], 16);
                        float s_194;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_194) : "f"(half_s4[0]), "f"(received5));
                        float bb_195;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_195) : "f"(s_194), "f"(half_s4[0]));
                        float _sub_121;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121) : "f"(s_194), "f"(bb_195));
                        float _sub_120;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(half_s4[0]), "f"(_sub_121));
                        float _sub_122;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_122) : "f"(received5), "f"(bb_195));
                        float e_196;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_196) : "f"(_sub_120), "f"(_sub_122));
                        float _add_123;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(half_e4[0]), "f"(received5_err));
                        float total_err;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_err) : "f"(_add_123), "f"(e_196));
                        float total_197;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_197) : "f"(s_194), "f"(total_err));
                        total_1 = total_197;
                    } else {
                        unsigned int row_11 = token % 64;
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
                        unsigned long long record_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_1) * 1536 + (unsigned long long)(row_11 * 24);
                        float _vec_load_6[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + (unsigned long long)col0) + 0);
                            _vec_load_6[0 + 0] = _v4.x;
                            _vec_load_6[0 + 1] = _v4.y;
                            _vec_load_6[0 + 2] = _v4.z;
                            _vec_load_6[0 + 3] = _v4.w;
                        }
                        float value_11 = ((split_1 < num_splits) ? _vec_load_6[0] : 0.0f);
                        sums_1[0] = value_11;
                        float value_0_1 = ((split_1 < num_splits) ? _vec_load_6[1] : 0.0f);
                        sums_1[8] = value_0_1;
                        float value_1_2 = ((split_1 < num_splits) ? _vec_load_6[2] : 0.0f);
                        sums_1[4] = value_1_2;
                        float value_2_1 = ((split_1 < num_splits) ? _vec_load_6[3] : 0.0f);
                        sums_1[12] = value_2_1;
                        float _vec_load_7[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_7[0 + 0] = _v4.x;
                            _vec_load_7[0 + 1] = _v4.y;
                            _vec_load_7[0 + 2] = _v4.z;
                            _vec_load_7[0 + 3] = _v4.w;
                        }
                        float value_3_1 = ((split_1 < num_splits) ? _vec_load_7[0] : 0.0f);
                        sums_1[2] = value_3_1;
                        float value_4_1 = ((split_1 < num_splits) ? _vec_load_7[1] : 0.0f);
                        sums_1[10] = value_4_1;
                        float value_5_1 = ((split_1 < num_splits) ? _vec_load_7[2] : 0.0f);
                        sums_1[6] = value_5_1;
                        float value_6_1 = ((split_1 < num_splits) ? _vec_load_7[3] : 0.0f);
                        sums_1[14] = value_6_1;
                        float _vec_load_8[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_8[0 + 0] = _v4.x;
                            _vec_load_8[0 + 1] = _v4.y;
                            _vec_load_8[0 + 2] = _v4.z;
                            _vec_load_8[0 + 3] = _v4.w;
                        }
                        float value_7_1 = ((split_1 < num_splits) ? _vec_load_8[0] : 0.0f);
                        sums_1[1] = value_7_1;
                        float value_8_2 = ((split_1 < num_splits) ? _vec_load_8[1] : 0.0f);
                        sums_1[9] = value_8_2;
                        float value_9_1 = ((split_1 < num_splits) ? _vec_load_8[2] : 0.0f);
                        sums_1[5] = value_9_1;
                        float value_10_1 = ((split_1 < num_splits) ? _vec_load_8[3] : 0.0f);
                        sums_1[13] = value_10_1;
                        float q_word_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_1 * 64) + (unsigned long long)row_11];
                        float q_value_1 = ((split_1 < num_splits) ? q_word_1 : 0.0f);
                        sums_1[3] = q_value_1;
                        unsigned int split_11_1 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_12_1 = ((split_11_1 < num_splits) ? split_11_1 : 0);
                        unsigned long long record_13_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_12_1) * 1536 + (unsigned long long)(row_11 * 24);
                        float _vec_load_9[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13_1 + (unsigned long long)col0) + 0);
                            _vec_load_9[0 + 0] = _v4.x;
                            _vec_load_9[0 + 1] = _v4.y;
                            _vec_load_9[0 + 2] = _v4.z;
                            _vec_load_9[0 + 3] = _v4.w;
                        }
                        float value_14_1 = ((split_11_1 < num_splits) ? _vec_load_9[0] : 0.0f);
                        float _add_128;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(sums_1[0]), "f"(value_14_1));
                        sums_1[0] = _add_128;
                        float value_15 = ((split_11_1 < num_splits) ? _vec_load_9[1] : 0.0f);
                        float _add_129;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(sums_1[8]), "f"(value_15));
                        sums_1[8] = _add_129;
                        float value_16_1 = ((split_11_1 < num_splits) ? _vec_load_9[2] : 0.0f);
                        float _add_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(sums_1[4]), "f"(value_16_1));
                        sums_1[4] = _add_130;
                        float value_17_1 = ((split_11_1 < num_splits) ? _vec_load_9[3] : 0.0f);
                        float _add_131;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(sums_1[12]), "f"(value_17_1));
                        sums_1[12] = _add_131;
                        float _vec_load_10[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13_1 + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_10[0 + 0] = _v4.x;
                            _vec_load_10[0 + 1] = _v4.y;
                            _vec_load_10[0 + 2] = _v4.z;
                            _vec_load_10[0 + 3] = _v4.w;
                        }
                        float value_18 = ((split_11_1 < num_splits) ? _vec_load_10[0] : 0.0f);
                        float _add_133;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(sums_1[2]), "f"(value_18));
                        sums_1[2] = _add_133;
                        float value_19 = ((split_11_1 < num_splits) ? _vec_load_10[1] : 0.0f);
                        float _add_134;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(sums_1[10]), "f"(value_19));
                        sums_1[10] = _add_134;
                        float value_20_1 = ((split_11_1 < num_splits) ? _vec_load_10[2] : 0.0f);
                        float _add_135;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(sums_1[6]), "f"(value_20_1));
                        sums_1[6] = _add_135;
                        float value_21 = ((split_11_1 < num_splits) ? _vec_load_10[3] : 0.0f);
                        float _add_136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(sums_1[14]), "f"(value_21));
                        sums_1[14] = _add_136;
                        float _vec_load_11[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13_1 + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_11[0 + 0] = _v4.x;
                            _vec_load_11[0 + 1] = _v4.y;
                            _vec_load_11[0 + 2] = _v4.z;
                            _vec_load_11[0 + 3] = _v4.w;
                        }
                        float value_22 = ((split_11_1 < num_splits) ? _vec_load_11[0] : 0.0f);
                        float _add_138;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(sums_1[1]), "f"(value_22));
                        sums_1[1] = _add_138;
                        float value_23 = ((split_11_1 < num_splits) ? _vec_load_11[1] : 0.0f);
                        float _add_139;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(sums_1[9]), "f"(value_23));
                        sums_1[9] = _add_139;
                        float value_24_1 = ((split_11_1 < num_splits) ? _vec_load_11[2] : 0.0f);
                        float _add_140;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(sums_1[5]), "f"(value_24_1));
                        sums_1[5] = _add_140;
                        float value_25 = ((split_11_1 < num_splits) ? _vec_load_11[3] : 0.0f);
                        float _add_141;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(sums_1[13]), "f"(value_25));
                        sums_1[13] = _add_141;
                        float q_word_26 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_12_1 * 64) + (unsigned long long)row_11];
                        float q_value_27 = ((split_11_1 < num_splits) ? q_word_26 : 0.0f);
                        float _add_142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(sums_1[3]), "f"(q_value_27));
                        sums_1[3] = _add_142;
                        float half_l1[8];
                        float keep_1 = ((lane_in_warp % 2 == 1) ? sums_1[8] : sums_1[0]);
                        float send_1 = ((lane_in_warp % 2 == 1) ? sums_1[0] : sums_1[8]);
                        float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, send_1, 1);
                        float _add_143;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(keep_1), "f"(_shfl_xor_sync_144));
                        half_l1[0] = _add_143;
                        float keep_28 = ((lane_in_warp % 2 == 1) ? sums_1[9] : sums_1[1]);
                        float send_29 = ((lane_in_warp % 2 == 1) ? sums_1[1] : sums_1[9]);
                        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, send_29, 1);
                        float _add_145;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(keep_28), "f"(_shfl_xor_sync_146));
                        half_l1[1] = _add_145;
                        float keep_30 = ((lane_in_warp % 2 == 1) ? sums_1[10] : sums_1[2]);
                        float send_31 = ((lane_in_warp % 2 == 1) ? sums_1[2] : sums_1[10]);
                        float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, send_31, 1);
                        float _add_147;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(keep_30), "f"(_shfl_xor_sync_148));
                        half_l1[2] = _add_147;
                        float keep_32 = ((lane_in_warp % 2 == 1) ? sums_1[11] : sums_1[3]);
                        float send_33 = ((lane_in_warp % 2 == 1) ? sums_1[3] : sums_1[11]);
                        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, send_33, 1);
                        float _add_149;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(keep_32), "f"(_shfl_xor_sync_150));
                        half_l1[3] = _add_149;
                        float keep_34 = ((lane_in_warp % 2 == 1) ? sums_1[12] : sums_1[4]);
                        float send_35 = ((lane_in_warp % 2 == 1) ? sums_1[4] : sums_1[12]);
                        float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, send_35, 1);
                        float _add_151;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(keep_34), "f"(_shfl_xor_sync_152));
                        half_l1[4] = _add_151;
                        float keep_36 = ((lane_in_warp % 2 == 1) ? sums_1[13] : sums_1[5]);
                        float send_37 = ((lane_in_warp % 2 == 1) ? sums_1[5] : sums_1[13]);
                        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, send_37, 1);
                        float _add_153;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(keep_36), "f"(_shfl_xor_sync_154));
                        half_l1[5] = _add_153;
                        float keep_38 = ((lane_in_warp % 2 == 1) ? sums_1[14] : sums_1[6]);
                        float send_39 = ((lane_in_warp % 2 == 1) ? sums_1[6] : sums_1[14]);
                        float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, send_39, 1);
                        float _add_155;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(keep_38), "f"(_shfl_xor_sync_156));
                        half_l1[6] = _add_155;
                        float keep_40 = ((lane_in_warp % 2 == 1) ? sums_1[15] : sums_1[7]);
                        float send_41 = ((lane_in_warp % 2 == 1) ? sums_1[7] : sums_1[15]);
                        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, send_41, 1);
                        float _add_157;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(keep_40), "f"(_shfl_xor_sync_158));
                        half_l1[7] = _add_157;
                        float half_l2[4];
                        float keep_42 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[4] : half_l1[0]);
                        float send_43 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[0] : half_l1[4]);
                        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, send_43, 2);
                        float _add_159;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(keep_42), "f"(_shfl_xor_sync_160));
                        half_l2[0] = _add_159;
                        float keep_44 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[5] : half_l1[1]);
                        float send_45 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[1] : half_l1[5]);
                        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, send_45, 2);
                        float _add_161;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(keep_44), "f"(_shfl_xor_sync_162));
                        half_l2[1] = _add_161;
                        float keep_46 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[6] : half_l1[2]);
                        float send_47 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[2] : half_l1[6]);
                        float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, send_47, 2);
                        float _add_163;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(keep_46), "f"(_shfl_xor_sync_164));
                        half_l2[2] = _add_163;
                        float keep_48 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[7] : half_l1[3]);
                        float send_49 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[3] : half_l1[7]);
                        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, send_49, 2);
                        float _add_165;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(keep_48), "f"(_shfl_xor_sync_166));
                        half_l2[3] = _add_165;
                        float half_l3[2];
                        float keep_50 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[2] : half_l2[0]);
                        float send_51 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[0] : half_l2[2]);
                        float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, send_51, 4);
                        float _add_167;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(keep_50), "f"(_shfl_xor_sync_168));
                        half_l3[0] = _add_167;
                        float keep_52 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[3] : half_l2[1]);
                        float send_53 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[1] : half_l2[3]);
                        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, send_53, 4);
                        float _add_169;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(keep_52), "f"(_shfl_xor_sync_170));
                        half_l3[1] = _add_169;
                        float half_l4[1];
                        float keep_54 = ((lane_in_warp / 8 % 2 == 1) ? half_l3[1] : half_l3[0]);
                        float send_55 = ((lane_in_warp / 8 % 2 == 1) ? half_l3[0] : half_l3[1]);
                        float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, send_55, 8);
                        float _add_171;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(keep_54), "f"(_shfl_xor_sync_172));
                        half_l4[0] = _add_171;
                        float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, half_l4[0], 16);
                        float total_56;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_56) : "f"(half_l4[0]), "f"(_shfl_xor_sync_173));
                        total_1 = total_56;
#endif
                    }
#if __CUDA_ARCH__ == 1070
                    float value_1 = ((split < num_splits) ? _vec_load_0[0] : 0.0f);
                    sums[0] = value_1;
                    float value_0 = ((split < num_splits) ? _vec_load_0[1] : 0.0f);
                    sums[16] = value_0;
                    float value_1_1 = ((split < num_splits) ? _vec_load_0[2] : 0.0f);
                    sums[8] = value_1_1;
                    float value_2 = ((split < num_splits) ? _vec_load_0[3] : 0.0f);
                    sums[24] = value_2;
                    float _vec_load_1[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 4) + 0);
                        _vec_load_1[0 + 0] = _v4.x;
                        _vec_load_1[0 + 1] = _v4.y;
                        _vec_load_1[0 + 2] = _v4.z;
                        _vec_load_1[0 + 3] = _v4.w;
#else
                    float total_0_1 = total_1;
                    if (flag == 1 && lane_in_warp < 12) {
                        pair_sums[w * 16 + lane_in_warp] = total_0_1;
#endif
                    }
#if __CUDA_ARCH__ == 1070
                    float value_3 = ((split < num_splits) ? _vec_load_1[0] : 0.0f);
                    sums[4] = value_3;
                    float value_4 = ((split < num_splits) ? _vec_load_1[1] : 0.0f);
                    sums[20] = value_4;
                    float value_5 = ((split < num_splits) ? _vec_load_1[2] : 0.0f);
                    sums[12] = value_5;
                    float value_6 = ((split < num_splits) ? _vec_load_1[3] : 0.0f);
                    sums[28] = value_6;
                    float _vec_load_2[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 8) + 0);
                        _vec_load_2[0 + 0] = _v4.x;
                        _vec_load_2[0 + 1] = _v4.y;
                        _vec_load_2[0 + 2] = _v4.z;
                        _vec_load_2[0 + 3] = _v4.w;
                    }
                    float value_7 = ((split < num_splits) ? _vec_load_2[0] : 0.0f);
                    sums[2] = value_7;
                    float value_8_1 = ((split < num_splits) ? _vec_load_2[1] : 0.0f);
                    sums[18] = value_8_1;
                    float value_9 = ((split < num_splits) ? _vec_load_2[2] : 0.0f);
                    sums[10] = value_9;
                    float value_10 = ((split < num_splits) ? _vec_load_2[3] : 0.0f);
                    sums[26] = value_10;
                    float _vec_load_3[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 12) + 0);
                        _vec_load_3[0 + 0] = _v4.x;
                        _vec_load_3[0 + 1] = _v4.y;
                        _vec_load_3[0 + 2] = _v4.z;
                        _vec_load_3[0 + 3] = _v4.w;
                    }
                    float value_11 = ((split < num_splits) ? _vec_load_3[0] : 0.0f);
                    sums[6] = value_11;
                    float value_12 = ((split < num_splits) ? _vec_load_3[1] : 0.0f);
                    sums[22] = value_12;
                    float value_13 = ((split < num_splits) ? _vec_load_3[2] : 0.0f);
                    sums[14] = value_13;
                    float value_14 = ((split < num_splits) ? _vec_load_3[3] : 0.0f);
                    sums[30] = value_14;
                    float _vec_load_4[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 16) + 0);
                        _vec_load_4[0 + 0] = _v4.x;
                        _vec_load_4[0 + 1] = _v4.y;
                        _vec_load_4[0 + 2] = _v4.z;
                        _vec_load_4[0 + 3] = _v4.w;
                    }
                    float value_15 = ((split < num_splits) ? _vec_load_4[0] : 0.0f);
                    sums[1] = value_15;
                    float value_16 = ((split < num_splits) ? _vec_load_4[1] : 0.0f);
                    sums[17] = value_16;
                    float value_17_1 = ((split < num_splits) ? _vec_load_4[2] : 0.0f);
                    sums[9] = value_17_1;
                    float value_18 = ((split < num_splits) ? _vec_load_4[3] : 0.0f);
                    sums[25] = value_18;
                    float _vec_load_5[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 20) + 0);
                        _vec_load_5[0 + 0] = _v4.x;
                        _vec_load_5[0 + 1] = _v4.y;
                        _vec_load_5[0 + 2] = _v4.z;
                        _vec_load_5[0 + 3] = _v4.w;
                    }
                    float value_19 = ((split < num_splits) ? _vec_load_5[0] : 0.0f);
                    sums[5] = value_19;
                    float value_20 = ((split < num_splits) ? _vec_load_5[1] : 0.0f);
                    sums[21] = value_20;
                    float value_21 = ((split < num_splits) ? _vec_load_5[2] : 0.0f);
                    sums[13] = value_21;
                    float value_22 = ((split < num_splits) ? _vec_load_5[3] : 0.0f);
                    sums[29] = value_22;
                    float q_word = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split * 64) + (unsigned long long)row_8];
                    float q_value = ((split < num_splits) ? q_word : 0.0f);
                    sums[3] = q_value;
                    unsigned int split_23 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_24 = ((split_23 < num_splits) ? split_23 : 0);
                    unsigned long long record_25 = ((unsigned long long)first_task + (unsigned long long)safe_split_24) * 1536 + (unsigned long long)(row_8 * 24);
                    float _vec_load_6[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25 + 0);
                        _vec_load_6[0 + 0] = _v4.x;
                        _vec_load_6[0 + 1] = _v4.y;
                        _vec_load_6[0 + 2] = _v4.z;
                        _vec_load_6[0 + 3] = _v4.w;
                    }
                    float value_26_1 = ((split_23 < num_splits) ? _vec_load_6[0] : 0.0f);
                    float _add_7_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7_1) : "f"(sums[0]), "f"(value_26_1));
                    sums[0] = _add_7_1;
                    float value_27 = ((split_23 < num_splits) ? _vec_load_6[1] : 0.0f);
                    float _add_8_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8_1) : "f"(sums[16]), "f"(value_27));
                    sums[16] = _add_8_1;
                    float value_28 = ((split_23 < num_splits) ? _vec_load_6[2] : 0.0f);
                    float _add_9_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9_1) : "f"(sums[8]), "f"(value_28));
                    sums[8] = _add_9_1;
                    float value_29 = ((split_23 < num_splits) ? _vec_load_6[3] : 0.0f);
                    float _add_10_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10_1) : "f"(sums[24]), "f"(value_29));
                    sums[24] = _add_10_1;
                    float _vec_load_7[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 4) + 0);
                        _vec_load_7[0 + 0] = _v4.x;
                        _vec_load_7[0 + 1] = _v4.y;
                        _vec_load_7[0 + 2] = _v4.z;
                        _vec_load_7[0 + 3] = _v4.w;
                    }
                    float value_30 = ((split_23 < num_splits) ? _vec_load_7[0] : 0.0f);
                    float _add_12_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12_1) : "f"(sums[4]), "f"(value_30));
                    sums[4] = _add_12_1;
                    float value_31 = ((split_23 < num_splits) ? _vec_load_7[1] : 0.0f);
                    float _add_13_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13_1) : "f"(sums[20]), "f"(value_31));
                    sums[20] = _add_13_1;
                    float value_32 = ((split_23 < num_splits) ? _vec_load_7[2] : 0.0f);
                    float _add_14_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14_1) : "f"(sums[12]), "f"(value_32));
                    sums[12] = _add_14_1;
                    float value_33 = ((split_23 < num_splits) ? _vec_load_7[3] : 0.0f);
                    float _add_15_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15_1) : "f"(sums[28]), "f"(value_33));
                    sums[28] = _add_15_1;
                    float _vec_load_8[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 8) + 0);
                        _vec_load_8[0 + 0] = _v4.x;
                        _vec_load_8[0 + 1] = _v4.y;
                        _vec_load_8[0 + 2] = _v4.z;
                        _vec_load_8[0 + 3] = _v4.w;
                    }
                    float value_34 = ((split_23 < num_splits) ? _vec_load_8[0] : 0.0f);
                    float _add_17_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17_1) : "f"(sums[2]), "f"(value_34));
                    sums[2] = _add_17_1;
                    float value_35 = ((split_23 < num_splits) ? _vec_load_8[1] : 0.0f);
                    float _add_18_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_2) : "f"(sums[18]), "f"(value_35));
                    sums[18] = _add_18_2;
                    float value_36 = ((split_23 < num_splits) ? _vec_load_8[2] : 0.0f);
                    float _add_19_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19_1) : "f"(sums[10]), "f"(value_36));
                    sums[10] = _add_19_1;
                    float value_37 = ((split_23 < num_splits) ? _vec_load_8[3] : 0.0f);
                    float _add_20_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20_1) : "f"(sums[26]), "f"(value_37));
                    sums[26] = _add_20_1;
                    float _vec_load_9[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 12) + 0);
                        _vec_load_9[0 + 0] = _v4.x;
                        _vec_load_9[0 + 1] = _v4.y;
                        _vec_load_9[0 + 2] = _v4.z;
                        _vec_load_9[0 + 3] = _v4.w;
                    }
                    float value_38 = ((split_23 < num_splits) ? _vec_load_9[0] : 0.0f);
                    float _add_22_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22_1) : "f"(sums[6]), "f"(value_38));
                    sums[6] = _add_22_1;
                    float value_39 = ((split_23 < num_splits) ? _vec_load_9[1] : 0.0f);
                    float _add_23_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23_1) : "f"(sums[22]), "f"(value_39));
                    sums[22] = _add_23_1;
                    float value_40 = ((split_23 < num_splits) ? _vec_load_9[2] : 0.0f);
                    float _add_24_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24_1) : "f"(sums[14]), "f"(value_40));
                    sums[14] = _add_24_1;
                    float value_41 = ((split_23 < num_splits) ? _vec_load_9[3] : 0.0f);
                    float _add_25_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_2) : "f"(sums[30]), "f"(value_41));
                    sums[30] = _add_25_2;
                    float _vec_load_10[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 16) + 0);
                        _vec_load_10[0 + 0] = _v4.x;
                        _vec_load_10[0 + 1] = _v4.y;
                        _vec_load_10[0 + 2] = _v4.z;
                        _vec_load_10[0 + 3] = _v4.w;
                    }
                    float value_42 = ((split_23 < num_splits) ? _vec_load_10[0] : 0.0f);
                    float _add_27_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27_1) : "f"(sums[1]), "f"(value_42));
                    sums[1] = _add_27_1;
                    float value_43 = ((split_23 < num_splits) ? _vec_load_10[1] : 0.0f);
                    float _add_28_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28_1) : "f"(sums[17]), "f"(value_43));
                    sums[17] = _add_28_1;
                    float value_44_1 = ((split_23 < num_splits) ? _vec_load_10[2] : 0.0f);
                    float _add_29_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_1) : "f"(sums[9]), "f"(value_44_1));
                    sums[9] = _add_29_1;
                    float value_45 = ((split_23 < num_splits) ? _vec_load_10[3] : 0.0f);
                    float _add_30_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30_1) : "f"(sums[25]), "f"(value_45));
                    sums[25] = _add_30_1;
                    float _vec_load_11[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 20) + 0);
                        _vec_load_11[0 + 0] = _v4.x;
                        _vec_load_11[0 + 1] = _v4.y;
                        _vec_load_11[0 + 2] = _v4.z;
                        _vec_load_11[0 + 3] = _v4.w;
                    }
                    float value_46 = ((split_23 < num_splits) ? _vec_load_11[0] : 0.0f);
                    float _add_32_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_2) : "f"(sums[5]), "f"(value_46));
                    sums[5] = _add_32_2;
                    float value_47 = ((split_23 < num_splits) ? _vec_load_11[1] : 0.0f);
                    float _add_33_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33_1) : "f"(sums[21]), "f"(value_47));
                    sums[21] = _add_33_1;
                    float value_48 = ((split_23 < num_splits) ? _vec_load_11[2] : 0.0f);
                    float _add_34_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34_1) : "f"(sums[13]), "f"(value_48));
                    sums[13] = _add_34_1;
                    float value_49 = ((split_23 < num_splits) ? _vec_load_11[3] : 0.0f);
                    float _add_35_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35_1) : "f"(sums[29]), "f"(value_49));
                    sums[29] = _add_35_1;
                    float q_word_50 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split_24 * 64) + (unsigned long long)row_8];
                    float q_value_51 = ((split_23 < num_splits) ? q_word_50 : 0.0f);
                    float _add_36;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(sums[3]), "f"(q_value_51));
                    sums[3] = _add_36;
                    float fold_level1[16];
                    float keep = ((lane_in_warp % 2 == 1) ? sums[16] : sums[0]);
                    float send = ((lane_in_warp % 2 == 1) ? sums[0] : sums[16]);
                    float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                    float _add_37;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(keep), "f"(_shfl_xor_sync_38));
                    fold_level1[0] = _add_37;
                    float keep_52 = ((lane_in_warp % 2 == 1) ? sums[17] : sums[1]);
                    float send_53 = ((lane_in_warp % 2 == 1) ? sums[1] : sums[17]);
                    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, send_53, 1);
                    float _add_39_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_1) : "f"(keep_52), "f"(_shfl_xor_sync_40));
                    fold_level1[1] = _add_39_1;
                    float keep_54 = ((lane_in_warp % 2 == 1) ? sums[18] : sums[2]);
                    float send_55 = ((lane_in_warp % 2 == 1) ? sums[2] : sums[18]);
                    float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, send_55, 1);
                    float _add_41;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(keep_54), "f"(_shfl_xor_sync_42));
                    fold_level1[2] = _add_41;
                    float keep_56 = ((lane_in_warp % 2 == 1) ? sums[19] : sums[3]);
                    float send_57 = ((lane_in_warp % 2 == 1) ? sums[3] : sums[19]);
                    float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, send_57, 1);
                    float _add_43;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(keep_56), "f"(_shfl_xor_sync_44));
                    fold_level1[3] = _add_43;
                    float keep_58 = ((lane_in_warp % 2 == 1) ? sums[20] : sums[4]);
                    float send_59 = ((lane_in_warp % 2 == 1) ? sums[4] : sums[20]);
                    float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, send_59, 1);
                    float _add_45;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(keep_58), "f"(_shfl_xor_sync_46));
                    fold_level1[4] = _add_45;
                    float keep_60 = ((lane_in_warp % 2 == 1) ? sums[21] : sums[5]);
                    float send_61 = ((lane_in_warp % 2 == 1) ? sums[5] : sums[21]);
                    float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, send_61, 1);
                    float _add_47;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(keep_60), "f"(_shfl_xor_sync_48));
                    fold_level1[5] = _add_47;
                    float keep_62 = ((lane_in_warp % 2 == 1) ? sums[22] : sums[6]);
                    float send_63 = ((lane_in_warp % 2 == 1) ? sums[6] : sums[22]);
                    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, send_63, 1);
                    float _add_49;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(keep_62), "f"(_shfl_xor_sync_50));
                    fold_level1[6] = _add_49;
                    float keep_64 = ((lane_in_warp % 2 == 1) ? sums[23] : sums[7]);
                    float send_65 = ((lane_in_warp % 2 == 1) ? sums[7] : sums[23]);
                    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, send_65, 1);
                    float _add_51;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(keep_64), "f"(_shfl_xor_sync_52));
                    fold_level1[7] = _add_51;
                    float keep_66 = ((lane_in_warp % 2 == 1) ? sums[24] : sums[8]);
                    float send_67 = ((lane_in_warp % 2 == 1) ? sums[8] : sums[24]);
                    float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, send_67, 1);
                    float _add_53_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_1) : "f"(keep_66), "f"(_shfl_xor_sync_54));
                    fold_level1[8] = _add_53_1;
                    float keep_68 = ((lane_in_warp % 2 == 1) ? sums[25] : sums[9]);
                    float send_69 = ((lane_in_warp % 2 == 1) ? sums[9] : sums[25]);
                    float _shfl_xor_sync_56_1 = __shfl_xor_sync(0xFFFFFFFF, send_69, 1);
                    float _add_55;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(keep_68), "f"(_shfl_xor_sync_56_1));
                    fold_level1[9] = _add_55;
                    float keep_70 = ((lane_in_warp % 2 == 1) ? sums[26] : sums[10]);
                    float send_71 = ((lane_in_warp % 2 == 1) ? sums[10] : sums[26]);
                    float _shfl_xor_sync_58_1 = __shfl_xor_sync(0xFFFFFFFF, send_71, 1);
                    float _add_57;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(keep_70), "f"(_shfl_xor_sync_58_1));
                    fold_level1[10] = _add_57;
                    float keep_72 = ((lane_in_warp % 2 == 1) ? sums[27] : sums[11]);
                    float send_73 = ((lane_in_warp % 2 == 1) ? sums[11] : sums[27]);
                    float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, send_73, 1);
                    float _add_59;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(keep_72), "f"(_shfl_xor_sync_60));
                    fold_level1[11] = _add_59;
                    float keep_74 = ((lane_in_warp % 2 == 1) ? sums[28] : sums[12]);
                    float send_75 = ((lane_in_warp % 2 == 1) ? sums[12] : sums[28]);
                    float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, send_75, 1);
                    float _add_61;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(keep_74), "f"(_shfl_xor_sync_62));
                    fold_level1[12] = _add_61;
                    float keep_76 = ((lane_in_warp % 2 == 1) ? sums[29] : sums[13]);
                    float send_77 = ((lane_in_warp % 2 == 1) ? sums[13] : sums[29]);
                    float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, send_77, 1);
                    float _add_63;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(keep_76), "f"(_shfl_xor_sync_64));
                    fold_level1[13] = _add_63;
                    float keep_78 = ((lane_in_warp % 2 == 1) ? sums[30] : sums[14]);
                    float send_79 = ((lane_in_warp % 2 == 1) ? sums[14] : sums[30]);
                    float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, send_79, 1);
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(keep_78), "f"(_shfl_xor_sync_66));
                    fold_level1[14] = _add_65;
                    float keep_80 = ((lane_in_warp % 2 == 1) ? sums[31] : sums[15]);
                    float send_81 = ((lane_in_warp % 2 == 1) ? sums[15] : sums[31]);
                    float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, send_81, 1);
                    float _add_67;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(keep_80), "f"(_shfl_xor_sync_68));
                    fold_level1[15] = _add_67;
                    float fold_level2[8];
                    float keep_82 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[8] : fold_level1[0]);
                    float send_83 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[0] : fold_level1[8]);
                    float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, send_83, 2);
                    float _add_69;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(keep_82), "f"(_shfl_xor_sync_70));
                    fold_level2[0] = _add_69;
                    float keep_84 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[9] : fold_level1[1]);
                    float send_85 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[1] : fold_level1[9]);
                    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, send_85, 2);
                    float _add_71;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(keep_84), "f"(_shfl_xor_sync_72));
                    fold_level2[1] = _add_71;
                    float keep_86 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[10] : fold_level1[2]);
                    float send_87 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[2] : fold_level1[10]);
                    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, send_87, 2);
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(keep_86), "f"(_shfl_xor_sync_74));
                    fold_level2[2] = _add_73;
                    float keep_88 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[11] : fold_level1[3]);
                    float send_89 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[3] : fold_level1[11]);
                    float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, send_89, 2);
                    float _add_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(keep_88), "f"(_shfl_xor_sync_76));
                    fold_level2[3] = _add_75;
                    float keep_90 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[12] : fold_level1[4]);
                    float send_91 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[4] : fold_level1[12]);
                    float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, send_91, 2);
                    float _add_77;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(keep_90), "f"(_shfl_xor_sync_78));
                    fold_level2[4] = _add_77;
                    float keep_92 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[13] : fold_level1[5]);
                    float send_93 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[5] : fold_level1[13]);
                    float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, send_93, 2);
                    float _add_79;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(keep_92), "f"(_shfl_xor_sync_80));
                    fold_level2[5] = _add_79;
                    float keep_94 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[14] : fold_level1[6]);
                    float send_95 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[6] : fold_level1[14]);
                    float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, send_95, 2);
                    float _add_81;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(keep_94), "f"(_shfl_xor_sync_82));
                    fold_level2[6] = _add_81;
                    float keep_96 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[15] : fold_level1[7]);
                    float send_97 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[7] : fold_level1[15]);
                    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, send_97, 2);
                    float _add_83;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(keep_96), "f"(_shfl_xor_sync_84));
                    fold_level2[7] = _add_83;
                    float fold_level3[4];
                    float keep_98 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[4] : fold_level2[0]);
                    float send_99 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[0] : fold_level2[4]);
                    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, send_99, 4);
                    float _add_85;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(keep_98), "f"(_shfl_xor_sync_86));
                    fold_level3[0] = _add_85;
                    float keep_100 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[5] : fold_level2[1]);
                    float send_101 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[1] : fold_level2[5]);
                    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, send_101, 4);
                    float _add_87;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(keep_100), "f"(_shfl_xor_sync_88));
                    fold_level3[1] = _add_87;
                    float keep_102 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[6] : fold_level2[2]);
                    float send_103 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[2] : fold_level2[6]);
                    float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, send_103, 4);
                    float _add_89;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(keep_102), "f"(_shfl_xor_sync_90));
                    fold_level3[2] = _add_89;
                    float keep_104 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[7] : fold_level2[3]);
                    float send_105 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[3] : fold_level2[7]);
                    float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, send_105, 4);
                    float _add_91;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(keep_104), "f"(_shfl_xor_sync_92));
                    fold_level3[3] = _add_91;
                    float fold_level4[2];
                    float keep_106 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[2] : fold_level3[0]);
                    float send_107 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[0] : fold_level3[2]);
                    float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, send_107, 8);
                    float _add_93;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(keep_106), "f"(_shfl_xor_sync_94));
                    fold_level4[0] = _add_93;
                    float keep_108 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[3] : fold_level3[1]);
                    float send_109 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[1] : fold_level3[3]);
                    float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, send_109, 8);
                    float _add_95;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(keep_108), "f"(_shfl_xor_sync_96));
                    fold_level4[1] = _add_95;
                    float fold_level5[1];
                    float keep_110 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[1] : fold_level4[0]);
                    float send_111 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[0] : fold_level4[1]);
                    float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, send_111, 16);
                    float _add_97;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(keep_110), "f"(_shfl_xor_sync_98));
                    fold_level5[0] = _add_97;
                    float total_1 = fold_level5[0];
                    float s_112 = ((lane_in_warp < 24) ? total_1 : 0.0f);
                    float q_113 = __shfl_sync(0xFFFFFFFF, total_1, 24);
                    s = s_112;
                    q = q_113;
                } else {
                    unsigned int row_11 = token % 64;
                    unsigned int first_task_1 = mb_1 * num_splits;
                    float sums_1[25];
                    unsigned int split_1 = lane_in_warp * 2;
                    unsigned int safe_split_1 = ((split_1 < num_splits) ? split_1 : 0);
                    unsigned long long record_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_1) * 1536 + (unsigned long long)(row_11 * 24);
#else
                    asm volatile("barrier.sync %0, 64;" :: "r"(4 + w) : "memory");
                    unsigned int pidx = ((lane_in_warp >= 12 && lane_in_warp < 24) ? lane_in_warp - 12 : 0);
                    float partner = pair_sums[w * 16 + pidx];
                    float s_1 = ((lane_in_warp < 12) ? total_0_1 : ((lane_in_warp < 24) ? partner : 0.0f));
                    float q_2 = __shfl_sync(0xFFFFFFFF, total_0_1, 12);
                    s = s_1;
                    q = q_2;
                } else if ((num_tokens + 63) / 64 * num_splits <= num_ctas && num_splits <= 22) {
                    unsigned int row_12 = token % 64;
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
                    unsigned long long record_2 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_2) * 1536 + (unsigned long long)(row_12 * 24);
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
                    float value_23 = ((split_1 < num_splits) ? _vec_load_12[0] : 0.0f);
                    sums_1[0] = value_23;
                    float value_0_1 = ((split_1 < num_splits) ? _vec_load_12[1] : 0.0f);
                    sums_1[1] = value_0_1;
                    float value_1_2 = ((split_1 < num_splits) ? _vec_load_12[2] : 0.0f);
                    sums_1[2] = value_1_2;
                    float value_2_1 = ((split_1 < num_splits) ? _vec_load_12[3] : 0.0f);
                    sums_1[3] = value_2_1;
#else
                    float value_12 = ((split_2 < num_splits) ? _vec_load_12[0] : 0.0f);
                    sums_2[0] = value_12;
                    float value_0_2 = ((split_2 < num_splits) ? _vec_load_12[1] : 0.0f);
                    sums_2[16] = value_0_2;
                    float value_1_3 = ((split_2 < num_splits) ? _vec_load_12[2] : 0.0f);
                    sums_2[8] = value_1_3;
                    float value_2_2 = ((split_2 < num_splits) ? _vec_load_12[3] : 0.0f);
                    sums_2[24] = value_2_2;
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
                    float value_3_1 = ((split_1 < num_splits) ? _vec_load_13[0] : 0.0f);
                    sums_1[4] = value_3_1;
                    float value_4_1 = ((split_1 < num_splits) ? _vec_load_13[1] : 0.0f);
                    sums_1[5] = value_4_1;
                    float value_5_1 = ((split_1 < num_splits) ? _vec_load_13[2] : 0.0f);
                    sums_1[6] = value_5_1;
                    float value_6_1 = ((split_1 < num_splits) ? _vec_load_13[3] : 0.0f);
                    sums_1[7] = value_6_1;
#else
                    float value_3_2 = ((split_2 < num_splits) ? _vec_load_13[0] : 0.0f);
                    sums_2[4] = value_3_2;
                    float value_4_2 = ((split_2 < num_splits) ? _vec_load_13[1] : 0.0f);
                    sums_2[20] = value_4_2;
                    float value_5_2 = ((split_2 < num_splits) ? _vec_load_13[2] : 0.0f);
                    sums_2[12] = value_5_2;
                    float value_6_2 = ((split_2 < num_splits) ? _vec_load_13[3] : 0.0f);
                    sums_2[28] = value_6_2;
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
                    float value_7_1 = ((split_1 < num_splits) ? _vec_load_14[0] : 0.0f);
                    sums_1[8] = value_7_1;
                    float value_8_2 = ((split_1 < num_splits) ? _vec_load_14[1] : 0.0f);
                    sums_1[9] = value_8_2;
                    float value_9_1 = ((split_1 < num_splits) ? _vec_load_14[2] : 0.0f);
                    sums_1[10] = value_9_1;
                    float value_10_1 = ((split_1 < num_splits) ? _vec_load_14[3] : 0.0f);
                    sums_1[11] = value_10_1;
#else
                    float value_7_2 = ((split_2 < num_splits) ? _vec_load_14[0] : 0.0f);
                    sums_2[2] = value_7_2;
                    float value_8_3 = ((split_2 < num_splits) ? _vec_load_14[1] : 0.0f);
                    sums_2[18] = value_8_3;
                    float value_9_2 = ((split_2 < num_splits) ? _vec_load_14[2] : 0.0f);
                    sums_2[10] = value_9_2;
                    float value_10_2 = ((split_2 < num_splits) ? _vec_load_14[3] : 0.0f);
                    sums_2[26] = value_10_2;
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
                    float value_11_1 = ((split_1 < num_splits) ? _vec_load_15[0] : 0.0f);
                    sums_1[12] = value_11_1;
                    float value_12_1 = ((split_1 < num_splits) ? _vec_load_15[1] : 0.0f);
                    sums_1[13] = value_12_1;
                    float value_13_1 = ((split_1 < num_splits) ? _vec_load_15[2] : 0.0f);
                    sums_1[14] = value_13_1;
                    float value_14_1 = ((split_1 < num_splits) ? _vec_load_15[3] : 0.0f);
                    sums_1[15] = value_14_1;
#else
                    float value_11_1 = ((split_2 < num_splits) ? _vec_load_15[0] : 0.0f);
                    sums_2[6] = value_11_1;
                    float value_12_1 = ((split_2 < num_splits) ? _vec_load_15[1] : 0.0f);
                    sums_2[22] = value_12_1;
                    float value_13 = ((split_2 < num_splits) ? _vec_load_15[2] : 0.0f);
                    sums_2[14] = value_13;
                    float value_14_2 = ((split_2 < num_splits) ? _vec_load_15[3] : 0.0f);
                    sums_2[30] = value_14_2;
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
                    float value_15_1 = ((split_1 < num_splits) ? _vec_load_16[0] : 0.0f);
                    sums_1[16] = value_15_1;
                    float value_16_1 = ((split_1 < num_splits) ? _vec_load_16[1] : 0.0f);
                    sums_1[17] = value_16_1;
                    float value_17_2 = ((split_1 < num_splits) ? _vec_load_16[2] : 0.0f);
                    sums_1[18] = value_17_2;
                    float value_18_1 = ((split_1 < num_splits) ? _vec_load_16[3] : 0.0f);
                    sums_1[19] = value_18_1;
#else
                    float value_15_1 = ((split_2 < num_splits) ? _vec_load_16[0] : 0.0f);
                    sums_2[1] = value_15_1;
                    float value_16_2 = ((split_2 < num_splits) ? _vec_load_16[1] : 0.0f);
                    sums_2[17] = value_16_2;
                    float value_17_2 = ((split_2 < num_splits) ? _vec_load_16[2] : 0.0f);
                    sums_2[9] = value_17_2;
                    float value_18_1 = ((split_2 < num_splits) ? _vec_load_16[3] : 0.0f);
                    sums_2[25] = value_18_1;
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
                    float value_19_1 = ((split_1 < num_splits) ? _vec_load_17[0] : 0.0f);
                    sums_1[20] = value_19_1;
                    float value_20_1 = ((split_1 < num_splits) ? _vec_load_17[1] : 0.0f);
                    sums_1[21] = value_20_1;
                    float value_21_1 = ((split_1 < num_splits) ? _vec_load_17[2] : 0.0f);
                    sums_1[22] = value_21_1;
                    float value_22_1 = ((split_1 < num_splits) ? _vec_load_17[3] : 0.0f);
                    sums_1[23] = value_22_1;
                    float q_word_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_1 * 64) + (unsigned long long)row_11];
                    float q_value_1 = ((split_1 < num_splits) ? q_word_1 : 0.0f);
                    sums_1[24] = q_value_1;
                    unsigned int split_23_1 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_24_1 = ((split_23_1 < num_splits) ? split_23_1 : 0);
                    unsigned long long record_25_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_24_1) * 1536 + (unsigned long long)(row_11 * 24);
#else
                    float value_19_1 = ((split_2 < num_splits) ? _vec_load_17[0] : 0.0f);
                    sums_2[5] = value_19_1;
                    float value_20_2 = ((split_2 < num_splits) ? _vec_load_17[1] : 0.0f);
                    sums_2[21] = value_20_2;
                    float value_21_1 = ((split_2 < num_splits) ? _vec_load_17[2] : 0.0f);
                    sums_2[13] = value_21_1;
                    float value_22_1 = ((split_2 < num_splits) ? _vec_load_17[3] : 0.0f);
                    sums_2[29] = value_22_1;
                    float q_word_2 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_2 * 64) + (unsigned long long)row_12];
                    float q_value_2 = ((split_2 < num_splits) ? q_word_2 : 0.0f);
                    sums_2[3] = q_value_2;
                    unsigned int split_23 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_24 = ((split_23 < num_splits) ? split_23 : 0);
                    unsigned long long record_25 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_24) * 1536 + (unsigned long long)(row_12 * 24);
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
                    float value_26_2 = ((split_23_1 < num_splits) ? _vec_load_18[0] : 0.0f);
                    float _add_106;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(sums_1[0]), "f"(value_26_2));
                    sums_1[0] = _add_106;
                    float value_27_1 = ((split_23_1 < num_splits) ? _vec_load_18[1] : 0.0f);
                    float _add_107;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(sums_1[1]), "f"(value_27_1));
                    sums_1[1] = _add_107;
                    float value_28_1 = ((split_23_1 < num_splits) ? _vec_load_18[2] : 0.0f);
                    float _add_108;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(sums_1[2]), "f"(value_28_1));
                    sums_1[2] = _add_108;
                    float value_29_1 = ((split_23_1 < num_splits) ? _vec_load_18[3] : 0.0f);
                    float _add_109;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(sums_1[3]), "f"(value_29_1));
                    sums_1[3] = _add_109;
#else
                    float value_26_1 = ((split_23 < num_splits) ? _vec_load_18[0] : 0.0f);
                    float _add_181;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(sums_2[0]), "f"(value_26_1));
                    sums_2[0] = _add_181;
                    float value_27 = ((split_23 < num_splits) ? _vec_load_18[1] : 0.0f);
                    float _add_182;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(sums_2[16]), "f"(value_27));
                    sums_2[16] = _add_182;
                    float value_28_1 = ((split_23 < num_splits) ? _vec_load_18[2] : 0.0f);
                    float _add_183;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(sums_2[8]), "f"(value_28_1));
                    sums_2[8] = _add_183;
                    float value_29 = ((split_23 < num_splits) ? _vec_load_18[3] : 0.0f);
                    float _add_184;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(sums_2[24]), "f"(value_29));
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
                    float value_30_1 = ((split_23_1 < num_splits) ? _vec_load_19[0] : 0.0f);
                    float _add_111;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(sums_1[4]), "f"(value_30_1));
                    sums_1[4] = _add_111;
                    float value_31_1 = ((split_23_1 < num_splits) ? _vec_load_19[1] : 0.0f);
                    float _add_112;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(sums_1[5]), "f"(value_31_1));
                    sums_1[5] = _add_112;
                    float value_32_1 = ((split_23_1 < num_splits) ? _vec_load_19[2] : 0.0f);
                    float _add_113;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(sums_1[6]), "f"(value_32_1));
                    sums_1[6] = _add_113;
                    float value_33_1 = ((split_23_1 < num_splits) ? _vec_load_19[3] : 0.0f);
                    float _add_114;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(sums_1[7]), "f"(value_33_1));
                    sums_1[7] = _add_114;
#else
                    float value_30 = ((split_23 < num_splits) ? _vec_load_19[0] : 0.0f);
                    float _add_186;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(sums_2[4]), "f"(value_30));
                    sums_2[4] = _add_186;
                    float value_31 = ((split_23 < num_splits) ? _vec_load_19[1] : 0.0f);
                    float _add_187;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(sums_2[20]), "f"(value_31));
                    sums_2[20] = _add_187;
                    float value_32_1 = ((split_23 < num_splits) ? _vec_load_19[2] : 0.0f);
                    float _add_188;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(sums_2[12]), "f"(value_32_1));
                    sums_2[12] = _add_188;
                    float value_33 = ((split_23 < num_splits) ? _vec_load_19[3] : 0.0f);
                    float _add_189;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(sums_2[28]), "f"(value_33));
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
                    float value_34_1 = ((split_23_1 < num_splits) ? _vec_load_20[0] : 0.0f);
                    float _add_116;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(sums_1[8]), "f"(value_34_1));
                    sums_1[8] = _add_116;
                    float value_35_1 = ((split_23_1 < num_splits) ? _vec_load_20[1] : 0.0f);
                    float _add_117;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(sums_1[9]), "f"(value_35_1));
                    sums_1[9] = _add_117;
                    float value_36_1 = ((split_23_1 < num_splits) ? _vec_load_20[2] : 0.0f);
                    float _add_118;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(sums_1[10]), "f"(value_36_1));
                    sums_1[10] = _add_118;
                    float value_37_1 = ((split_23_1 < num_splits) ? _vec_load_20[3] : 0.0f);
                    float _add_119;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(sums_1[11]), "f"(value_37_1));
                    sums_1[11] = _add_119;
#else
                    float value_34 = ((split_23 < num_splits) ? _vec_load_20[0] : 0.0f);
                    float _add_191;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(sums_2[2]), "f"(value_34));
                    sums_2[2] = _add_191;
                    float value_35 = ((split_23 < num_splits) ? _vec_load_20[1] : 0.0f);
                    float _add_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(sums_2[18]), "f"(value_35));
                    sums_2[18] = _add_192;
                    float value_36_1 = ((split_23 < num_splits) ? _vec_load_20[2] : 0.0f);
                    float _add_193;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(sums_2[10]), "f"(value_36_1));
                    sums_2[10] = _add_193;
                    float value_37 = ((split_23 < num_splits) ? _vec_load_20[3] : 0.0f);
                    float _add_194;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(sums_2[26]), "f"(value_37));
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
                    float value_38_1 = ((split_23_1 < num_splits) ? _vec_load_21[0] : 0.0f);
                    float _add_121;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(sums_1[12]), "f"(value_38_1));
                    sums_1[12] = _add_121;
                    float value_39_1 = ((split_23_1 < num_splits) ? _vec_load_21[1] : 0.0f);
                    float _add_122;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(sums_1[13]), "f"(value_39_1));
                    sums_1[13] = _add_122;
                    float value_40_1 = ((split_23_1 < num_splits) ? _vec_load_21[2] : 0.0f);
                    float _add_123;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(sums_1[14]), "f"(value_40_1));
                    sums_1[14] = _add_123;
                    float value_41_1 = ((split_23_1 < num_splits) ? _vec_load_21[3] : 0.0f);
                    float _add_124;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(sums_1[15]), "f"(value_41_1));
                    sums_1[15] = _add_124;
#else
                    float value_38 = ((split_23 < num_splits) ? _vec_load_21[0] : 0.0f);
                    float _add_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(sums_2[6]), "f"(value_38));
                    sums_2[6] = _add_196;
                    float value_39 = ((split_23 < num_splits) ? _vec_load_21[1] : 0.0f);
                    float _add_197;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(sums_2[22]), "f"(value_39));
                    sums_2[22] = _add_197;
                    float value_40_1 = ((split_23 < num_splits) ? _vec_load_21[2] : 0.0f);
                    float _add_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(sums_2[14]), "f"(value_40_1));
                    sums_2[14] = _add_198;
                    float value_41 = ((split_23 < num_splits) ? _vec_load_21[3] : 0.0f);
                    float _add_199;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(sums_2[30]), "f"(value_41));
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
                    float value_42_1 = ((split_23_1 < num_splits) ? _vec_load_22[0] : 0.0f);
                    float _add_126;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(sums_1[16]), "f"(value_42_1));
                    sums_1[16] = _add_126;
                    float value_43_1 = ((split_23_1 < num_splits) ? _vec_load_22[1] : 0.0f);
                    float _add_127;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(sums_1[17]), "f"(value_43_1));
                    sums_1[17] = _add_127;
                    float value_44_2 = ((split_23_1 < num_splits) ? _vec_load_22[2] : 0.0f);
                    float _add_128;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(sums_1[18]), "f"(value_44_2));
                    sums_1[18] = _add_128;
                    float value_45_1 = ((split_23_1 < num_splits) ? _vec_load_22[3] : 0.0f);
                    float _add_129;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(sums_1[19]), "f"(value_45_1));
                    sums_1[19] = _add_129;
#else
                    float value_42 = ((split_23 < num_splits) ? _vec_load_22[0] : 0.0f);
                    float _add_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(sums_2[1]), "f"(value_42));
                    sums_2[1] = _add_201;
                    float value_43 = ((split_23 < num_splits) ? _vec_load_22[1] : 0.0f);
                    float _add_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(sums_2[17]), "f"(value_43));
                    sums_2[17] = _add_202;
                    float value_44_2 = ((split_23 < num_splits) ? _vec_load_22[2] : 0.0f);
                    float _add_203;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(sums_2[9]), "f"(value_44_2));
                    sums_2[9] = _add_203;
                    float value_45 = ((split_23 < num_splits) ? _vec_load_22[3] : 0.0f);
                    float _add_204;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(sums_2[25]), "f"(value_45));
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
                    float value_46_1 = ((split_23_1 < num_splits) ? _vec_load_23[0] : 0.0f);
                    float _add_131;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(sums_1[20]), "f"(value_46_1));
                    sums_1[20] = _add_131;
                    float value_47_1 = ((split_23_1 < num_splits) ? _vec_load_23[1] : 0.0f);
                    float _add_132;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(sums_1[21]), "f"(value_47_1));
                    sums_1[21] = _add_132;
                    float value_48_1 = ((split_23_1 < num_splits) ? _vec_load_23[2] : 0.0f);
                    float _add_133;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(sums_1[22]), "f"(value_48_1));
                    sums_1[22] = _add_133;
                    float value_49_1 = ((split_23_1 < num_splits) ? _vec_load_23[3] : 0.0f);
                    float _add_134;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(sums_1[23]), "f"(value_49_1));
                    sums_1[23] = _add_134;
                    float q_word_50_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_24_1 * 64) + (unsigned long long)row_11];
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
                    float value_46 = ((split_23 < num_splits) ? _vec_load_23[0] : 0.0f);
#endif
                    float _add_206;
#if __CUDA_ARCH__ == 1070
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(sums_1[10]), "f"(_shfl_xor_sync_207));
                    sums_1[10] = _add_206;
                    float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 2);
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(sums_2[5]), "f"(value_46));
                    sums_2[5] = _add_206;
                    float value_47 = ((split_23 < num_splits) ? _vec_load_23[1] : 0.0f);
                    float _add_207;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(sums_2[21]), "f"(value_47));
                    sums_2[21] = _add_207;
                    float value_48_1 = ((split_23 < num_splits) ? _vec_load_23[2] : 0.0f);
#endif
                    float _add_208;
#if __CUDA_ARCH__ == 1070
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(sums_1[11]), "f"(_shfl_xor_sync_209));
                    sums_1[11] = _add_208;
                    float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 2);
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(sums_2[13]), "f"(value_48_1));
                    sums_2[13] = _add_208;
                    float value_49 = ((split_23 < num_splits) ? _vec_load_23[3] : 0.0f);
                    float _add_209;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(sums_2[29]), "f"(value_49));
                    sums_2[29] = _add_209;
                    float q_word_50 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_24 * 64) + (unsigned long long)row_12];
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
                float _exp_0 = __nv_expf(_sub_388);
                float _add_36_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36_1) : "f"(1.0f), "f"(_exp_0));
                float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_36_1);
                float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_36_1);
                float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
                float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
                float _add_37_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37_1) : "f"(sig), "f"(hc_pre_eps));
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
                    float keep_2 = ((lane_in_warp % 2 == 1) ? sums_2[16] : sums_2[0]);
                    float send_2 = ((lane_in_warp % 2 == 1) ? sums_2[0] : sums_2[16]);
                    float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, send_2, 1);
                    float _add_211;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(keep_2), "f"(_shfl_xor_sync_212));
                    fold_level1[0] = _add_211;
                    float keep_52_1 = ((lane_in_warp % 2 == 1) ? sums_2[17] : sums_2[1]);
                    float send_53_1 = ((lane_in_warp % 2 == 1) ? sums_2[1] : sums_2[17]);
                    float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, send_53_1, 1);
                    float _add_213;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_213) : "f"(keep_52_1), "f"(_shfl_xor_sync_214));
                    fold_level1[1] = _add_213;
                    float keep_54_1 = ((lane_in_warp % 2 == 1) ? sums_2[18] : sums_2[2]);
                    float send_55_1 = ((lane_in_warp % 2 == 1) ? sums_2[2] : sums_2[18]);
                    float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, send_55_1, 1);
                    float _add_215;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(keep_54_1), "f"(_shfl_xor_sync_216));
                    fold_level1[2] = _add_215;
                    float keep_56 = ((lane_in_warp % 2 == 1) ? sums_2[19] : sums_2[3]);
                    float send_57 = ((lane_in_warp % 2 == 1) ? sums_2[3] : sums_2[19]);
                    float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, send_57, 1);
                    float _add_217;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(keep_56), "f"(_shfl_xor_sync_218));
                    fold_level1[3] = _add_217;
                    float keep_58 = ((lane_in_warp % 2 == 1) ? sums_2[20] : sums_2[4]);
                    float send_59 = ((lane_in_warp % 2 == 1) ? sums_2[4] : sums_2[20]);
                    float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, send_59, 1);
                    float _add_219;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(keep_58), "f"(_shfl_xor_sync_220));
                    fold_level1[4] = _add_219;
                    float keep_60 = ((lane_in_warp % 2 == 1) ? sums_2[21] : sums_2[5]);
                    float send_61 = ((lane_in_warp % 2 == 1) ? sums_2[5] : sums_2[21]);
                    float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, send_61, 1);
                    float _add_221;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(keep_60), "f"(_shfl_xor_sync_222));
                    fold_level1[5] = _add_221;
                    float keep_62 = ((lane_in_warp % 2 == 1) ? sums_2[22] : sums_2[6]);
                    float send_63 = ((lane_in_warp % 2 == 1) ? sums_2[6] : sums_2[22]);
                    float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, send_63, 1);
                    float _add_223;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(keep_62), "f"(_shfl_xor_sync_224));
                    fold_level1[6] = _add_223;
                    float keep_64 = ((lane_in_warp % 2 == 1) ? sums_2[23] : sums_2[7]);
                    float send_65 = ((lane_in_warp % 2 == 1) ? sums_2[7] : sums_2[23]);
                    float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, send_65, 1);
                    float _add_225;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(keep_64), "f"(_shfl_xor_sync_226));
                    fold_level1[7] = _add_225;
                    float keep_66 = ((lane_in_warp % 2 == 1) ? sums_2[24] : sums_2[8]);
                    float send_67 = ((lane_in_warp % 2 == 1) ? sums_2[8] : sums_2[24]);
                    float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, send_67, 1);
                    float _add_227;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(keep_66), "f"(_shfl_xor_sync_228));
                    fold_level1[8] = _add_227;
                    float keep_68_1 = ((lane_in_warp % 2 == 1) ? sums_2[25] : sums_2[9]);
                    float send_69_1 = ((lane_in_warp % 2 == 1) ? sums_2[9] : sums_2[25]);
                    float _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, send_69_1, 1);
                    float _add_229;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(keep_68_1), "f"(_shfl_xor_sync_230));
                    fold_level1[9] = _add_229;
                    float keep_70 = ((lane_in_warp % 2 == 1) ? sums_2[26] : sums_2[10]);
                    float send_71 = ((lane_in_warp % 2 == 1) ? sums_2[10] : sums_2[26]);
                    float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, send_71, 1);
                    float _add_231;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(keep_70), "f"(_shfl_xor_sync_232));
                    fold_level1[10] = _add_231;
                    float keep_72 = ((lane_in_warp % 2 == 1) ? sums_2[27] : sums_2[11]);
                    float send_73 = ((lane_in_warp % 2 == 1) ? sums_2[11] : sums_2[27]);
                    float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, send_73, 1);
                    float _add_233;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_233) : "f"(keep_72), "f"(_shfl_xor_sync_234));
                    fold_level1[11] = _add_233;
                    float keep_74 = ((lane_in_warp % 2 == 1) ? sums_2[28] : sums_2[12]);
                    float send_75 = ((lane_in_warp % 2 == 1) ? sums_2[12] : sums_2[28]);
                    float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, send_75, 1);
                    float _add_235;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(keep_74), "f"(_shfl_xor_sync_236));
                    fold_level1[12] = _add_235;
                    float keep_76 = ((lane_in_warp % 2 == 1) ? sums_2[29] : sums_2[13]);
                    float send_77 = ((lane_in_warp % 2 == 1) ? sums_2[13] : sums_2[29]);
                    float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, send_77, 1);
                    float _add_237;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(keep_76), "f"(_shfl_xor_sync_238));
                    fold_level1[13] = _add_237;
                    float keep_78 = ((lane_in_warp % 2 == 1) ? sums_2[30] : sums_2[14]);
                    float send_79 = ((lane_in_warp % 2 == 1) ? sums_2[14] : sums_2[30]);
                    float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, send_79, 1);
                    float _add_239;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(keep_78), "f"(_shfl_xor_sync_240));
                    fold_level1[14] = _add_239;
                    float keep_80 = ((lane_in_warp % 2 == 1) ? sums_2[31] : sums_2[15]);
                    float send_81 = ((lane_in_warp % 2 == 1) ? sums_2[15] : sums_2[31]);
                    float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, send_81, 1);
                    float _add_241;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(keep_80), "f"(_shfl_xor_sync_242));
                    fold_level1[15] = _add_241;
                    float fold_level2[8];
                    float keep_82 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[8] : fold_level1[0]);
                    float send_83 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[0] : fold_level1[8]);
                    float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, send_83, 2);
                    float _add_243;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(keep_82), "f"(_shfl_xor_sync_244));
                    fold_level2[0] = _add_243;
                    float keep_84 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[9] : fold_level1[1]);
                    float send_85 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[1] : fold_level1[9]);
                    float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, send_85, 2);
                    float _add_245;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(keep_84), "f"(_shfl_xor_sync_246));
                    fold_level2[1] = _add_245;
                    float keep_86_1 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[10] : fold_level1[2]);
                    float send_87_1 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[2] : fold_level1[10]);
                    float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, send_87_1, 2);
                    float _add_247;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(keep_86_1), "f"(_shfl_xor_sync_248));
                    fold_level2[2] = _add_247;
                    float keep_88 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[11] : fold_level1[3]);
                    float send_89 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[3] : fold_level1[11]);
                    float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, send_89, 2);
                    float _add_249;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(keep_88), "f"(_shfl_xor_sync_250));
                    fold_level2[3] = _add_249;
                    float keep_90 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[12] : fold_level1[4]);
                    float send_91 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[4] : fold_level1[12]);
                    float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, send_91, 2);
                    float _add_251;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(keep_90), "f"(_shfl_xor_sync_252));
                    fold_level2[4] = _add_251;
                    float keep_92 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[13] : fold_level1[5]);
                    float send_93 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[5] : fold_level1[13]);
                    float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, send_93, 2);
                    float _add_253;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(keep_92), "f"(_shfl_xor_sync_254));
                    fold_level2[5] = _add_253;
                    float keep_94 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[14] : fold_level1[6]);
                    float send_95 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[6] : fold_level1[14]);
                    float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, send_95, 2);
                    float _add_255;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(keep_94), "f"(_shfl_xor_sync_256));
                    fold_level2[6] = _add_255;
                    float keep_96 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[15] : fold_level1[7]);
                    float send_97 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[7] : fold_level1[15]);
                    float _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, send_97, 2);
                    float _add_257;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(keep_96), "f"(_shfl_xor_sync_258));
                    fold_level2[7] = _add_257;
                    float fold_level3[4];
                    float keep_98 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[4] : fold_level2[0]);
                    float send_99 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[0] : fold_level2[4]);
                    float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, send_99, 4);
                    float _add_259;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(keep_98), "f"(_shfl_xor_sync_260));
                    fold_level3[0] = _add_259;
                    float keep_100 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[5] : fold_level2[1]);
                    float send_101 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[1] : fold_level2[5]);
                    float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, send_101, 4);
                    float _add_261;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(keep_100), "f"(_shfl_xor_sync_262));
                    fold_level3[1] = _add_261;
                    float keep_102 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[6] : fold_level2[2]);
                    float send_103 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[2] : fold_level2[6]);
                    float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, send_103, 4);
                    float _add_263;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(keep_102), "f"(_shfl_xor_sync_264));
                    fold_level3[2] = _add_263;
                    float keep_104_1 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[7] : fold_level2[3]);
                    float send_105_1 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[3] : fold_level2[7]);
                    float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, send_105_1, 4);
                    float _add_265;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(keep_104_1), "f"(_shfl_xor_sync_266));
                    fold_level3[3] = _add_265;
                    float fold_level4[2];
                    float keep_106 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[2] : fold_level3[0]);
                    float send_107 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[0] : fold_level3[2]);
                    float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, send_107, 8);
                    float _add_267;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(keep_106), "f"(_shfl_xor_sync_268));
                    fold_level4[0] = _add_267;
                    float keep_108 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[3] : fold_level3[1]);
                    float send_109 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[1] : fold_level3[3]);
                    float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, send_109, 8);
                    float _add_269;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(keep_108), "f"(_shfl_xor_sync_270));
                    fold_level4[1] = _add_269;
                    float fold_level5[1];
                    float keep_110 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[1] : fold_level4[0]);
                    float send_111 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[0] : fold_level4[1]);
                    float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, send_111, 16);
                    float _add_271;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(keep_110), "f"(_shfl_xor_sync_272));
                    fold_level5[0] = _add_271;
                    float total_2 = fold_level5[0];
                    float s_112 = ((lane_in_warp < 24) ? total_2 : 0.0f);
                    float q_113 = __shfl_sync(0xFFFFFFFF, total_2, 24);
                    s = s_112;
                    q = q_113;
                } else {
                    float s_0 = 0.0f;
                    float q_1 = 0.0f;
                    if (num_splits > 22) {
                        unsigned int row_14 = token % 64;
                        unsigned int first_task_3 = mb_1 * num_splits;
                        float sums_3[25];
                        float errs_1[25];
                        unsigned int split_3 = lane_in_warp * 2;
                        unsigned int safe_split_3 = ((split_3 < num_splits) ? split_3 : 0);
                        unsigned long long record_3 = ((unsigned long long)first_task_3 + (unsigned long long)safe_split_3) * 1536 + (unsigned long long)(row_14 * 24);
                        float _vec_load_24[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_3 + 0);
                            _vec_load_24[0 + 0] = _v4.x;
                            _vec_load_24[0 + 1] = _v4.y;
                            _vec_load_24[0 + 2] = _v4.z;
                            _vec_load_24[0 + 3] = _v4.w;
                        }
                        float value_50 = ((split_3 < num_splits) ? _vec_load_24[0] : 0.0f);
                        sums_3[0] = value_50;
                        float value_0_3 = ((split_3 < num_splits) ? _vec_load_24[1] : 0.0f);
                        sums_3[1] = value_0_3;
                        float value_1_4 = ((split_3 < num_splits) ? _vec_load_24[2] : 0.0f);
                        sums_3[2] = value_1_4;
                        float value_2_3 = ((split_3 < num_splits) ? _vec_load_24[3] : 0.0f);
                        sums_3[3] = value_2_3;
                        float _vec_load_25[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 4) + 0);
                            _vec_load_25[0 + 0] = _v4.x;
                            _vec_load_25[0 + 1] = _v4.y;
                            _vec_load_25[0 + 2] = _v4.z;
                            _vec_load_25[0 + 3] = _v4.w;
                        }
                        float value_3_3 = ((split_3 < num_splits) ? _vec_load_25[0] : 0.0f);
                        sums_3[4] = value_3_3;
                        float value_4_3 = ((split_3 < num_splits) ? _vec_load_25[1] : 0.0f);
                        sums_3[5] = value_4_3;
                        float value_5_3 = ((split_3 < num_splits) ? _vec_load_25[2] : 0.0f);
                        sums_3[6] = value_5_3;
                        float value_6_3 = ((split_3 < num_splits) ? _vec_load_25[3] : 0.0f);
                        sums_3[7] = value_6_3;
                        float _vec_load_26[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 8) + 0);
                            _vec_load_26[0 + 0] = _v4.x;
                            _vec_load_26[0 + 1] = _v4.y;
                            _vec_load_26[0 + 2] = _v4.z;
                            _vec_load_26[0 + 3] = _v4.w;
                        }
                        float value_7_3 = ((split_3 < num_splits) ? _vec_load_26[0] : 0.0f);
                        sums_3[8] = value_7_3;
                        float value_8_4 = ((split_3 < num_splits) ? _vec_load_26[1] : 0.0f);
                        sums_3[9] = value_8_4;
                        float value_9_3 = ((split_3 < num_splits) ? _vec_load_26[2] : 0.0f);
                        sums_3[10] = value_9_3;
                        float value_10_3 = ((split_3 < num_splits) ? _vec_load_26[3] : 0.0f);
                        sums_3[11] = value_10_3;
                        float _vec_load_27[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 12) + 0);
                            _vec_load_27[0 + 0] = _v4.x;
                            _vec_load_27[0 + 1] = _v4.y;
                            _vec_load_27[0 + 2] = _v4.z;
                            _vec_load_27[0 + 3] = _v4.w;
                        }
                        float value_11_2 = ((split_3 < num_splits) ? _vec_load_27[0] : 0.0f);
                        sums_3[12] = value_11_2;
                        float value_12_2 = ((split_3 < num_splits) ? _vec_load_27[1] : 0.0f);
                        sums_3[13] = value_12_2;
                        float value_13_1 = ((split_3 < num_splits) ? _vec_load_27[2] : 0.0f);
                        sums_3[14] = value_13_1;
                        float value_14_3 = ((split_3 < num_splits) ? _vec_load_27[3] : 0.0f);
                        sums_3[15] = value_14_3;
                        float _vec_load_28[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 16) + 0);
                            _vec_load_28[0 + 0] = _v4.x;
                            _vec_load_28[0 + 1] = _v4.y;
                            _vec_load_28[0 + 2] = _v4.z;
                            _vec_load_28[0 + 3] = _v4.w;
                        }
                        float value_15_2 = ((split_3 < num_splits) ? _vec_load_28[0] : 0.0f);
                        sums_3[16] = value_15_2;
                        float value_16_3 = ((split_3 < num_splits) ? _vec_load_28[1] : 0.0f);
                        sums_3[17] = value_16_3;
                        float value_17_3 = ((split_3 < num_splits) ? _vec_load_28[2] : 0.0f);
                        sums_3[18] = value_17_3;
                        float value_18_2 = ((split_3 < num_splits) ? _vec_load_28[3] : 0.0f);
                        sums_3[19] = value_18_2;
                        float _vec_load_29[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 20) + 0);
                            _vec_load_29[0 + 0] = _v4.x;
                            _vec_load_29[0 + 1] = _v4.y;
                            _vec_load_29[0 + 2] = _v4.z;
                            _vec_load_29[0 + 3] = _v4.w;
                        }
                        float value_19_2 = ((split_3 < num_splits) ? _vec_load_29[0] : 0.0f);
                        sums_3[20] = value_19_2;
                        float value_20_3 = ((split_3 < num_splits) ? _vec_load_29[1] : 0.0f);
                        sums_3[21] = value_20_3;
                        float value_21_2 = ((split_3 < num_splits) ? _vec_load_29[2] : 0.0f);
                        sums_3[22] = value_21_2;
                        float value_22_2 = ((split_3 < num_splits) ? _vec_load_29[3] : 0.0f);
                        sums_3[23] = value_22_2;
                        float q_word_3 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_3 * 64 + (unsigned long long)(safe_split_3 * 64) + (unsigned long long)row_14];
                        float q_value_3 = ((split_3 < num_splits) ? q_word_3 : 0.0f);
                        sums_3[24] = q_value_3;
                        unsigned int split_23_1 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_24_1 = ((split_23_1 < num_splits) ? split_23_1 : 0);
                        unsigned long long record_25_1 = ((unsigned long long)first_task_3 + (unsigned long long)safe_split_24_1) * 1536 + (unsigned long long)(row_14 * 24);
                        float _vec_load_30[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25_1 + 0);
                            _vec_load_30[0 + 0] = _v4.x;
                            _vec_load_30[0 + 1] = _v4.y;
                            _vec_load_30[0 + 2] = _v4.z;
                            _vec_load_30[0 + 3] = _v4.w;
                        }
                        float value_26_2 = ((split_23_1 < num_splits) ? _vec_load_30[0] : 0.0f);
                        float s_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(sums_3[0]), "f"(value_26_2));
                        float bb_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_1) : "f"(s_27), "f"(sums_3[0]));
                        float _sub_281;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_281) : "f"(s_27), "f"(bb_1));
                        float _sub_280;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_280) : "f"(sums_3[0]), "f"(_sub_281));
                        float _sub_282;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(value_26_2), "f"(bb_1));
                        float e_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_280), "f"(_sub_282));
                        sums_3[0] = s_27;
                        errs_1[0] = e_1;
                        float value_28_2 = ((split_23_1 < num_splits) ? _vec_load_30[1] : 0.0f);
                        float s_29_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29_1) : "f"(sums_3[1]), "f"(value_28_2));
                        float bb_30_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_30_1) : "f"(s_29_1), "f"(sums_3[1]));
                        float _sub_284;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(s_29_1), "f"(bb_30_1));
                        float _sub_283;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_283) : "f"(sums_3[1]), "f"(_sub_284));
                        float _sub_285;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_285) : "f"(value_28_2), "f"(bb_30_1));
                        float e_31_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_31_1) : "f"(_sub_283), "f"(_sub_285));
                        sums_3[1] = s_29_1;
                        errs_1[1] = e_31_1;
                        float value_32_2 = ((split_23_1 < num_splits) ? _vec_load_30[2] : 0.0f);
                        float s_33_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33_1) : "f"(sums_3[2]), "f"(value_32_2));
                        float bb_34_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_34_1) : "f"(s_33_1), "f"(sums_3[2]));
                        float _sub_287;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_287) : "f"(s_33_1), "f"(bb_34_1));
                        float _sub_286;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_286) : "f"(sums_3[2]), "f"(_sub_287));
                        float _sub_288;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_288) : "f"(value_32_2), "f"(bb_34_1));
                        float e_35_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_35_1) : "f"(_sub_286), "f"(_sub_288));
                        sums_3[2] = s_33_1;
                        errs_1[2] = e_35_1;
                        float value_36_2 = ((split_23_1 < num_splits) ? _vec_load_30[3] : 0.0f);
                        float s_37_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37_1) : "f"(sums_3[3]), "f"(value_36_2));
                        float bb_38_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_38_1) : "f"(s_37_1), "f"(sums_3[3]));
                        float _sub_290;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_290) : "f"(s_37_1), "f"(bb_38_1));
                        float _sub_289;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(sums_3[3]), "f"(_sub_290));
                        float _sub_291;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(value_36_2), "f"(bb_38_1));
                        float e_39_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_39_1) : "f"(_sub_289), "f"(_sub_291));
                        sums_3[3] = s_37_1;
                        errs_1[3] = e_39_1;
                        float _vec_load_31[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 4) + 0);
                            _vec_load_31[0 + 0] = _v4.x;
                            _vec_load_31[0 + 1] = _v4.y;
                            _vec_load_31[0 + 2] = _v4.z;
                            _vec_load_31[0 + 3] = _v4.w;
                        }
                        float value_40_2 = ((split_23_1 < num_splits) ? _vec_load_31[0] : 0.0f);
                        float s_41_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41_1) : "f"(sums_3[4]), "f"(value_40_2));
                        float bb_42_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_42_1) : "f"(s_41_1), "f"(sums_3[4]));
                        float _sub_294;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_294) : "f"(s_41_1), "f"(bb_42_1));
                        float _sub_293;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(sums_3[4]), "f"(_sub_294));
                        float _sub_295;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(value_40_2), "f"(bb_42_1));
                        float e_43_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_43_1) : "f"(_sub_293), "f"(_sub_295));
                        sums_3[4] = s_41_1;
                        errs_1[4] = e_43_1;
                        float value_44_3 = ((split_23_1 < num_splits) ? _vec_load_31[1] : 0.0f);
                        float s_45_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45_1) : "f"(sums_3[5]), "f"(value_44_3));
                        float bb_46_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_46_1) : "f"(s_45_1), "f"(sums_3[5]));
                        float _sub_297;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_297) : "f"(s_45_1), "f"(bb_46_1));
                        float _sub_296;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_296) : "f"(sums_3[5]), "f"(_sub_297));
                        float _sub_298;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(value_44_3), "f"(bb_46_1));
                        float e_47_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_47_1) : "f"(_sub_296), "f"(_sub_298));
                        sums_3[5] = s_45_1;
                        errs_1[5] = e_47_1;
                        float value_48_2 = ((split_23_1 < num_splits) ? _vec_load_31[2] : 0.0f);
                        float s_49_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49_1) : "f"(sums_3[6]), "f"(value_48_2));
                        float bb_50_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_50_1) : "f"(s_49_1), "f"(sums_3[6]));
                        float _sub_300;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_300) : "f"(s_49_1), "f"(bb_50_1));
                        float _sub_299;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_299) : "f"(sums_3[6]), "f"(_sub_300));
                        float _sub_301;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(value_48_2), "f"(bb_50_1));
                        float e_51_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_51_1) : "f"(_sub_299), "f"(_sub_301));
                        sums_3[6] = s_49_1;
                        errs_1[6] = e_51_1;
                        float value_52_1 = ((split_23_1 < num_splits) ? _vec_load_31[3] : 0.0f);
                        float s_53_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53_1) : "f"(sums_3[7]), "f"(value_52_1));
                        float bb_54_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_54_1) : "f"(s_53_1), "f"(sums_3[7]));
                        float _sub_303;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_303) : "f"(s_53_1), "f"(bb_54_1));
                        float _sub_302;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_302) : "f"(sums_3[7]), "f"(_sub_303));
                        float _sub_304;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_304) : "f"(value_52_1), "f"(bb_54_1));
                        float e_55_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_55_1) : "f"(_sub_302), "f"(_sub_304));
                        sums_3[7] = s_53_1;
                        errs_1[7] = e_55_1;
                        float _vec_load_32[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 8) + 0);
                            _vec_load_32[0 + 0] = _v4.x;
                            _vec_load_32[0 + 1] = _v4.y;
                            _vec_load_32[0 + 2] = _v4.z;
                            _vec_load_32[0 + 3] = _v4.w;
                        }
                        float value_56_1 = ((split_23_1 < num_splits) ? _vec_load_32[0] : 0.0f);
                        float s_57_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57_1) : "f"(sums_3[8]), "f"(value_56_1));
                        float bb_58_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_58_1) : "f"(s_57_1), "f"(sums_3[8]));
                        float _sub_307;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_307) : "f"(s_57_1), "f"(bb_58_1));
                        float _sub_306;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_306) : "f"(sums_3[8]), "f"(_sub_307));
                        float _sub_308;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_308) : "f"(value_56_1), "f"(bb_58_1));
                        float e_59_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_59_1) : "f"(_sub_306), "f"(_sub_308));
                        sums_3[8] = s_57_1;
                        errs_1[8] = e_59_1;
                        float value_60 = ((split_23_1 < num_splits) ? _vec_load_32[1] : 0.0f);
                        float s_61;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(sums_3[9]), "f"(value_60));
                        float bb_62;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_62) : "f"(s_61), "f"(sums_3[9]));
                        float _sub_310;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_310) : "f"(s_61), "f"(bb_62));
                        float _sub_309;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_309) : "f"(sums_3[9]), "f"(_sub_310));
                        float _sub_311;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_311) : "f"(value_60), "f"(bb_62));
                        float e_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_63) : "f"(_sub_309), "f"(_sub_311));
                        sums_3[9] = s_61;
                        errs_1[9] = e_63;
                        float value_64 = ((split_23_1 < num_splits) ? _vec_load_32[2] : 0.0f);
                        float s_65_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65_1) : "f"(sums_3[10]), "f"(value_64));
                        float bb_66_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_66_1) : "f"(s_65_1), "f"(sums_3[10]));
                        float _sub_313;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_313) : "f"(s_65_1), "f"(bb_66_1));
                        float _sub_312;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_312) : "f"(sums_3[10]), "f"(_sub_313));
                        float _sub_314;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_314) : "f"(value_64), "f"(bb_66_1));
                        float e_67_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_67_1) : "f"(_sub_312), "f"(_sub_314));
                        sums_3[10] = s_65_1;
                        errs_1[10] = e_67_1;
                        float value_68 = ((split_23_1 < num_splits) ? _vec_load_32[3] : 0.0f);
                        float s_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(sums_3[11]), "f"(value_68));
                        float bb_70;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_70) : "f"(s_69), "f"(sums_3[11]));
                        float _sub_316;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_316) : "f"(s_69), "f"(bb_70));
                        float _sub_315;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_315) : "f"(sums_3[11]), "f"(_sub_316));
                        float _sub_317;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_317) : "f"(value_68), "f"(bb_70));
                        float e_71;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_71) : "f"(_sub_315), "f"(_sub_317));
                        sums_3[11] = s_69;
                        errs_1[11] = e_71;
                        float _vec_load_33[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 12) + 0);
                            _vec_load_33[0 + 0] = _v4.x;
                            _vec_load_33[0 + 1] = _v4.y;
                            _vec_load_33[0 + 2] = _v4.z;
                            _vec_load_33[0 + 3] = _v4.w;
                        }
                        float value_72 = ((split_23_1 < num_splits) ? _vec_load_33[0] : 0.0f);
                        float s_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_73) : "f"(sums_3[12]), "f"(value_72));
                        float bb_74;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_74) : "f"(s_73), "f"(sums_3[12]));
                        float _sub_320;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_320) : "f"(s_73), "f"(bb_74));
                        float _sub_319;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(sums_3[12]), "f"(_sub_320));
                        float _sub_321;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_321) : "f"(value_72), "f"(bb_74));
                        float e_75;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_75) : "f"(_sub_319), "f"(_sub_321));
                        sums_3[12] = s_73;
                        errs_1[12] = e_75;
                        float value_76 = ((split_23_1 < num_splits) ? _vec_load_33[1] : 0.0f);
                        float s_77;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_77) : "f"(sums_3[13]), "f"(value_76));
                        float bb_78;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_78) : "f"(s_77), "f"(sums_3[13]));
                        float _sub_323;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(s_77), "f"(bb_78));
                        float _sub_322;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_322) : "f"(sums_3[13]), "f"(_sub_323));
                        float _sub_324;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_324) : "f"(value_76), "f"(bb_78));
                        float e_79;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_79) : "f"(_sub_322), "f"(_sub_324));
                        sums_3[13] = s_77;
                        errs_1[13] = e_79;
                        float value_80 = ((split_23_1 < num_splits) ? _vec_load_33[2] : 0.0f);
                        float s_81;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_81) : "f"(sums_3[14]), "f"(value_80));
                        float bb_82;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_82) : "f"(s_81), "f"(sums_3[14]));
                        float _sub_326;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_326) : "f"(s_81), "f"(bb_82));
                        float _sub_325;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_325) : "f"(sums_3[14]), "f"(_sub_326));
                        float _sub_327;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_327) : "f"(value_80), "f"(bb_82));
                        float e_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_83) : "f"(_sub_325), "f"(_sub_327));
                        sums_3[14] = s_81;
                        errs_1[14] = e_83;
                        float value_84 = ((split_23_1 < num_splits) ? _vec_load_33[3] : 0.0f);
                        float s_85;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_85) : "f"(sums_3[15]), "f"(value_84));
                        float bb_86;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_86) : "f"(s_85), "f"(sums_3[15]));
                        float _sub_329;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(s_85), "f"(bb_86));
                        float _sub_328;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_328) : "f"(sums_3[15]), "f"(_sub_329));
                        float _sub_330;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_330) : "f"(value_84), "f"(bb_86));
                        float e_87;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_87) : "f"(_sub_328), "f"(_sub_330));
                        sums_3[15] = s_85;
                        errs_1[15] = e_87;
                        float _vec_load_34[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 16) + 0);
                            _vec_load_34[0 + 0] = _v4.x;
                            _vec_load_34[0 + 1] = _v4.y;
                            _vec_load_34[0 + 2] = _v4.z;
                            _vec_load_34[0 + 3] = _v4.w;
                        }
                        float value_88 = ((split_23_1 < num_splits) ? _vec_load_34[0] : 0.0f);
                        float s_89;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_89) : "f"(sums_3[16]), "f"(value_88));
                        float bb_90;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_90) : "f"(s_89), "f"(sums_3[16]));
                        float _sub_333;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_333) : "f"(s_89), "f"(bb_90));
                        float _sub_332;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_332) : "f"(sums_3[16]), "f"(_sub_333));
                        float _sub_334;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(value_88), "f"(bb_90));
                        float e_91;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_91) : "f"(_sub_332), "f"(_sub_334));
                        sums_3[16] = s_89;
                        errs_1[16] = e_91;
                        float value_92 = ((split_23_1 < num_splits) ? _vec_load_34[1] : 0.0f);
                        float s_93;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_93) : "f"(sums_3[17]), "f"(value_92));
                        float bb_94;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_94) : "f"(s_93), "f"(sums_3[17]));
                        float _sub_336;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(s_93), "f"(bb_94));
                        float _sub_335;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_335) : "f"(sums_3[17]), "f"(_sub_336));
                        float _sub_337;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_337) : "f"(value_92), "f"(bb_94));
                        float e_95;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_95) : "f"(_sub_335), "f"(_sub_337));
                        sums_3[17] = s_93;
                        errs_1[17] = e_95;
                        float value_96 = ((split_23_1 < num_splits) ? _vec_load_34[2] : 0.0f);
                        float s_97;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_97) : "f"(sums_3[18]), "f"(value_96));
                        float bb_98;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_98) : "f"(s_97), "f"(sums_3[18]));
                        float _sub_339;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_339) : "f"(s_97), "f"(bb_98));
                        float _sub_338;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_338) : "f"(sums_3[18]), "f"(_sub_339));
                        float _sub_340;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_340) : "f"(value_96), "f"(bb_98));
                        float e_99;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_99) : "f"(_sub_338), "f"(_sub_340));
                        sums_3[18] = s_97;
                        errs_1[18] = e_99;
                        float value_100 = ((split_23_1 < num_splits) ? _vec_load_34[3] : 0.0f);
                        float s_101_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_101_1) : "f"(sums_3[19]), "f"(value_100));
                        float bb_102_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_102_1) : "f"(s_101_1), "f"(sums_3[19]));
                        float _sub_342;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_342) : "f"(s_101_1), "f"(bb_102_1));
                        float _sub_341;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_341) : "f"(sums_3[19]), "f"(_sub_342));
                        float _sub_343;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_343) : "f"(value_100), "f"(bb_102_1));
                        float e_103_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_103_1) : "f"(_sub_341), "f"(_sub_343));
                        sums_3[19] = s_101_1;
                        errs_1[19] = e_103_1;
                        float _vec_load_35[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 20) + 0);
                            _vec_load_35[0 + 0] = _v4.x;
                            _vec_load_35[0 + 1] = _v4.y;
                            _vec_load_35[0 + 2] = _v4.z;
                            _vec_load_35[0 + 3] = _v4.w;
                        }
                        float value_104 = ((split_23_1 < num_splits) ? _vec_load_35[0] : 0.0f);
                        float s_105;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_105) : "f"(sums_3[20]), "f"(value_104));
                        float bb_106;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_106) : "f"(s_105), "f"(sums_3[20]));
                        float _sub_346;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_346) : "f"(s_105), "f"(bb_106));
                        float _sub_345;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_345) : "f"(sums_3[20]), "f"(_sub_346));
                        float _sub_347;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_347) : "f"(value_104), "f"(bb_106));
                        float e_107;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_107) : "f"(_sub_345), "f"(_sub_347));
                        sums_3[20] = s_105;
                        errs_1[20] = e_107;
                        float value_108 = ((split_23_1 < num_splits) ? _vec_load_35[1] : 0.0f);
                        float s_109;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_109) : "f"(sums_3[21]), "f"(value_108));
                        float bb_110;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_110) : "f"(s_109), "f"(sums_3[21]));
                        float _sub_349;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_349) : "f"(s_109), "f"(bb_110));
                        float _sub_348;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_348) : "f"(sums_3[21]), "f"(_sub_349));
                        float _sub_350;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_350) : "f"(value_108), "f"(bb_110));
                        float e_111;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_111) : "f"(_sub_348), "f"(_sub_350));
                        sums_3[21] = s_109;
                        errs_1[21] = e_111;
                        float value_112 = ((split_23_1 < num_splits) ? _vec_load_35[2] : 0.0f);
                        float s_113;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_113) : "f"(sums_3[22]), "f"(value_112));
                        float bb_114;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_114) : "f"(s_113), "f"(sums_3[22]));
                        float _sub_352;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_352) : "f"(s_113), "f"(bb_114));
                        float _sub_351;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_351) : "f"(sums_3[22]), "f"(_sub_352));
                        float _sub_353;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_353) : "f"(value_112), "f"(bb_114));
                        float e_115;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_115) : "f"(_sub_351), "f"(_sub_353));
                        sums_3[22] = s_113;
                        errs_1[22] = e_115;
                        float value_116_1 = ((split_23_1 < num_splits) ? _vec_load_35[3] : 0.0f);
                        float s_117;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_117) : "f"(sums_3[23]), "f"(value_116_1));
                        float bb_118;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_118) : "f"(s_117), "f"(sums_3[23]));
                        float _sub_355;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_355) : "f"(s_117), "f"(bb_118));
                        float _sub_354;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_354) : "f"(sums_3[23]), "f"(_sub_355));
                        float _sub_356;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_356) : "f"(value_116_1), "f"(bb_118));
                        float e_119;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_119) : "f"(_sub_354), "f"(_sub_356));
                        sums_3[23] = s_117;
                        errs_1[23] = e_119;
                        float q_word_120 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_3 * 64 + (unsigned long long)(safe_split_24_1 * 64) + (unsigned long long)row_14];
                        float q_value_121 = ((split_23_1 < num_splits) ? q_word_120 : 0.0f);
                        float s_122;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_122) : "f"(sums_3[24]), "f"(q_value_121));
                        float bb_123;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_123) : "f"(s_122), "f"(sums_3[24]));
                        float _sub_358;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_358) : "f"(s_122), "f"(bb_123));
                        float _sub_357;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_357) : "f"(sums_3[24]), "f"(_sub_358));
                        float _sub_359;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_359) : "f"(q_value_121), "f"(bb_123));
                        float e_124;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_124) : "f"(_sub_357), "f"(_sub_359));
                        sums_3[24] = s_122;
                        errs_1[24] = e_124;
                        float partner_1 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 1);
                        float partner_err = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 1);
                        float s_125;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_125) : "f"(sums_3[0]), "f"(partner_1));
                        float bb_126;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_126) : "f"(s_125), "f"(sums_3[0]));
                        float _sub_361;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_361) : "f"(s_125), "f"(bb_126));
                        float _sub_360;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_360) : "f"(sums_3[0]), "f"(_sub_361));
                        float _sub_362;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_362) : "f"(partner_1), "f"(bb_126));
                        float e_127;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_127) : "f"(_sub_360), "f"(_sub_362));
                        sums_3[0] = s_125;
                        float _add_364;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(errs_1[0]), "f"(partner_err));
                        float _add_363;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_363) : "f"(_add_364), "f"(e_127));
                        errs_1[0] = _add_363;
                        float partner_128 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 1);
                        float partner_err_129 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 1);
                        float s_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_130) : "f"(sums_3[1]), "f"(partner_128));
                        float bb_131;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_131) : "f"(s_130), "f"(sums_3[1]));
                        float _sub_366;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_366) : "f"(s_130), "f"(bb_131));
                        float _sub_365;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(sums_3[1]), "f"(_sub_366));
                        float _sub_367;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_367) : "f"(partner_128), "f"(bb_131));
                        float e_132;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_132) : "f"(_sub_365), "f"(_sub_367));
                        sums_3[1] = s_130;
                        float _add_369;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_369) : "f"(errs_1[1]), "f"(partner_err_129));
                        float _add_368;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_368) : "f"(_add_369), "f"(e_132));
                        errs_1[1] = _add_368;
                        float partner_133 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 1);
                        float partner_err_134 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 1);
                        float s_135;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_135) : "f"(sums_3[2]), "f"(partner_133));
                        float bb_136;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_136) : "f"(s_135), "f"(sums_3[2]));
                        float _sub_371;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_371) : "f"(s_135), "f"(bb_136));
                        float _sub_370;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_370) : "f"(sums_3[2]), "f"(_sub_371));
                        float _sub_372;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_372) : "f"(partner_133), "f"(bb_136));
                        float e_137;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_137) : "f"(_sub_370), "f"(_sub_372));
                        sums_3[2] = s_135;
                        float _add_374;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(errs_1[2]), "f"(partner_err_134));
                        float _add_373;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_373) : "f"(_add_374), "f"(e_137));
                        errs_1[2] = _add_373;
                        float partner_138 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 1);
                        float partner_err_139 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 1);
                        float s_140;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_140) : "f"(sums_3[3]), "f"(partner_138));
                        float bb_141;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_141) : "f"(s_140), "f"(sums_3[3]));
                        float _sub_376;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_376) : "f"(s_140), "f"(bb_141));
                        float _sub_375;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_375) : "f"(sums_3[3]), "f"(_sub_376));
                        float _sub_377;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_377) : "f"(partner_138), "f"(bb_141));
                        float e_142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_142) : "f"(_sub_375), "f"(_sub_377));
                        sums_3[3] = s_140;
                        float _add_379;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_379) : "f"(errs_1[3]), "f"(partner_err_139));
                        float _add_378;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(_add_379), "f"(e_142));
                        errs_1[3] = _add_378;
                        float partner_143 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 1);
                        float partner_err_144 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 1);
                        float s_145;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_145) : "f"(sums_3[4]), "f"(partner_143));
                        float bb_146;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_146) : "f"(s_145), "f"(sums_3[4]));
                        float _sub_381;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_381) : "f"(s_145), "f"(bb_146));
                        float _sub_380;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_380) : "f"(sums_3[4]), "f"(_sub_381));
                        float _sub_382;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_382) : "f"(partner_143), "f"(bb_146));
                        float e_147;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_147) : "f"(_sub_380), "f"(_sub_382));
                        sums_3[4] = s_145;
                        float _add_384;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_384) : "f"(errs_1[4]), "f"(partner_err_144));
                        float _add_383;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_383) : "f"(_add_384), "f"(e_147));
                        errs_1[4] = _add_383;
                        float partner_148 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 1);
                        float partner_err_149 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 1);
                        float s_150;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_150) : "f"(sums_3[5]), "f"(partner_148));
                        float bb_151;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_151) : "f"(s_150), "f"(sums_3[5]));
                        float _sub_386;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_386) : "f"(s_150), "f"(bb_151));
                        float _sub_385;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_385) : "f"(sums_3[5]), "f"(_sub_386));
                        float _sub_387;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_387) : "f"(partner_148), "f"(bb_151));
                        float e_152;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_152) : "f"(_sub_385), "f"(_sub_387));
                        sums_3[5] = s_150;
                        float _add_389;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_389) : "f"(errs_1[5]), "f"(partner_err_149));
                        float _add_388;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_388) : "f"(_add_389), "f"(e_152));
                        errs_1[5] = _add_388;
                        float partner_153 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 1);
                        float partner_err_154 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 1);
                        float s_155_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_155_1) : "f"(sums_3[6]), "f"(partner_153));
                        float bb_156_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_156_1) : "f"(s_155_1), "f"(sums_3[6]));
                        float _sub_391;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_391) : "f"(s_155_1), "f"(bb_156_1));
                        float _sub_390;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_390) : "f"(sums_3[6]), "f"(_sub_391));
                        float _sub_392;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_392) : "f"(partner_153), "f"(bb_156_1));
                        float e_157_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_157_1) : "f"(_sub_390), "f"(_sub_392));
                        sums_3[6] = s_155_1;
                        float _add_394;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_394) : "f"(errs_1[6]), "f"(partner_err_154));
                        float _add_393;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_393) : "f"(_add_394), "f"(e_157_1));
                        errs_1[6] = _add_393;
                        float partner_158 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 1);
                        float partner_err_159 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 1);
                        float s_160;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_160) : "f"(sums_3[7]), "f"(partner_158));
                        float bb_161;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_161) : "f"(s_160), "f"(sums_3[7]));
                        float _sub_396;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_396) : "f"(s_160), "f"(bb_161));
                        float _sub_395;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_395) : "f"(sums_3[7]), "f"(_sub_396));
                        float _sub_397;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_397) : "f"(partner_158), "f"(bb_161));
                        float e_162;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_162) : "f"(_sub_395), "f"(_sub_397));
                        sums_3[7] = s_160;
                        float _add_399;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_399) : "f"(errs_1[7]), "f"(partner_err_159));
                        float _add_398;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_398) : "f"(_add_399), "f"(e_162));
                        errs_1[7] = _add_398;
                        float partner_163 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 1);
                        float partner_err_164 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 1);
                        float s_165;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_165) : "f"(sums_3[8]), "f"(partner_163));
                        float bb_166;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_166) : "f"(s_165), "f"(sums_3[8]));
                        float _sub_401;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_401) : "f"(s_165), "f"(bb_166));
                        float _sub_400;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_400) : "f"(sums_3[8]), "f"(_sub_401));
                        float _sub_402;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_402) : "f"(partner_163), "f"(bb_166));
                        float e_167;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_167) : "f"(_sub_400), "f"(_sub_402));
                        sums_3[8] = s_165;
                        float _add_404;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_404) : "f"(errs_1[8]), "f"(partner_err_164));
                        float _add_403;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_403) : "f"(_add_404), "f"(e_167));
                        errs_1[8] = _add_403;
                        float partner_168 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 1);
                        float partner_err_169 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 1);
                        float s_170;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_170) : "f"(sums_3[9]), "f"(partner_168));
                        float bb_171;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_171) : "f"(s_170), "f"(sums_3[9]));
                        float _sub_406;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_406) : "f"(s_170), "f"(bb_171));
                        float _sub_405;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_405) : "f"(sums_3[9]), "f"(_sub_406));
                        float _sub_407;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_407) : "f"(partner_168), "f"(bb_171));
                        float e_172;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_172) : "f"(_sub_405), "f"(_sub_407));
                        sums_3[9] = s_170;
                        float _add_409;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_409) : "f"(errs_1[9]), "f"(partner_err_169));
                        float _add_408;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(_add_409), "f"(e_172));
                        errs_1[9] = _add_408;
                        float partner_173 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 1);
                        float partner_err_174 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 1);
                        float s_175;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_175) : "f"(sums_3[10]), "f"(partner_173));
                        float bb_176;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_176) : "f"(s_175), "f"(sums_3[10]));
                        float _sub_411;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_411) : "f"(s_175), "f"(bb_176));
                        float _sub_410;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_410) : "f"(sums_3[10]), "f"(_sub_411));
                        float _sub_412;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_412) : "f"(partner_173), "f"(bb_176));
                        float e_177;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_177) : "f"(_sub_410), "f"(_sub_412));
                        sums_3[10] = s_175;
                        float _add_414;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_414) : "f"(errs_1[10]), "f"(partner_err_174));
                        float _add_413;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_413) : "f"(_add_414), "f"(e_177));
                        errs_1[10] = _add_413;
                        float partner_178 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 1);
                        float partner_err_179 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 1);
                        float s_180;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_180) : "f"(sums_3[11]), "f"(partner_178));
                        float bb_181;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_181) : "f"(s_180), "f"(sums_3[11]));
                        float _sub_416;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_416) : "f"(s_180), "f"(bb_181));
                        float _sub_415;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_415) : "f"(sums_3[11]), "f"(_sub_416));
                        float _sub_417;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_417) : "f"(partner_178), "f"(bb_181));
                        float e_182;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_182) : "f"(_sub_415), "f"(_sub_417));
                        sums_3[11] = s_180;
                        float _add_419;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(errs_1[11]), "f"(partner_err_179));
                        float _add_418;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_418) : "f"(_add_419), "f"(e_182));
                        errs_1[11] = _add_418;
                        float partner_183 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 1);
                        float partner_err_184 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 1);
                        float s_185;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_185) : "f"(sums_3[12]), "f"(partner_183));
                        float bb_186;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_186) : "f"(s_185), "f"(sums_3[12]));
                        float _sub_421;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_421) : "f"(s_185), "f"(bb_186));
                        float _sub_420;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_420) : "f"(sums_3[12]), "f"(_sub_421));
                        float _sub_422;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_422) : "f"(partner_183), "f"(bb_186));
                        float e_187;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_187) : "f"(_sub_420), "f"(_sub_422));
                        sums_3[12] = s_185;
                        float _add_424;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_424) : "f"(errs_1[12]), "f"(partner_err_184));
                        float _add_423;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_423) : "f"(_add_424), "f"(e_187));
                        errs_1[12] = _add_423;
                        float partner_188 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 1);
                        float partner_err_189 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 1);
                        float s_190;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_190) : "f"(sums_3[13]), "f"(partner_188));
                        float bb_191;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_191) : "f"(s_190), "f"(sums_3[13]));
                        float _sub_426;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_426) : "f"(s_190), "f"(bb_191));
                        float _sub_425;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_425) : "f"(sums_3[13]), "f"(_sub_426));
                        float _sub_427;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_427) : "f"(partner_188), "f"(bb_191));
                        float e_192;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_192) : "f"(_sub_425), "f"(_sub_427));
                        sums_3[13] = s_190;
                        float _add_429;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_429) : "f"(errs_1[13]), "f"(partner_err_189));
                        float _add_428;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_428) : "f"(_add_429), "f"(e_192));
                        errs_1[13] = _add_428;
                        float partner_193 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 1);
                        float partner_err_194 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 1);
                        float s_195;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_195) : "f"(sums_3[14]), "f"(partner_193));
                        float bb_196;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_196) : "f"(s_195), "f"(sums_3[14]));
                        float _sub_431;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_431) : "f"(s_195), "f"(bb_196));
                        float _sub_430;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_430) : "f"(sums_3[14]), "f"(_sub_431));
                        float _sub_432;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_432) : "f"(partner_193), "f"(bb_196));
                        float e_197;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_197) : "f"(_sub_430), "f"(_sub_432));
                        sums_3[14] = s_195;
                        float _add_434;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_434) : "f"(errs_1[14]), "f"(partner_err_194));
                        float _add_433;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_433) : "f"(_add_434), "f"(e_197));
                        errs_1[14] = _add_433;
                        float partner_198 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 1);
                        float partner_err_199 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 1);
                        float s_200;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_200) : "f"(sums_3[15]), "f"(partner_198));
                        float bb_201;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_201) : "f"(s_200), "f"(sums_3[15]));
                        float _sub_436;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_436) : "f"(s_200), "f"(bb_201));
                        float _sub_435;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_435) : "f"(sums_3[15]), "f"(_sub_436));
                        float _sub_437;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_437) : "f"(partner_198), "f"(bb_201));
                        float e_202;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_202) : "f"(_sub_435), "f"(_sub_437));
                        sums_3[15] = s_200;
                        float _add_439;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_439) : "f"(errs_1[15]), "f"(partner_err_199));
                        float _add_438;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_438) : "f"(_add_439), "f"(e_202));
                        errs_1[15] = _add_438;
                        float partner_203 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 1);
                        float partner_err_204 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 1);
                        float s_205;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_205) : "f"(sums_3[16]), "f"(partner_203));
                        float bb_206;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_206) : "f"(s_205), "f"(sums_3[16]));
                        float _sub_441;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_441) : "f"(s_205), "f"(bb_206));
                        float _sub_440;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_440) : "f"(sums_3[16]), "f"(_sub_441));
                        float _sub_442;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_442) : "f"(partner_203), "f"(bb_206));
                        float e_207;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_207) : "f"(_sub_440), "f"(_sub_442));
                        sums_3[16] = s_205;
                        float _add_444;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_444) : "f"(errs_1[16]), "f"(partner_err_204));
                        float _add_443;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_443) : "f"(_add_444), "f"(e_207));
                        errs_1[16] = _add_443;
                        float partner_208 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 1);
                        float partner_err_209 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 1);
                        float s_210;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_210) : "f"(sums_3[17]), "f"(partner_208));
                        float bb_211;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_211) : "f"(s_210), "f"(sums_3[17]));
                        float _sub_446;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_446) : "f"(s_210), "f"(bb_211));
                        float _sub_445;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_445) : "f"(sums_3[17]), "f"(_sub_446));
                        float _sub_447;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_447) : "f"(partner_208), "f"(bb_211));
                        float e_212;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_212) : "f"(_sub_445), "f"(_sub_447));
                        sums_3[17] = s_210;
                        float _add_449;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(errs_1[17]), "f"(partner_err_209));
                        float _add_448;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_448) : "f"(_add_449), "f"(e_212));
                        errs_1[17] = _add_448;
                        float partner_213 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 1);
                        float partner_err_214 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 1);
                        float s_215;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_215) : "f"(sums_3[18]), "f"(partner_213));
                        float bb_216;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_216) : "f"(s_215), "f"(sums_3[18]));
                        float _sub_451;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_451) : "f"(s_215), "f"(bb_216));
                        float _sub_450;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_450) : "f"(sums_3[18]), "f"(_sub_451));
                        float _sub_452;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_452) : "f"(partner_213), "f"(bb_216));
                        float e_217;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_217) : "f"(_sub_450), "f"(_sub_452));
                        sums_3[18] = s_215;
                        float _add_454;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_454) : "f"(errs_1[18]), "f"(partner_err_214));
                        float _add_453;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_453) : "f"(_add_454), "f"(e_217));
                        errs_1[18] = _add_453;
                        float partner_218 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 1);
                        float partner_err_219 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 1);
                        float s_220;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_220) : "f"(sums_3[19]), "f"(partner_218));
                        float bb_221;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_221) : "f"(s_220), "f"(sums_3[19]));
                        float _sub_456;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_456) : "f"(s_220), "f"(bb_221));
                        float _sub_455;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_455) : "f"(sums_3[19]), "f"(_sub_456));
                        float _sub_457;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_457) : "f"(partner_218), "f"(bb_221));
                        float e_222;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_222) : "f"(_sub_455), "f"(_sub_457));
                        sums_3[19] = s_220;
                        float _add_459;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_459) : "f"(errs_1[19]), "f"(partner_err_219));
                        float _add_458;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_458) : "f"(_add_459), "f"(e_222));
                        errs_1[19] = _add_458;
                        float partner_223 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 1);
                        float partner_err_224 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 1);
                        float s_225;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_225) : "f"(sums_3[20]), "f"(partner_223));
                        float bb_226;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_226) : "f"(s_225), "f"(sums_3[20]));
                        float _sub_461;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_461) : "f"(s_225), "f"(bb_226));
                        float _sub_460;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_460) : "f"(sums_3[20]), "f"(_sub_461));
                        float _sub_462;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_462) : "f"(partner_223), "f"(bb_226));
                        float e_227;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_227) : "f"(_sub_460), "f"(_sub_462));
                        sums_3[20] = s_225;
                        float _add_464;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_464) : "f"(errs_1[20]), "f"(partner_err_224));
                        float _add_463;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_463) : "f"(_add_464), "f"(e_227));
                        errs_1[20] = _add_463;
                        float partner_228 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 1);
                        float partner_err_229 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 1);
                        float s_230;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_230) : "f"(sums_3[21]), "f"(partner_228));
                        float bb_231;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_231) : "f"(s_230), "f"(sums_3[21]));
                        float _sub_466;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_466) : "f"(s_230), "f"(bb_231));
                        float _sub_465;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_465) : "f"(sums_3[21]), "f"(_sub_466));
                        float _sub_467;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_467) : "f"(partner_228), "f"(bb_231));
                        float e_232;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_232) : "f"(_sub_465), "f"(_sub_467));
                        sums_3[21] = s_230;
                        float _add_469;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_469) : "f"(errs_1[21]), "f"(partner_err_229));
                        float _add_468;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_468) : "f"(_add_469), "f"(e_232));
                        errs_1[21] = _add_468;
                        float partner_233 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 1);
                        float partner_err_234 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 1);
                        float s_235;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_235) : "f"(sums_3[22]), "f"(partner_233));
                        float bb_236;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_236) : "f"(s_235), "f"(sums_3[22]));
                        float _sub_471;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_471) : "f"(s_235), "f"(bb_236));
                        float _sub_470;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_470) : "f"(sums_3[22]), "f"(_sub_471));
                        float _sub_472;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_472) : "f"(partner_233), "f"(bb_236));
                        float e_237;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_237) : "f"(_sub_470), "f"(_sub_472));
                        sums_3[22] = s_235;
                        float _add_474;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_474) : "f"(errs_1[22]), "f"(partner_err_234));
                        float _add_473;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_473) : "f"(_add_474), "f"(e_237));
                        errs_1[22] = _add_473;
                        float partner_238 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 1);
                        float partner_err_239 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 1);
                        float s_240;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_240) : "f"(sums_3[23]), "f"(partner_238));
                        float bb_241;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_241) : "f"(s_240), "f"(sums_3[23]));
                        float _sub_476;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_476) : "f"(s_240), "f"(bb_241));
                        float _sub_475;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_475) : "f"(sums_3[23]), "f"(_sub_476));
                        float _sub_477;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_477) : "f"(partner_238), "f"(bb_241));
                        float e_242;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_242) : "f"(_sub_475), "f"(_sub_477));
                        sums_3[23] = s_240;
                        float _add_479;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_479) : "f"(errs_1[23]), "f"(partner_err_239));
                        float _add_478;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_478) : "f"(_add_479), "f"(e_242));
                        errs_1[23] = _add_478;
                        float partner_243 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 1);
                        float partner_err_244 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 1);
                        float s_245;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_245) : "f"(sums_3[24]), "f"(partner_243));
                        float bb_246;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_246) : "f"(s_245), "f"(sums_3[24]));
                        float _sub_481;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_481) : "f"(s_245), "f"(bb_246));
                        float _sub_480;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_480) : "f"(sums_3[24]), "f"(_sub_481));
                        float _sub_482;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_482) : "f"(partner_243), "f"(bb_246));
                        float e_247;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_247) : "f"(_sub_480), "f"(_sub_482));
                        sums_3[24] = s_245;
                        float _add_484;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_484) : "f"(errs_1[24]), "f"(partner_err_244));
                        float _add_483;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_483) : "f"(_add_484), "f"(e_247));
                        errs_1[24] = _add_483;
                        float partner_248 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 2);
                        float partner_err_249 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 2);
                        float s_250;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_250) : "f"(sums_3[0]), "f"(partner_248));
                        float bb_251;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_251) : "f"(s_250), "f"(sums_3[0]));
                        float _sub_486;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_486) : "f"(s_250), "f"(bb_251));
                        float _sub_485;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_485) : "f"(sums_3[0]), "f"(_sub_486));
                        float _sub_487;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_487) : "f"(partner_248), "f"(bb_251));
                        float e_252;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_252) : "f"(_sub_485), "f"(_sub_487));
                        sums_3[0] = s_250;
                        float _add_489;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_489) : "f"(errs_1[0]), "f"(partner_err_249));
                        float _add_488;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_488) : "f"(_add_489), "f"(e_252));
                        errs_1[0] = _add_488;
                        float partner_253 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 2);
                        float partner_err_254 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 2);
                        float s_255;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_255) : "f"(sums_3[1]), "f"(partner_253));
                        float bb_256;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_256) : "f"(s_255), "f"(sums_3[1]));
                        float _sub_491;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_491) : "f"(s_255), "f"(bb_256));
                        float _sub_490;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_490) : "f"(sums_3[1]), "f"(_sub_491));
                        float _sub_492;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_492) : "f"(partner_253), "f"(bb_256));
                        float e_257;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_257) : "f"(_sub_490), "f"(_sub_492));
                        sums_3[1] = s_255;
                        float _add_494;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_494) : "f"(errs_1[1]), "f"(partner_err_254));
                        float _add_493;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_493) : "f"(_add_494), "f"(e_257));
                        errs_1[1] = _add_493;
                        float partner_258 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 2);
                        float partner_err_259 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 2);
                        float s_260;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_260) : "f"(sums_3[2]), "f"(partner_258));
                        float bb_261;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_261) : "f"(s_260), "f"(sums_3[2]));
                        float _sub_496;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_496) : "f"(s_260), "f"(bb_261));
                        float _sub_495;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_495) : "f"(sums_3[2]), "f"(_sub_496));
                        float _sub_497;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_497) : "f"(partner_258), "f"(bb_261));
                        float e_262;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_262) : "f"(_sub_495), "f"(_sub_497));
                        sums_3[2] = s_260;
                        float _add_499;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_499) : "f"(errs_1[2]), "f"(partner_err_259));
                        float _add_498;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_498) : "f"(_add_499), "f"(e_262));
                        errs_1[2] = _add_498;
                        float partner_263 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 2);
                        float partner_err_264 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 2);
                        float s_265;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_265) : "f"(sums_3[3]), "f"(partner_263));
                        float bb_266;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_266) : "f"(s_265), "f"(sums_3[3]));
                        float _sub_501;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_501) : "f"(s_265), "f"(bb_266));
                        float _sub_500;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_500) : "f"(sums_3[3]), "f"(_sub_501));
                        float _sub_502;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_502) : "f"(partner_263), "f"(bb_266));
                        float e_267;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_267) : "f"(_sub_500), "f"(_sub_502));
                        sums_3[3] = s_265;
                        float _add_504;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_504) : "f"(errs_1[3]), "f"(partner_err_264));
                        float _add_503;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_503) : "f"(_add_504), "f"(e_267));
                        errs_1[3] = _add_503;
                        float partner_268 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 2);
                        float partner_err_269 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 2);
                        float s_270;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_270) : "f"(sums_3[4]), "f"(partner_268));
                        float bb_271;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_271) : "f"(s_270), "f"(sums_3[4]));
                        float _sub_506;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_506) : "f"(s_270), "f"(bb_271));
                        float _sub_505;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_505) : "f"(sums_3[4]), "f"(_sub_506));
                        float _sub_507;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_507) : "f"(partner_268), "f"(bb_271));
                        float e_272;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_272) : "f"(_sub_505), "f"(_sub_507));
                        sums_3[4] = s_270;
                        float _add_509;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_509) : "f"(errs_1[4]), "f"(partner_err_269));
                        float _add_508;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_508) : "f"(_add_509), "f"(e_272));
                        errs_1[4] = _add_508;
                        float partner_273 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 2);
                        float partner_err_274 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 2);
                        float s_275;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_275) : "f"(sums_3[5]), "f"(partner_273));
                        float bb_276;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_276) : "f"(s_275), "f"(sums_3[5]));
                        float _sub_511;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_511) : "f"(s_275), "f"(bb_276));
                        float _sub_510;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_510) : "f"(sums_3[5]), "f"(_sub_511));
                        float _sub_512;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_512) : "f"(partner_273), "f"(bb_276));
                        float e_277;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_277) : "f"(_sub_510), "f"(_sub_512));
                        sums_3[5] = s_275;
                        float _add_514;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_514) : "f"(errs_1[5]), "f"(partner_err_274));
                        float _add_513;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_513) : "f"(_add_514), "f"(e_277));
                        errs_1[5] = _add_513;
                        float partner_278 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 2);
                        float partner_err_279 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 2);
                        float s_280;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_280) : "f"(sums_3[6]), "f"(partner_278));
                        float bb_281;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_281) : "f"(s_280), "f"(sums_3[6]));
                        float _sub_516;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_516) : "f"(s_280), "f"(bb_281));
                        float _sub_515;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_515) : "f"(sums_3[6]), "f"(_sub_516));
                        float _sub_517;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_517) : "f"(partner_278), "f"(bb_281));
                        float e_282;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_282) : "f"(_sub_515), "f"(_sub_517));
                        sums_3[6] = s_280;
                        float _add_519;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_519) : "f"(errs_1[6]), "f"(partner_err_279));
                        float _add_518;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_518) : "f"(_add_519), "f"(e_282));
                        errs_1[6] = _add_518;
                        float partner_283 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 2);
                        float partner_err_284 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 2);
                        float s_285;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_285) : "f"(sums_3[7]), "f"(partner_283));
                        float bb_286;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_286) : "f"(s_285), "f"(sums_3[7]));
                        float _sub_521;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_521) : "f"(s_285), "f"(bb_286));
                        float _sub_520;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_520) : "f"(sums_3[7]), "f"(_sub_521));
                        float _sub_522;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_522) : "f"(partner_283), "f"(bb_286));
                        float e_287;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_287) : "f"(_sub_520), "f"(_sub_522));
                        sums_3[7] = s_285;
                        float _add_524;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_524) : "f"(errs_1[7]), "f"(partner_err_284));
                        float _add_523;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_523) : "f"(_add_524), "f"(e_287));
                        errs_1[7] = _add_523;
                        float partner_288 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 2);
                        float partner_err_289 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 2);
                        float s_290;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_290) : "f"(sums_3[8]), "f"(partner_288));
                        float bb_291;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_291) : "f"(s_290), "f"(sums_3[8]));
                        float _sub_526;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_526) : "f"(s_290), "f"(bb_291));
                        float _sub_525;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_525) : "f"(sums_3[8]), "f"(_sub_526));
                        float _sub_527;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_527) : "f"(partner_288), "f"(bb_291));
                        float e_292;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_292) : "f"(_sub_525), "f"(_sub_527));
                        sums_3[8] = s_290;
                        float _add_529;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_529) : "f"(errs_1[8]), "f"(partner_err_289));
                        float _add_528;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_528) : "f"(_add_529), "f"(e_292));
                        errs_1[8] = _add_528;
                        float partner_293 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 2);
                        float partner_err_294 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 2);
                        float s_295;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_295) : "f"(sums_3[9]), "f"(partner_293));
                        float bb_296;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_296) : "f"(s_295), "f"(sums_3[9]));
                        float _sub_531;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_531) : "f"(s_295), "f"(bb_296));
                        float _sub_530;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_530) : "f"(sums_3[9]), "f"(_sub_531));
                        float _sub_532;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_532) : "f"(partner_293), "f"(bb_296));
                        float e_297;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_297) : "f"(_sub_530), "f"(_sub_532));
                        sums_3[9] = s_295;
                        float _add_534;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_534) : "f"(errs_1[9]), "f"(partner_err_294));
                        float _add_533;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_533) : "f"(_add_534), "f"(e_297));
                        errs_1[9] = _add_533;
                        float partner_298 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 2);
                        float partner_err_299 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 2);
                        float s_300;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_300) : "f"(sums_3[10]), "f"(partner_298));
                        float bb_301;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_301) : "f"(s_300), "f"(sums_3[10]));
                        float _sub_536;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_536) : "f"(s_300), "f"(bb_301));
                        float _sub_535;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_535) : "f"(sums_3[10]), "f"(_sub_536));
                        float _sub_537;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_537) : "f"(partner_298), "f"(bb_301));
                        float e_302;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_302) : "f"(_sub_535), "f"(_sub_537));
                        sums_3[10] = s_300;
                        float _add_539;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_539) : "f"(errs_1[10]), "f"(partner_err_299));
                        float _add_538;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_538) : "f"(_add_539), "f"(e_302));
                        errs_1[10] = _add_538;
                        float partner_303 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 2);
                        float partner_err_304 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 2);
                        float s_305;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_305) : "f"(sums_3[11]), "f"(partner_303));
                        float bb_306;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_306) : "f"(s_305), "f"(sums_3[11]));
                        float _sub_541;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_541) : "f"(s_305), "f"(bb_306));
                        float _sub_540;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_540) : "f"(sums_3[11]), "f"(_sub_541));
                        float _sub_542;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_542) : "f"(partner_303), "f"(bb_306));
                        float e_307;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_307) : "f"(_sub_540), "f"(_sub_542));
                        sums_3[11] = s_305;
                        float _add_544;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_544) : "f"(errs_1[11]), "f"(partner_err_304));
                        float _add_543;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_543) : "f"(_add_544), "f"(e_307));
                        errs_1[11] = _add_543;
                        float partner_308 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 2);
                        float partner_err_309 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 2);
                        float s_310;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_310) : "f"(sums_3[12]), "f"(partner_308));
                        float bb_311;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_311) : "f"(s_310), "f"(sums_3[12]));
                        float _sub_546;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_546) : "f"(s_310), "f"(bb_311));
                        float _sub_545;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_545) : "f"(sums_3[12]), "f"(_sub_546));
                        float _sub_547;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_547) : "f"(partner_308), "f"(bb_311));
                        float e_312;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_312) : "f"(_sub_545), "f"(_sub_547));
                        sums_3[12] = s_310;
                        float _add_549;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_549) : "f"(errs_1[12]), "f"(partner_err_309));
                        float _add_548;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_548) : "f"(_add_549), "f"(e_312));
                        errs_1[12] = _add_548;
                        float partner_313 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 2);
                        float partner_err_314 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 2);
                        float s_315;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_315) : "f"(sums_3[13]), "f"(partner_313));
                        float bb_316;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_316) : "f"(s_315), "f"(sums_3[13]));
                        float _sub_551;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_551) : "f"(s_315), "f"(bb_316));
                        float _sub_550;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_550) : "f"(sums_3[13]), "f"(_sub_551));
                        float _sub_552;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_552) : "f"(partner_313), "f"(bb_316));
                        float e_317;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_317) : "f"(_sub_550), "f"(_sub_552));
                        sums_3[13] = s_315;
                        float _add_554;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_554) : "f"(errs_1[13]), "f"(partner_err_314));
                        float _add_553;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_553) : "f"(_add_554), "f"(e_317));
                        errs_1[13] = _add_553;
                        float partner_318 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 2);
                        float partner_err_319 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 2);
                        float s_320;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_320) : "f"(sums_3[14]), "f"(partner_318));
                        float bb_321;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_321) : "f"(s_320), "f"(sums_3[14]));
                        float _sub_556;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_556) : "f"(s_320), "f"(bb_321));
                        float _sub_555;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_555) : "f"(sums_3[14]), "f"(_sub_556));
                        float _sub_557;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_557) : "f"(partner_318), "f"(bb_321));
                        float e_322;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_322) : "f"(_sub_555), "f"(_sub_557));
                        sums_3[14] = s_320;
                        float _add_559;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_559) : "f"(errs_1[14]), "f"(partner_err_319));
                        float _add_558;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_558) : "f"(_add_559), "f"(e_322));
                        errs_1[14] = _add_558;
                        float partner_323 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 2);
                        float partner_err_324 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 2);
                        float s_325;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_325) : "f"(sums_3[15]), "f"(partner_323));
                        float bb_326;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_326) : "f"(s_325), "f"(sums_3[15]));
                        float _sub_561;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_561) : "f"(s_325), "f"(bb_326));
                        float _sub_560;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_560) : "f"(sums_3[15]), "f"(_sub_561));
                        float _sub_562;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_562) : "f"(partner_323), "f"(bb_326));
                        float e_327;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_327) : "f"(_sub_560), "f"(_sub_562));
                        sums_3[15] = s_325;
                        float _add_564;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_564) : "f"(errs_1[15]), "f"(partner_err_324));
                        float _add_563;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_563) : "f"(_add_564), "f"(e_327));
                        errs_1[15] = _add_563;
                        float partner_328 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 2);
                        float partner_err_329 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 2);
                        float s_330;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_330) : "f"(sums_3[16]), "f"(partner_328));
                        float bb_331;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_331) : "f"(s_330), "f"(sums_3[16]));
                        float _sub_566;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_566) : "f"(s_330), "f"(bb_331));
                        float _sub_565;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_565) : "f"(sums_3[16]), "f"(_sub_566));
                        float _sub_567;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_567) : "f"(partner_328), "f"(bb_331));
                        float e_332;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_332) : "f"(_sub_565), "f"(_sub_567));
                        sums_3[16] = s_330;
                        float _add_569;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_569) : "f"(errs_1[16]), "f"(partner_err_329));
                        float _add_568;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_568) : "f"(_add_569), "f"(e_332));
                        errs_1[16] = _add_568;
                        float partner_333 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 2);
                        float partner_err_334 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 2);
                        float s_335;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_335) : "f"(sums_3[17]), "f"(partner_333));
                        float bb_336;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_336) : "f"(s_335), "f"(sums_3[17]));
                        float _sub_571;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_571) : "f"(s_335), "f"(bb_336));
                        float _sub_570;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_570) : "f"(sums_3[17]), "f"(_sub_571));
                        float _sub_572;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_572) : "f"(partner_333), "f"(bb_336));
                        float e_337;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_337) : "f"(_sub_570), "f"(_sub_572));
                        sums_3[17] = s_335;
                        float _add_574;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_574) : "f"(errs_1[17]), "f"(partner_err_334));
                        float _add_573;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_573) : "f"(_add_574), "f"(e_337));
                        errs_1[17] = _add_573;
                        float partner_338 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 2);
                        float partner_err_339 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 2);
                        float s_340;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_340) : "f"(sums_3[18]), "f"(partner_338));
                        float bb_341;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_341) : "f"(s_340), "f"(sums_3[18]));
                        float _sub_576;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_576) : "f"(s_340), "f"(bb_341));
                        float _sub_575;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_575) : "f"(sums_3[18]), "f"(_sub_576));
                        float _sub_577;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_577) : "f"(partner_338), "f"(bb_341));
                        float e_342;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_342) : "f"(_sub_575), "f"(_sub_577));
                        sums_3[18] = s_340;
                        float _add_579;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_579) : "f"(errs_1[18]), "f"(partner_err_339));
                        float _add_578;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_578) : "f"(_add_579), "f"(e_342));
                        errs_1[18] = _add_578;
                        float partner_343 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 2);
                        float partner_err_344 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 2);
                        float s_345;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_345) : "f"(sums_3[19]), "f"(partner_343));
                        float bb_346;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_346) : "f"(s_345), "f"(sums_3[19]));
                        float _sub_581;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_581) : "f"(s_345), "f"(bb_346));
                        float _sub_580;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_580) : "f"(sums_3[19]), "f"(_sub_581));
                        float _sub_582;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_582) : "f"(partner_343), "f"(bb_346));
                        float e_347;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_347) : "f"(_sub_580), "f"(_sub_582));
                        sums_3[19] = s_345;
                        float _add_584;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_584) : "f"(errs_1[19]), "f"(partner_err_344));
                        float _add_583;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_583) : "f"(_add_584), "f"(e_347));
                        errs_1[19] = _add_583;
                        float partner_348 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 2);
                        float partner_err_349 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 2);
                        float s_350;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_350) : "f"(sums_3[20]), "f"(partner_348));
                        float bb_351;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_351) : "f"(s_350), "f"(sums_3[20]));
                        float _sub_586;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_586) : "f"(s_350), "f"(bb_351));
                        float _sub_585;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_585) : "f"(sums_3[20]), "f"(_sub_586));
                        float _sub_587;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_587) : "f"(partner_348), "f"(bb_351));
                        float e_352;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_352) : "f"(_sub_585), "f"(_sub_587));
                        sums_3[20] = s_350;
                        float _add_589;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_589) : "f"(errs_1[20]), "f"(partner_err_349));
                        float _add_588;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_588) : "f"(_add_589), "f"(e_352));
                        errs_1[20] = _add_588;
                        float partner_353 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 2);
                        float partner_err_354 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 2);
                        float s_355;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_355) : "f"(sums_3[21]), "f"(partner_353));
                        float bb_356;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_356) : "f"(s_355), "f"(sums_3[21]));
                        float _sub_591;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_591) : "f"(s_355), "f"(bb_356));
                        float _sub_590;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_590) : "f"(sums_3[21]), "f"(_sub_591));
                        float _sub_592;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_592) : "f"(partner_353), "f"(bb_356));
                        float e_357;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_357) : "f"(_sub_590), "f"(_sub_592));
                        sums_3[21] = s_355;
                        float _add_594;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_594) : "f"(errs_1[21]), "f"(partner_err_354));
                        float _add_593;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_593) : "f"(_add_594), "f"(e_357));
                        errs_1[21] = _add_593;
                        float partner_358 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 2);
                        float partner_err_359 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 2);
                        float s_360;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_360) : "f"(sums_3[22]), "f"(partner_358));
                        float bb_361;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_361) : "f"(s_360), "f"(sums_3[22]));
                        float _sub_596;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_596) : "f"(s_360), "f"(bb_361));
                        float _sub_595;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_595) : "f"(sums_3[22]), "f"(_sub_596));
                        float _sub_597;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_597) : "f"(partner_358), "f"(bb_361));
                        float e_362;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_362) : "f"(_sub_595), "f"(_sub_597));
                        sums_3[22] = s_360;
                        float _add_599;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_599) : "f"(errs_1[22]), "f"(partner_err_359));
                        float _add_598;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_598) : "f"(_add_599), "f"(e_362));
                        errs_1[22] = _add_598;
                        float partner_363 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 2);
                        float partner_err_364 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 2);
                        float s_365;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_365) : "f"(sums_3[23]), "f"(partner_363));
                        float bb_366;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_366) : "f"(s_365), "f"(sums_3[23]));
                        float _sub_601;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_601) : "f"(s_365), "f"(bb_366));
                        float _sub_600;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_600) : "f"(sums_3[23]), "f"(_sub_601));
                        float _sub_602;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_602) : "f"(partner_363), "f"(bb_366));
                        float e_367;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_367) : "f"(_sub_600), "f"(_sub_602));
                        sums_3[23] = s_365;
                        float _add_604;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_604) : "f"(errs_1[23]), "f"(partner_err_364));
                        float _add_603;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_603) : "f"(_add_604), "f"(e_367));
                        errs_1[23] = _add_603;
                        float partner_368 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 2);
                        float partner_err_369 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 2);
                        float s_370;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_370) : "f"(sums_3[24]), "f"(partner_368));
                        float bb_371;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_371) : "f"(s_370), "f"(sums_3[24]));
                        float _sub_606;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_606) : "f"(s_370), "f"(bb_371));
                        float _sub_605;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_605) : "f"(sums_3[24]), "f"(_sub_606));
                        float _sub_607;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_607) : "f"(partner_368), "f"(bb_371));
                        float e_372;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_372) : "f"(_sub_605), "f"(_sub_607));
                        sums_3[24] = s_370;
                        float _add_609;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_609) : "f"(errs_1[24]), "f"(partner_err_369));
                        float _add_608;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_608) : "f"(_add_609), "f"(e_372));
                        errs_1[24] = _add_608;
                        float partner_373 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 4);
                        float partner_err_374 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 4);
                        float s_375;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_375) : "f"(sums_3[0]), "f"(partner_373));
                        float bb_376;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_376) : "f"(s_375), "f"(sums_3[0]));
                        float _sub_611;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_611) : "f"(s_375), "f"(bb_376));
                        float _sub_610;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_610) : "f"(sums_3[0]), "f"(_sub_611));
                        float _sub_612;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_612) : "f"(partner_373), "f"(bb_376));
                        float e_377;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_377) : "f"(_sub_610), "f"(_sub_612));
                        sums_3[0] = s_375;
                        float _add_614;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_614) : "f"(errs_1[0]), "f"(partner_err_374));
                        float _add_613;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_613) : "f"(_add_614), "f"(e_377));
                        errs_1[0] = _add_613;
                        float partner_378 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 4);
                        float partner_err_379 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 4);
                        float s_380;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_380) : "f"(sums_3[1]), "f"(partner_378));
                        float bb_381;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_381) : "f"(s_380), "f"(sums_3[1]));
                        float _sub_616;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_616) : "f"(s_380), "f"(bb_381));
                        float _sub_615;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_615) : "f"(sums_3[1]), "f"(_sub_616));
                        float _sub_617;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_617) : "f"(partner_378), "f"(bb_381));
                        float e_382;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_382) : "f"(_sub_615), "f"(_sub_617));
                        sums_3[1] = s_380;
                        float _add_619;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_619) : "f"(errs_1[1]), "f"(partner_err_379));
                        float _add_618;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_618) : "f"(_add_619), "f"(e_382));
                        errs_1[1] = _add_618;
                        float partner_383 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 4);
                        float partner_err_384 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 4);
                        float s_385;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_385) : "f"(sums_3[2]), "f"(partner_383));
                        float bb_386;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_386) : "f"(s_385), "f"(sums_3[2]));
                        float _sub_621;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_621) : "f"(s_385), "f"(bb_386));
                        float _sub_620;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_620) : "f"(sums_3[2]), "f"(_sub_621));
                        float _sub_622;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_622) : "f"(partner_383), "f"(bb_386));
                        float e_387;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_387) : "f"(_sub_620), "f"(_sub_622));
                        sums_3[2] = s_385;
                        float _add_624;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_624) : "f"(errs_1[2]), "f"(partner_err_384));
                        float _add_623;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_623) : "f"(_add_624), "f"(e_387));
                        errs_1[2] = _add_623;
                        float partner_388 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 4);
                        float partner_err_389 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 4);
                        float s_390;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_390) : "f"(sums_3[3]), "f"(partner_388));
                        float bb_391;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_391) : "f"(s_390), "f"(sums_3[3]));
                        float _sub_626;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_626) : "f"(s_390), "f"(bb_391));
                        float _sub_625;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_625) : "f"(sums_3[3]), "f"(_sub_626));
                        float _sub_627;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_627) : "f"(partner_388), "f"(bb_391));
                        float e_392;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_392) : "f"(_sub_625), "f"(_sub_627));
                        sums_3[3] = s_390;
                        float _add_629;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_629) : "f"(errs_1[3]), "f"(partner_err_389));
                        float _add_628;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_628) : "f"(_add_629), "f"(e_392));
                        errs_1[3] = _add_628;
                        float partner_393 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 4);
                        float partner_err_394 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 4);
                        float s_395;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_395) : "f"(sums_3[4]), "f"(partner_393));
                        float bb_396;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_396) : "f"(s_395), "f"(sums_3[4]));
                        float _sub_631;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_631) : "f"(s_395), "f"(bb_396));
                        float _sub_630;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_630) : "f"(sums_3[4]), "f"(_sub_631));
                        float _sub_632;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_632) : "f"(partner_393), "f"(bb_396));
                        float e_397;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_397) : "f"(_sub_630), "f"(_sub_632));
                        sums_3[4] = s_395;
                        float _add_634;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_634) : "f"(errs_1[4]), "f"(partner_err_394));
                        float _add_633;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_633) : "f"(_add_634), "f"(e_397));
                        errs_1[4] = _add_633;
                        float partner_398 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 4);
                        float partner_err_399 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 4);
                        float s_400;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_400) : "f"(sums_3[5]), "f"(partner_398));
                        float bb_401;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_401) : "f"(s_400), "f"(sums_3[5]));
                        float _sub_636;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_636) : "f"(s_400), "f"(bb_401));
                        float _sub_635;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_635) : "f"(sums_3[5]), "f"(_sub_636));
                        float _sub_637;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_637) : "f"(partner_398), "f"(bb_401));
                        float e_402;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_402) : "f"(_sub_635), "f"(_sub_637));
                        sums_3[5] = s_400;
                        float _add_639;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_639) : "f"(errs_1[5]), "f"(partner_err_399));
                        float _add_638;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_638) : "f"(_add_639), "f"(e_402));
                        errs_1[5] = _add_638;
                        float partner_403 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 4);
                        float partner_err_404 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 4);
                        float s_405;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_405) : "f"(sums_3[6]), "f"(partner_403));
                        float bb_406;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_406) : "f"(s_405), "f"(sums_3[6]));
                        float _sub_641;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_641) : "f"(s_405), "f"(bb_406));
                        float _sub_640;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_640) : "f"(sums_3[6]), "f"(_sub_641));
                        float _sub_642;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_642) : "f"(partner_403), "f"(bb_406));
                        float e_407;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_407) : "f"(_sub_640), "f"(_sub_642));
                        sums_3[6] = s_405;
                        float _add_644;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_644) : "f"(errs_1[6]), "f"(partner_err_404));
                        float _add_643;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_643) : "f"(_add_644), "f"(e_407));
                        errs_1[6] = _add_643;
                        float partner_408 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 4);
                        float partner_err_409 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 4);
                        float s_410;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_410) : "f"(sums_3[7]), "f"(partner_408));
                        float bb_411;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_411) : "f"(s_410), "f"(sums_3[7]));
                        float _sub_646;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_646) : "f"(s_410), "f"(bb_411));
                        float _sub_645;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_645) : "f"(sums_3[7]), "f"(_sub_646));
                        float _sub_647;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_647) : "f"(partner_408), "f"(bb_411));
                        float e_412;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_412) : "f"(_sub_645), "f"(_sub_647));
                        sums_3[7] = s_410;
                        float _add_649;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_649) : "f"(errs_1[7]), "f"(partner_err_409));
                        float _add_648;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_648) : "f"(_add_649), "f"(e_412));
                        errs_1[7] = _add_648;
                        float partner_413 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 4);
                        float partner_err_414 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 4);
                        float s_415;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_415) : "f"(sums_3[8]), "f"(partner_413));
                        float bb_416;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_416) : "f"(s_415), "f"(sums_3[8]));
                        float _sub_651;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_651) : "f"(s_415), "f"(bb_416));
                        float _sub_650;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_650) : "f"(sums_3[8]), "f"(_sub_651));
                        float _sub_652;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_652) : "f"(partner_413), "f"(bb_416));
                        float e_417;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_417) : "f"(_sub_650), "f"(_sub_652));
                        sums_3[8] = s_415;
                        float _add_654;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(errs_1[8]), "f"(partner_err_414));
                        float _add_653;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_653) : "f"(_add_654), "f"(e_417));
                        errs_1[8] = _add_653;
                        float partner_418 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 4);
                        float partner_err_419 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 4);
                        float s_420;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_420) : "f"(sums_3[9]), "f"(partner_418));
                        float bb_421;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_421) : "f"(s_420), "f"(sums_3[9]));
                        float _sub_656;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_656) : "f"(s_420), "f"(bb_421));
                        float _sub_655;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_655) : "f"(sums_3[9]), "f"(_sub_656));
                        float _sub_657;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_657) : "f"(partner_418), "f"(bb_421));
                        float e_422;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_422) : "f"(_sub_655), "f"(_sub_657));
                        sums_3[9] = s_420;
                        float _add_659;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_659) : "f"(errs_1[9]), "f"(partner_err_419));
                        float _add_658;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_658) : "f"(_add_659), "f"(e_422));
                        errs_1[9] = _add_658;
                        float partner_423 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 4);
                        float partner_err_424 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 4);
                        float s_425;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_425) : "f"(sums_3[10]), "f"(partner_423));
                        float bb_426;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_426) : "f"(s_425), "f"(sums_3[10]));
                        float _sub_661;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_661) : "f"(s_425), "f"(bb_426));
                        float _sub_660;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_660) : "f"(sums_3[10]), "f"(_sub_661));
                        float _sub_662;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_662) : "f"(partner_423), "f"(bb_426));
                        float e_427;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_427) : "f"(_sub_660), "f"(_sub_662));
                        sums_3[10] = s_425;
                        float _add_664;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_664) : "f"(errs_1[10]), "f"(partner_err_424));
                        float _add_663;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_663) : "f"(_add_664), "f"(e_427));
                        errs_1[10] = _add_663;
                        float partner_428 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 4);
                        float partner_err_429 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 4);
                        float s_430;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_430) : "f"(sums_3[11]), "f"(partner_428));
                        float bb_431;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_431) : "f"(s_430), "f"(sums_3[11]));
                        float _sub_666;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_666) : "f"(s_430), "f"(bb_431));
                        float _sub_665;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_665) : "f"(sums_3[11]), "f"(_sub_666));
                        float _sub_667;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_667) : "f"(partner_428), "f"(bb_431));
                        float e_432;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_432) : "f"(_sub_665), "f"(_sub_667));
                        sums_3[11] = s_430;
                        float _add_669;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_669) : "f"(errs_1[11]), "f"(partner_err_429));
                        float _add_668;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_668) : "f"(_add_669), "f"(e_432));
                        errs_1[11] = _add_668;
                        float partner_433 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 4);
                        float partner_err_434 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 4);
                        float s_435;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_435) : "f"(sums_3[12]), "f"(partner_433));
                        float bb_436;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_436) : "f"(s_435), "f"(sums_3[12]));
                        float _sub_671;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_671) : "f"(s_435), "f"(bb_436));
                        float _sub_670;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_670) : "f"(sums_3[12]), "f"(_sub_671));
                        float _sub_672;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_672) : "f"(partner_433), "f"(bb_436));
                        float e_437;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_437) : "f"(_sub_670), "f"(_sub_672));
                        sums_3[12] = s_435;
                        float _add_674;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_674) : "f"(errs_1[12]), "f"(partner_err_434));
                        float _add_673;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_673) : "f"(_add_674), "f"(e_437));
                        errs_1[12] = _add_673;
                        float partner_438 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 4);
                        float partner_err_439 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 4);
                        float s_440;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_440) : "f"(sums_3[13]), "f"(partner_438));
                        float bb_441;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_441) : "f"(s_440), "f"(sums_3[13]));
                        float _sub_676;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_676) : "f"(s_440), "f"(bb_441));
                        float _sub_675;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_675) : "f"(sums_3[13]), "f"(_sub_676));
                        float _sub_677;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_677) : "f"(partner_438), "f"(bb_441));
                        float e_442;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_442) : "f"(_sub_675), "f"(_sub_677));
                        sums_3[13] = s_440;
                        float _add_679;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_679) : "f"(errs_1[13]), "f"(partner_err_439));
                        float _add_678;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_678) : "f"(_add_679), "f"(e_442));
                        errs_1[13] = _add_678;
                        float partner_443 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 4);
                        float partner_err_444 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 4);
                        float s_445;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_445) : "f"(sums_3[14]), "f"(partner_443));
                        float bb_446;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_446) : "f"(s_445), "f"(sums_3[14]));
                        float _sub_681;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_681) : "f"(s_445), "f"(bb_446));
                        float _sub_680;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_680) : "f"(sums_3[14]), "f"(_sub_681));
                        float _sub_682;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_682) : "f"(partner_443), "f"(bb_446));
                        float e_447;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_447) : "f"(_sub_680), "f"(_sub_682));
                        sums_3[14] = s_445;
                        float _add_684;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_684) : "f"(errs_1[14]), "f"(partner_err_444));
                        float _add_683;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_683) : "f"(_add_684), "f"(e_447));
                        errs_1[14] = _add_683;
                        float partner_448 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 4);
                        float partner_err_449 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 4);
                        float s_450;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_450) : "f"(sums_3[15]), "f"(partner_448));
                        float bb_451;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_451) : "f"(s_450), "f"(sums_3[15]));
                        float _sub_686;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_686) : "f"(s_450), "f"(bb_451));
                        float _sub_685;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_685) : "f"(sums_3[15]), "f"(_sub_686));
                        float _sub_687;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_687) : "f"(partner_448), "f"(bb_451));
                        float e_452;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_452) : "f"(_sub_685), "f"(_sub_687));
                        sums_3[15] = s_450;
                        float _add_689;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_689) : "f"(errs_1[15]), "f"(partner_err_449));
                        float _add_688;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_688) : "f"(_add_689), "f"(e_452));
                        errs_1[15] = _add_688;
                        float partner_453 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 4);
                        float partner_err_454 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 4);
                        float s_455;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_455) : "f"(sums_3[16]), "f"(partner_453));
                        float bb_456;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_456) : "f"(s_455), "f"(sums_3[16]));
                        float _sub_691;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_691) : "f"(s_455), "f"(bb_456));
                        float _sub_690;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_690) : "f"(sums_3[16]), "f"(_sub_691));
                        float _sub_692;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_692) : "f"(partner_453), "f"(bb_456));
                        float e_457;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_457) : "f"(_sub_690), "f"(_sub_692));
                        sums_3[16] = s_455;
                        float _add_694;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_694) : "f"(errs_1[16]), "f"(partner_err_454));
                        float _add_693;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_693) : "f"(_add_694), "f"(e_457));
                        errs_1[16] = _add_693;
                        float partner_458 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 4);
                        float partner_err_459 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 4);
                        float s_460;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_460) : "f"(sums_3[17]), "f"(partner_458));
                        float bb_461;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_461) : "f"(s_460), "f"(sums_3[17]));
                        float _sub_696;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_696) : "f"(s_460), "f"(bb_461));
                        float _sub_695;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_695) : "f"(sums_3[17]), "f"(_sub_696));
                        float _sub_697;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_697) : "f"(partner_458), "f"(bb_461));
                        float e_462;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_462) : "f"(_sub_695), "f"(_sub_697));
                        sums_3[17] = s_460;
                        float _add_699;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_699) : "f"(errs_1[17]), "f"(partner_err_459));
                        float _add_698;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_698) : "f"(_add_699), "f"(e_462));
                        errs_1[17] = _add_698;
                        float partner_463 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 4);
                        float partner_err_464 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 4);
                        float s_465;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_465) : "f"(sums_3[18]), "f"(partner_463));
                        float bb_466;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_466) : "f"(s_465), "f"(sums_3[18]));
                        float _sub_701;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_701) : "f"(s_465), "f"(bb_466));
                        float _sub_700;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_700) : "f"(sums_3[18]), "f"(_sub_701));
                        float _sub_702;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_702) : "f"(partner_463), "f"(bb_466));
                        float e_467;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_467) : "f"(_sub_700), "f"(_sub_702));
                        sums_3[18] = s_465;
                        float _add_704;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_704) : "f"(errs_1[18]), "f"(partner_err_464));
                        float _add_703;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_703) : "f"(_add_704), "f"(e_467));
                        errs_1[18] = _add_703;
                        float partner_468 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 4);
                        float partner_err_469 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 4);
                        float s_470;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_470) : "f"(sums_3[19]), "f"(partner_468));
                        float bb_471;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_471) : "f"(s_470), "f"(sums_3[19]));
                        float _sub_706;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_706) : "f"(s_470), "f"(bb_471));
                        float _sub_705;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_705) : "f"(sums_3[19]), "f"(_sub_706));
                        float _sub_707;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_707) : "f"(partner_468), "f"(bb_471));
                        float e_472;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_472) : "f"(_sub_705), "f"(_sub_707));
                        sums_3[19] = s_470;
                        float _add_709;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_709) : "f"(errs_1[19]), "f"(partner_err_469));
                        float _add_708;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_708) : "f"(_add_709), "f"(e_472));
                        errs_1[19] = _add_708;
                        float partner_473 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 4);
                        float partner_err_474 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 4);
                        float s_475;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_475) : "f"(sums_3[20]), "f"(partner_473));
                        float bb_476;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_476) : "f"(s_475), "f"(sums_3[20]));
                        float _sub_711;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_711) : "f"(s_475), "f"(bb_476));
                        float _sub_710;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_710) : "f"(sums_3[20]), "f"(_sub_711));
                        float _sub_712;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_712) : "f"(partner_473), "f"(bb_476));
                        float e_477;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_477) : "f"(_sub_710), "f"(_sub_712));
                        sums_3[20] = s_475;
                        float _add_714;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_714) : "f"(errs_1[20]), "f"(partner_err_474));
                        float _add_713;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_713) : "f"(_add_714), "f"(e_477));
                        errs_1[20] = _add_713;
                        float partner_478 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 4);
                        float partner_err_479 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 4);
                        float s_480;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_480) : "f"(sums_3[21]), "f"(partner_478));
                        float bb_481;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_481) : "f"(s_480), "f"(sums_3[21]));
                        float _sub_716;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_716) : "f"(s_480), "f"(bb_481));
                        float _sub_715;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_715) : "f"(sums_3[21]), "f"(_sub_716));
                        float _sub_717;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_717) : "f"(partner_478), "f"(bb_481));
                        float e_482;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_482) : "f"(_sub_715), "f"(_sub_717));
                        sums_3[21] = s_480;
                        float _add_719;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_719) : "f"(errs_1[21]), "f"(partner_err_479));
                        float _add_718;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_718) : "f"(_add_719), "f"(e_482));
                        errs_1[21] = _add_718;
                        float partner_483 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 4);
                        float partner_err_484 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 4);
                        float s_485;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_485) : "f"(sums_3[22]), "f"(partner_483));
                        float bb_486;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_486) : "f"(s_485), "f"(sums_3[22]));
                        float _sub_721;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_721) : "f"(s_485), "f"(bb_486));
                        float _sub_720;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_720) : "f"(sums_3[22]), "f"(_sub_721));
                        float _sub_722;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_722) : "f"(partner_483), "f"(bb_486));
                        float e_487;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_487) : "f"(_sub_720), "f"(_sub_722));
                        sums_3[22] = s_485;
                        float _add_724;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_724) : "f"(errs_1[22]), "f"(partner_err_484));
                        float _add_723;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_723) : "f"(_add_724), "f"(e_487));
                        errs_1[22] = _add_723;
                        float partner_488 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 4);
                        float partner_err_489 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 4);
                        float s_490;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_490) : "f"(sums_3[23]), "f"(partner_488));
                        float bb_491;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_491) : "f"(s_490), "f"(sums_3[23]));
                        float _sub_726;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_726) : "f"(s_490), "f"(bb_491));
                        float _sub_725;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_725) : "f"(sums_3[23]), "f"(_sub_726));
                        float _sub_727;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_727) : "f"(partner_488), "f"(bb_491));
                        float e_492;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_492) : "f"(_sub_725), "f"(_sub_727));
                        sums_3[23] = s_490;
                        float _add_729;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_729) : "f"(errs_1[23]), "f"(partner_err_489));
                        float _add_728;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_728) : "f"(_add_729), "f"(e_492));
                        errs_1[23] = _add_728;
                        float partner_493 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 4);
                        float partner_err_494 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 4);
                        float s_495;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_495) : "f"(sums_3[24]), "f"(partner_493));
                        float bb_496;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_496) : "f"(s_495), "f"(sums_3[24]));
                        float _sub_731;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_731) : "f"(s_495), "f"(bb_496));
                        float _sub_730;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_730) : "f"(sums_3[24]), "f"(_sub_731));
                        float _sub_732;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_732) : "f"(partner_493), "f"(bb_496));
                        float e_497;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_497) : "f"(_sub_730), "f"(_sub_732));
                        sums_3[24] = s_495;
                        float _add_734;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_734) : "f"(errs_1[24]), "f"(partner_err_494));
                        float _add_733;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_733) : "f"(_add_734), "f"(e_497));
                        errs_1[24] = _add_733;
                        float partner_498 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 8);
                        float partner_err_499 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 8);
                        float s_500;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_500) : "f"(sums_3[0]), "f"(partner_498));
                        float bb_501;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_501) : "f"(s_500), "f"(sums_3[0]));
                        float _sub_736;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_736) : "f"(s_500), "f"(bb_501));
                        float _sub_735;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_735) : "f"(sums_3[0]), "f"(_sub_736));
                        float _sub_737;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_737) : "f"(partner_498), "f"(bb_501));
                        float e_502;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_502) : "f"(_sub_735), "f"(_sub_737));
                        sums_3[0] = s_500;
                        float _add_739;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_739) : "f"(errs_1[0]), "f"(partner_err_499));
                        float _add_738;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_738) : "f"(_add_739), "f"(e_502));
                        errs_1[0] = _add_738;
                        float partner_503 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 8);
                        float partner_err_504 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 8);
                        float s_505;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_505) : "f"(sums_3[1]), "f"(partner_503));
                        float bb_506;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_506) : "f"(s_505), "f"(sums_3[1]));
                        float _sub_741;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_741) : "f"(s_505), "f"(bb_506));
                        float _sub_740;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_740) : "f"(sums_3[1]), "f"(_sub_741));
                        float _sub_742;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_742) : "f"(partner_503), "f"(bb_506));
                        float e_507;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_507) : "f"(_sub_740), "f"(_sub_742));
                        sums_3[1] = s_505;
                        float _add_744;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_744) : "f"(errs_1[1]), "f"(partner_err_504));
                        float _add_743;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_743) : "f"(_add_744), "f"(e_507));
                        errs_1[1] = _add_743;
                        float partner_508 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 8);
                        float partner_err_509 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 8);
                        float s_510;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_510) : "f"(sums_3[2]), "f"(partner_508));
                        float bb_511;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_511) : "f"(s_510), "f"(sums_3[2]));
                        float _sub_746;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_746) : "f"(s_510), "f"(bb_511));
                        float _sub_745;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_745) : "f"(sums_3[2]), "f"(_sub_746));
                        float _sub_747;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_747) : "f"(partner_508), "f"(bb_511));
                        float e_512;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_512) : "f"(_sub_745), "f"(_sub_747));
                        sums_3[2] = s_510;
                        float _add_749;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_749) : "f"(errs_1[2]), "f"(partner_err_509));
                        float _add_748;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_748) : "f"(_add_749), "f"(e_512));
                        errs_1[2] = _add_748;
                        float partner_513 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 8);
                        float partner_err_514 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 8);
                        float s_515;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_515) : "f"(sums_3[3]), "f"(partner_513));
                        float bb_516;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_516) : "f"(s_515), "f"(sums_3[3]));
                        float _sub_751;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_751) : "f"(s_515), "f"(bb_516));
                        float _sub_750;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_750) : "f"(sums_3[3]), "f"(_sub_751));
                        float _sub_752;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_752) : "f"(partner_513), "f"(bb_516));
                        float e_517;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_517) : "f"(_sub_750), "f"(_sub_752));
                        sums_3[3] = s_515;
                        float _add_754;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_754) : "f"(errs_1[3]), "f"(partner_err_514));
                        float _add_753;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_753) : "f"(_add_754), "f"(e_517));
                        errs_1[3] = _add_753;
                        float partner_518 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 8);
                        float partner_err_519 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 8);
                        float s_520;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_520) : "f"(sums_3[4]), "f"(partner_518));
                        float bb_521;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_521) : "f"(s_520), "f"(sums_3[4]));
                        float _sub_756;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_756) : "f"(s_520), "f"(bb_521));
                        float _sub_755;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_755) : "f"(sums_3[4]), "f"(_sub_756));
                        float _sub_757;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_757) : "f"(partner_518), "f"(bb_521));
                        float e_522;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_522) : "f"(_sub_755), "f"(_sub_757));
                        sums_3[4] = s_520;
                        float _add_759;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_759) : "f"(errs_1[4]), "f"(partner_err_519));
                        float _add_758;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_758) : "f"(_add_759), "f"(e_522));
                        errs_1[4] = _add_758;
                        float partner_523 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 8);
                        float partner_err_524 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 8);
                        float s_525;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_525) : "f"(sums_3[5]), "f"(partner_523));
                        float bb_526;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_526) : "f"(s_525), "f"(sums_3[5]));
                        float _sub_761;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_761) : "f"(s_525), "f"(bb_526));
                        float _sub_760;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_760) : "f"(sums_3[5]), "f"(_sub_761));
                        float _sub_762;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_762) : "f"(partner_523), "f"(bb_526));
                        float e_527;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_527) : "f"(_sub_760), "f"(_sub_762));
                        sums_3[5] = s_525;
                        float _add_764;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_764) : "f"(errs_1[5]), "f"(partner_err_524));
                        float _add_763;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_763) : "f"(_add_764), "f"(e_527));
                        errs_1[5] = _add_763;
                        float partner_528 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 8);
                        float partner_err_529 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 8);
                        float s_530;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_530) : "f"(sums_3[6]), "f"(partner_528));
                        float bb_531;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_531) : "f"(s_530), "f"(sums_3[6]));
                        float _sub_766;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_766) : "f"(s_530), "f"(bb_531));
                        float _sub_765;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_765) : "f"(sums_3[6]), "f"(_sub_766));
                        float _sub_767;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_767) : "f"(partner_528), "f"(bb_531));
                        float e_532;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_532) : "f"(_sub_765), "f"(_sub_767));
                        sums_3[6] = s_530;
                        float _add_769;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_769) : "f"(errs_1[6]), "f"(partner_err_529));
                        float _add_768;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_768) : "f"(_add_769), "f"(e_532));
                        errs_1[6] = _add_768;
                        float partner_533 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 8);
                        float partner_err_534 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 8);
                        float s_535;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_535) : "f"(sums_3[7]), "f"(partner_533));
                        float bb_536;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_536) : "f"(s_535), "f"(sums_3[7]));
                        float _sub_771;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_771) : "f"(s_535), "f"(bb_536));
                        float _sub_770;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_770) : "f"(sums_3[7]), "f"(_sub_771));
                        float _sub_772;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_772) : "f"(partner_533), "f"(bb_536));
                        float e_537;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_537) : "f"(_sub_770), "f"(_sub_772));
                        sums_3[7] = s_535;
                        float _add_774;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_774) : "f"(errs_1[7]), "f"(partner_err_534));
                        float _add_773;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_773) : "f"(_add_774), "f"(e_537));
                        errs_1[7] = _add_773;
                        float partner_538 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 8);
                        float partner_err_539 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 8);
                        float s_540;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_540) : "f"(sums_3[8]), "f"(partner_538));
                        float bb_541;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_541) : "f"(s_540), "f"(sums_3[8]));
                        float _sub_776;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_776) : "f"(s_540), "f"(bb_541));
                        float _sub_775;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_775) : "f"(sums_3[8]), "f"(_sub_776));
                        float _sub_777;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_777) : "f"(partner_538), "f"(bb_541));
                        float e_542;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_542) : "f"(_sub_775), "f"(_sub_777));
                        sums_3[8] = s_540;
                        float _add_779;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_779) : "f"(errs_1[8]), "f"(partner_err_539));
                        float _add_778;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_778) : "f"(_add_779), "f"(e_542));
                        errs_1[8] = _add_778;
                        float partner_543 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 8);
                        float partner_err_544 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 8);
                        float s_545;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_545) : "f"(sums_3[9]), "f"(partner_543));
                        float bb_546;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_546) : "f"(s_545), "f"(sums_3[9]));
                        float _sub_781;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_781) : "f"(s_545), "f"(bb_546));
                        float _sub_780;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_780) : "f"(sums_3[9]), "f"(_sub_781));
                        float _sub_782;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_782) : "f"(partner_543), "f"(bb_546));
                        float e_547;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_547) : "f"(_sub_780), "f"(_sub_782));
                        sums_3[9] = s_545;
                        float _add_784;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_784) : "f"(errs_1[9]), "f"(partner_err_544));
                        float _add_783;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_783) : "f"(_add_784), "f"(e_547));
                        errs_1[9] = _add_783;
                        float partner_548 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 8);
                        float partner_err_549 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 8);
                        float s_550;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_550) : "f"(sums_3[10]), "f"(partner_548));
                        float bb_551;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_551) : "f"(s_550), "f"(sums_3[10]));
                        float _sub_786;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_786) : "f"(s_550), "f"(bb_551));
                        float _sub_785;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_785) : "f"(sums_3[10]), "f"(_sub_786));
                        float _sub_787;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_787) : "f"(partner_548), "f"(bb_551));
                        float e_552;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_552) : "f"(_sub_785), "f"(_sub_787));
                        sums_3[10] = s_550;
                        float _add_789;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_789) : "f"(errs_1[10]), "f"(partner_err_549));
                        float _add_788;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_788) : "f"(_add_789), "f"(e_552));
                        errs_1[10] = _add_788;
                        float partner_553 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 8);
                        float partner_err_554 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 8);
                        float s_555;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_555) : "f"(sums_3[11]), "f"(partner_553));
                        float bb_556;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_556) : "f"(s_555), "f"(sums_3[11]));
                        float _sub_791;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_791) : "f"(s_555), "f"(bb_556));
                        float _sub_790;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_790) : "f"(sums_3[11]), "f"(_sub_791));
                        float _sub_792;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_792) : "f"(partner_553), "f"(bb_556));
                        float e_557;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_557) : "f"(_sub_790), "f"(_sub_792));
                        sums_3[11] = s_555;
                        float _add_794;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_794) : "f"(errs_1[11]), "f"(partner_err_554));
                        float _add_793;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_793) : "f"(_add_794), "f"(e_557));
                        errs_1[11] = _add_793;
                        float partner_558 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 8);
                        float partner_err_559 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 8);
                        float s_560;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_560) : "f"(sums_3[12]), "f"(partner_558));
                        float bb_561;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_561) : "f"(s_560), "f"(sums_3[12]));
                        float _sub_796;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_796) : "f"(s_560), "f"(bb_561));
                        float _sub_795;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_795) : "f"(sums_3[12]), "f"(_sub_796));
                        float _sub_797;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_797) : "f"(partner_558), "f"(bb_561));
                        float e_562;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_562) : "f"(_sub_795), "f"(_sub_797));
                        sums_3[12] = s_560;
                        float _add_799;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_799) : "f"(errs_1[12]), "f"(partner_err_559));
                        float _add_798;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_798) : "f"(_add_799), "f"(e_562));
                        errs_1[12] = _add_798;
                        float partner_563 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 8);
                        float partner_err_564 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 8);
                        float s_565;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_565) : "f"(sums_3[13]), "f"(partner_563));
                        float bb_566;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_566) : "f"(s_565), "f"(sums_3[13]));
                        float _sub_801;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_801) : "f"(s_565), "f"(bb_566));
                        float _sub_800;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_800) : "f"(sums_3[13]), "f"(_sub_801));
                        float _sub_802;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_802) : "f"(partner_563), "f"(bb_566));
                        float e_567;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_567) : "f"(_sub_800), "f"(_sub_802));
                        sums_3[13] = s_565;
                        float _add_804;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_804) : "f"(errs_1[13]), "f"(partner_err_564));
                        float _add_803;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_803) : "f"(_add_804), "f"(e_567));
                        errs_1[13] = _add_803;
                        float partner_568 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 8);
                        float partner_err_569 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 8);
                        float s_570;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_570) : "f"(sums_3[14]), "f"(partner_568));
                        float bb_571;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_571) : "f"(s_570), "f"(sums_3[14]));
                        float _sub_806;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_806) : "f"(s_570), "f"(bb_571));
                        float _sub_805;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_805) : "f"(sums_3[14]), "f"(_sub_806));
                        float _sub_807;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_807) : "f"(partner_568), "f"(bb_571));
                        float e_572;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_572) : "f"(_sub_805), "f"(_sub_807));
                        sums_3[14] = s_570;
                        float _add_809;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_809) : "f"(errs_1[14]), "f"(partner_err_569));
                        float _add_808;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_808) : "f"(_add_809), "f"(e_572));
                        errs_1[14] = _add_808;
                        float partner_573 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 8);
                        float partner_err_574 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 8);
                        float s_575;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_575) : "f"(sums_3[15]), "f"(partner_573));
                        float bb_576;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_576) : "f"(s_575), "f"(sums_3[15]));
                        float _sub_811;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_811) : "f"(s_575), "f"(bb_576));
                        float _sub_810;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_810) : "f"(sums_3[15]), "f"(_sub_811));
                        float _sub_812;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_812) : "f"(partner_573), "f"(bb_576));
                        float e_577;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_577) : "f"(_sub_810), "f"(_sub_812));
                        sums_3[15] = s_575;
                        float _add_814;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_814) : "f"(errs_1[15]), "f"(partner_err_574));
                        float _add_813;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_813) : "f"(_add_814), "f"(e_577));
                        errs_1[15] = _add_813;
                        float partner_578 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 8);
                        float partner_err_579 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 8);
                        float s_580;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_580) : "f"(sums_3[16]), "f"(partner_578));
                        float bb_581;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_581) : "f"(s_580), "f"(sums_3[16]));
                        float _sub_816;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_816) : "f"(s_580), "f"(bb_581));
                        float _sub_815;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_815) : "f"(sums_3[16]), "f"(_sub_816));
                        float _sub_817;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_817) : "f"(partner_578), "f"(bb_581));
                        float e_582;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_582) : "f"(_sub_815), "f"(_sub_817));
                        sums_3[16] = s_580;
                        float _add_819;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_819) : "f"(errs_1[16]), "f"(partner_err_579));
                        float _add_818;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_818) : "f"(_add_819), "f"(e_582));
                        errs_1[16] = _add_818;
                        float partner_583 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 8);
                        float partner_err_584 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 8);
                        float s_585;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_585) : "f"(sums_3[17]), "f"(partner_583));
                        float bb_586;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_586) : "f"(s_585), "f"(sums_3[17]));
                        float _sub_821;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_821) : "f"(s_585), "f"(bb_586));
                        float _sub_820;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_820) : "f"(sums_3[17]), "f"(_sub_821));
                        float _sub_822;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_822) : "f"(partner_583), "f"(bb_586));
                        float e_587;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_587) : "f"(_sub_820), "f"(_sub_822));
                        sums_3[17] = s_585;
                        float _add_824;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_824) : "f"(errs_1[17]), "f"(partner_err_584));
                        float _add_823;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_823) : "f"(_add_824), "f"(e_587));
                        errs_1[17] = _add_823;
                        float partner_588 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 8);
                        float partner_err_589 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 8);
                        float s_590;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_590) : "f"(sums_3[18]), "f"(partner_588));
                        float bb_591;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_591) : "f"(s_590), "f"(sums_3[18]));
                        float _sub_826;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_826) : "f"(s_590), "f"(bb_591));
                        float _sub_825;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_825) : "f"(sums_3[18]), "f"(_sub_826));
                        float _sub_827;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_827) : "f"(partner_588), "f"(bb_591));
                        float e_592;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_592) : "f"(_sub_825), "f"(_sub_827));
                        sums_3[18] = s_590;
                        float _add_829;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_829) : "f"(errs_1[18]), "f"(partner_err_589));
                        float _add_828;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_828) : "f"(_add_829), "f"(e_592));
                        errs_1[18] = _add_828;
                        float partner_593 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 8);
                        float partner_err_594 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 8);
                        float s_595;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_595) : "f"(sums_3[19]), "f"(partner_593));
                        float bb_596;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_596) : "f"(s_595), "f"(sums_3[19]));
                        float _sub_831;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_831) : "f"(s_595), "f"(bb_596));
                        float _sub_830;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_830) : "f"(sums_3[19]), "f"(_sub_831));
                        float _sub_832;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_832) : "f"(partner_593), "f"(bb_596));
                        float e_597;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_597) : "f"(_sub_830), "f"(_sub_832));
                        sums_3[19] = s_595;
                        float _add_834;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_834) : "f"(errs_1[19]), "f"(partner_err_594));
                        float _add_833;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_833) : "f"(_add_834), "f"(e_597));
                        errs_1[19] = _add_833;
                        float partner_598 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 8);
                        float partner_err_599 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 8);
                        float s_600;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_600) : "f"(sums_3[20]), "f"(partner_598));
                        float bb_601;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_601) : "f"(s_600), "f"(sums_3[20]));
                        float _sub_836;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_836) : "f"(s_600), "f"(bb_601));
                        float _sub_835;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_835) : "f"(sums_3[20]), "f"(_sub_836));
                        float _sub_837;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_837) : "f"(partner_598), "f"(bb_601));
                        float e_602;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_602) : "f"(_sub_835), "f"(_sub_837));
                        sums_3[20] = s_600;
                        float _add_839;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_839) : "f"(errs_1[20]), "f"(partner_err_599));
                        float _add_838;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_838) : "f"(_add_839), "f"(e_602));
                        errs_1[20] = _add_838;
                        float partner_603 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 8);
                        float partner_err_604 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 8);
                        float s_605;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_605) : "f"(sums_3[21]), "f"(partner_603));
                        float bb_606;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_606) : "f"(s_605), "f"(sums_3[21]));
                        float _sub_841;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_841) : "f"(s_605), "f"(bb_606));
                        float _sub_840;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_840) : "f"(sums_3[21]), "f"(_sub_841));
                        float _sub_842;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_842) : "f"(partner_603), "f"(bb_606));
                        float e_607;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_607) : "f"(_sub_840), "f"(_sub_842));
                        sums_3[21] = s_605;
                        float _add_844;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_844) : "f"(errs_1[21]), "f"(partner_err_604));
                        float _add_843;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_843) : "f"(_add_844), "f"(e_607));
                        errs_1[21] = _add_843;
                        float partner_608 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 8);
                        float partner_err_609 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 8);
                        float s_610;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_610) : "f"(sums_3[22]), "f"(partner_608));
                        float bb_611;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_611) : "f"(s_610), "f"(sums_3[22]));
                        float _sub_846;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_846) : "f"(s_610), "f"(bb_611));
                        float _sub_845;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_845) : "f"(sums_3[22]), "f"(_sub_846));
                        float _sub_847;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_847) : "f"(partner_608), "f"(bb_611));
                        float e_612;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_612) : "f"(_sub_845), "f"(_sub_847));
                        sums_3[22] = s_610;
                        float _add_849;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_849) : "f"(errs_1[22]), "f"(partner_err_609));
                        float _add_848;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_848) : "f"(_add_849), "f"(e_612));
                        errs_1[22] = _add_848;
                        float partner_613 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 8);
                        float partner_err_614 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 8);
                        float s_615;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_615) : "f"(sums_3[23]), "f"(partner_613));
                        float bb_616;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_616) : "f"(s_615), "f"(sums_3[23]));
                        float _sub_851;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_851) : "f"(s_615), "f"(bb_616));
                        float _sub_850;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_850) : "f"(sums_3[23]), "f"(_sub_851));
                        float _sub_852;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_852) : "f"(partner_613), "f"(bb_616));
                        float e_617;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_617) : "f"(_sub_850), "f"(_sub_852));
                        sums_3[23] = s_615;
                        float _add_854;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_854) : "f"(errs_1[23]), "f"(partner_err_614));
                        float _add_853;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_853) : "f"(_add_854), "f"(e_617));
                        errs_1[23] = _add_853;
                        float partner_618 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 8);
                        float partner_err_619 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 8);
                        float s_620;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_620) : "f"(sums_3[24]), "f"(partner_618));
                        float bb_621;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_621) : "f"(s_620), "f"(sums_3[24]));
                        float _sub_856;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_856) : "f"(s_620), "f"(bb_621));
                        float _sub_855;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_855) : "f"(sums_3[24]), "f"(_sub_856));
                        float _sub_857;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_857) : "f"(partner_618), "f"(bb_621));
                        float e_622;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_622) : "f"(_sub_855), "f"(_sub_857));
                        sums_3[24] = s_620;
                        float _add_859;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_859) : "f"(errs_1[24]), "f"(partner_err_619));
                        float _add_858;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_858) : "f"(_add_859), "f"(e_622));
                        errs_1[24] = _add_858;
                        float partner_623 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 16);
                        float partner_err_624 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 16);
                        float s_625;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_625) : "f"(sums_3[0]), "f"(partner_623));
                        float bb_626;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_626) : "f"(s_625), "f"(sums_3[0]));
                        float _sub_861;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_861) : "f"(s_625), "f"(bb_626));
                        float _sub_860;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_860) : "f"(sums_3[0]), "f"(_sub_861));
                        float _sub_862;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_862) : "f"(partner_623), "f"(bb_626));
                        float e_627;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_627) : "f"(_sub_860), "f"(_sub_862));
                        sums_3[0] = s_625;
                        float _add_864;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_864) : "f"(errs_1[0]), "f"(partner_err_624));
                        float _add_863;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_863) : "f"(_add_864), "f"(e_627));
                        errs_1[0] = _add_863;
                        float partner_628 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 16);
                        float partner_err_629 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 16);
                        float s_630;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_630) : "f"(sums_3[1]), "f"(partner_628));
                        float bb_631;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_631) : "f"(s_630), "f"(sums_3[1]));
                        float _sub_866;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_866) : "f"(s_630), "f"(bb_631));
                        float _sub_865;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_865) : "f"(sums_3[1]), "f"(_sub_866));
                        float _sub_867;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_867) : "f"(partner_628), "f"(bb_631));
                        float e_632;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_632) : "f"(_sub_865), "f"(_sub_867));
                        sums_3[1] = s_630;
                        float _add_869;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_869) : "f"(errs_1[1]), "f"(partner_err_629));
                        float _add_868;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_868) : "f"(_add_869), "f"(e_632));
                        errs_1[1] = _add_868;
                        float partner_633 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 16);
                        float partner_err_634 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 16);
                        float s_635;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_635) : "f"(sums_3[2]), "f"(partner_633));
                        float bb_636;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_636) : "f"(s_635), "f"(sums_3[2]));
                        float _sub_871;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_871) : "f"(s_635), "f"(bb_636));
                        float _sub_870;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_870) : "f"(sums_3[2]), "f"(_sub_871));
                        float _sub_872;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_872) : "f"(partner_633), "f"(bb_636));
                        float e_637;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_637) : "f"(_sub_870), "f"(_sub_872));
                        sums_3[2] = s_635;
                        float _add_874;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_874) : "f"(errs_1[2]), "f"(partner_err_634));
                        float _add_873;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_873) : "f"(_add_874), "f"(e_637));
                        errs_1[2] = _add_873;
                        float partner_638 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 16);
                        float partner_err_639 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 16);
                        float s_640;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_640) : "f"(sums_3[3]), "f"(partner_638));
                        float bb_641;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_641) : "f"(s_640), "f"(sums_3[3]));
                        float _sub_876;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_876) : "f"(s_640), "f"(bb_641));
                        float _sub_875;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_875) : "f"(sums_3[3]), "f"(_sub_876));
                        float _sub_877;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_877) : "f"(partner_638), "f"(bb_641));
                        float e_642;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_642) : "f"(_sub_875), "f"(_sub_877));
                        sums_3[3] = s_640;
                        float _add_879;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_879) : "f"(errs_1[3]), "f"(partner_err_639));
                        float _add_878;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_878) : "f"(_add_879), "f"(e_642));
                        errs_1[3] = _add_878;
                        float partner_643 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 16);
                        float partner_err_644 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 16);
                        float s_645;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_645) : "f"(sums_3[4]), "f"(partner_643));
                        float bb_646;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_646) : "f"(s_645), "f"(sums_3[4]));
                        float _sub_881;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_881) : "f"(s_645), "f"(bb_646));
                        float _sub_880;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_880) : "f"(sums_3[4]), "f"(_sub_881));
                        float _sub_882;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_882) : "f"(partner_643), "f"(bb_646));
                        float e_647;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_647) : "f"(_sub_880), "f"(_sub_882));
                        sums_3[4] = s_645;
                        float _add_884;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_884) : "f"(errs_1[4]), "f"(partner_err_644));
                        float _add_883;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_883) : "f"(_add_884), "f"(e_647));
                        errs_1[4] = _add_883;
                        float partner_648 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 16);
                        float partner_err_649 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 16);
                        float s_650;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_650) : "f"(sums_3[5]), "f"(partner_648));
                        float bb_651;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_651) : "f"(s_650), "f"(sums_3[5]));
                        float _sub_886;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_886) : "f"(s_650), "f"(bb_651));
                        float _sub_885;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_885) : "f"(sums_3[5]), "f"(_sub_886));
                        float _sub_887;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_887) : "f"(partner_648), "f"(bb_651));
                        float e_652;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_652) : "f"(_sub_885), "f"(_sub_887));
                        sums_3[5] = s_650;
                        float _add_889;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_889) : "f"(errs_1[5]), "f"(partner_err_649));
                        float _add_888;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_888) : "f"(_add_889), "f"(e_652));
                        errs_1[5] = _add_888;
                        float partner_653 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 16);
                        float partner_err_654 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 16);
                        float s_655;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_655) : "f"(sums_3[6]), "f"(partner_653));
                        float bb_656;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_656) : "f"(s_655), "f"(sums_3[6]));
                        float _sub_891;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_891) : "f"(s_655), "f"(bb_656));
                        float _sub_890;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_890) : "f"(sums_3[6]), "f"(_sub_891));
                        float _sub_892;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_892) : "f"(partner_653), "f"(bb_656));
                        float e_657;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_657) : "f"(_sub_890), "f"(_sub_892));
                        sums_3[6] = s_655;
                        float _add_894;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_894) : "f"(errs_1[6]), "f"(partner_err_654));
                        float _add_893;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_893) : "f"(_add_894), "f"(e_657));
                        errs_1[6] = _add_893;
                        float partner_658 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 16);
                        float partner_err_659 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 16);
                        float s_660;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_660) : "f"(sums_3[7]), "f"(partner_658));
                        float bb_661;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_661) : "f"(s_660), "f"(sums_3[7]));
                        float _sub_896;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_896) : "f"(s_660), "f"(bb_661));
                        float _sub_895;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_895) : "f"(sums_3[7]), "f"(_sub_896));
                        float _sub_897;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_897) : "f"(partner_658), "f"(bb_661));
                        float e_662;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_662) : "f"(_sub_895), "f"(_sub_897));
                        sums_3[7] = s_660;
                        float _add_899;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_899) : "f"(errs_1[7]), "f"(partner_err_659));
                        float _add_898;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_898) : "f"(_add_899), "f"(e_662));
                        errs_1[7] = _add_898;
                        float partner_663 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 16);
                        float partner_err_664 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 16);
                        float s_665;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_665) : "f"(sums_3[8]), "f"(partner_663));
                        float bb_666;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_666) : "f"(s_665), "f"(sums_3[8]));
                        float _sub_901;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_901) : "f"(s_665), "f"(bb_666));
                        float _sub_900;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_900) : "f"(sums_3[8]), "f"(_sub_901));
                        float _sub_902;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_902) : "f"(partner_663), "f"(bb_666));
                        float e_667;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_667) : "f"(_sub_900), "f"(_sub_902));
                        sums_3[8] = s_665;
                        float _add_904;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_904) : "f"(errs_1[8]), "f"(partner_err_664));
                        float _add_903;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_903) : "f"(_add_904), "f"(e_667));
                        errs_1[8] = _add_903;
                        float partner_668 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 16);
                        float partner_err_669 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 16);
                        float s_670;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_670) : "f"(sums_3[9]), "f"(partner_668));
                        float bb_671;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_671) : "f"(s_670), "f"(sums_3[9]));
                        float _sub_906;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_906) : "f"(s_670), "f"(bb_671));
                        float _sub_905;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_905) : "f"(sums_3[9]), "f"(_sub_906));
                        float _sub_907;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_907) : "f"(partner_668), "f"(bb_671));
                        float e_672;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_672) : "f"(_sub_905), "f"(_sub_907));
                        sums_3[9] = s_670;
                        float _add_909;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_909) : "f"(errs_1[9]), "f"(partner_err_669));
                        float _add_908;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_908) : "f"(_add_909), "f"(e_672));
                        errs_1[9] = _add_908;
                        float partner_673 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 16);
                        float partner_err_674 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 16);
                        float s_675;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_675) : "f"(sums_3[10]), "f"(partner_673));
                        float bb_676;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_676) : "f"(s_675), "f"(sums_3[10]));
                        float _sub_911;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_911) : "f"(s_675), "f"(bb_676));
                        float _sub_910;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_910) : "f"(sums_3[10]), "f"(_sub_911));
                        float _sub_912;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_912) : "f"(partner_673), "f"(bb_676));
                        float e_677;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_677) : "f"(_sub_910), "f"(_sub_912));
                        sums_3[10] = s_675;
                        float _add_914;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_914) : "f"(errs_1[10]), "f"(partner_err_674));
                        float _add_913;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_913) : "f"(_add_914), "f"(e_677));
                        errs_1[10] = _add_913;
                        float partner_678 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 16);
                        float partner_err_679 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 16);
                        float s_680;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_680) : "f"(sums_3[11]), "f"(partner_678));
                        float bb_681;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_681) : "f"(s_680), "f"(sums_3[11]));
                        float _sub_916;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_916) : "f"(s_680), "f"(bb_681));
                        float _sub_915;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_915) : "f"(sums_3[11]), "f"(_sub_916));
                        float _sub_917;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_917) : "f"(partner_678), "f"(bb_681));
                        float e_682;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_682) : "f"(_sub_915), "f"(_sub_917));
                        sums_3[11] = s_680;
                        float _add_919;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_919) : "f"(errs_1[11]), "f"(partner_err_679));
                        float _add_918;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_918) : "f"(_add_919), "f"(e_682));
                        errs_1[11] = _add_918;
                        float partner_683 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 16);
                        float partner_err_684 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 16);
                        float s_685;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_685) : "f"(sums_3[12]), "f"(partner_683));
                        float bb_686;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_686) : "f"(s_685), "f"(sums_3[12]));
                        float _sub_921;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_921) : "f"(s_685), "f"(bb_686));
                        float _sub_920;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_920) : "f"(sums_3[12]), "f"(_sub_921));
                        float _sub_922;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_922) : "f"(partner_683), "f"(bb_686));
                        float e_687;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_687) : "f"(_sub_920), "f"(_sub_922));
                        sums_3[12] = s_685;
                        float _add_924;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_924) : "f"(errs_1[12]), "f"(partner_err_684));
                        float _add_923;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_923) : "f"(_add_924), "f"(e_687));
                        errs_1[12] = _add_923;
                        float partner_688 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 16);
                        float partner_err_689 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 16);
                        float s_690;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_690) : "f"(sums_3[13]), "f"(partner_688));
                        float bb_691;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_691) : "f"(s_690), "f"(sums_3[13]));
                        float _sub_926;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_926) : "f"(s_690), "f"(bb_691));
                        float _sub_925;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_925) : "f"(sums_3[13]), "f"(_sub_926));
                        float _sub_927;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_927) : "f"(partner_688), "f"(bb_691));
                        float e_692;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_692) : "f"(_sub_925), "f"(_sub_927));
                        sums_3[13] = s_690;
                        float _add_929;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_929) : "f"(errs_1[13]), "f"(partner_err_689));
                        float _add_928;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_928) : "f"(_add_929), "f"(e_692));
                        errs_1[13] = _add_928;
                        float partner_693 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 16);
                        float partner_err_694 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 16);
                        float s_695;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_695) : "f"(sums_3[14]), "f"(partner_693));
                        float bb_696;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_696) : "f"(s_695), "f"(sums_3[14]));
                        float _sub_931;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_931) : "f"(s_695), "f"(bb_696));
                        float _sub_930;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_930) : "f"(sums_3[14]), "f"(_sub_931));
                        float _sub_932;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_932) : "f"(partner_693), "f"(bb_696));
                        float e_697;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_697) : "f"(_sub_930), "f"(_sub_932));
                        sums_3[14] = s_695;
                        float _add_934;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_934) : "f"(errs_1[14]), "f"(partner_err_694));
                        float _add_933;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_933) : "f"(_add_934), "f"(e_697));
                        errs_1[14] = _add_933;
                        float partner_698 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 16);
                        float partner_err_699 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 16);
                        float s_700;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_700) : "f"(sums_3[15]), "f"(partner_698));
                        float bb_701;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_701) : "f"(s_700), "f"(sums_3[15]));
                        float _sub_936;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_936) : "f"(s_700), "f"(bb_701));
                        float _sub_935;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_935) : "f"(sums_3[15]), "f"(_sub_936));
                        float _sub_937;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_937) : "f"(partner_698), "f"(bb_701));
                        float e_702;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_702) : "f"(_sub_935), "f"(_sub_937));
                        sums_3[15] = s_700;
                        float _add_939;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_939) : "f"(errs_1[15]), "f"(partner_err_699));
                        float _add_938;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_938) : "f"(_add_939), "f"(e_702));
                        errs_1[15] = _add_938;
                        float partner_703 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 16);
                        float partner_err_704 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 16);
                        float s_705;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_705) : "f"(sums_3[16]), "f"(partner_703));
                        float bb_706;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_706) : "f"(s_705), "f"(sums_3[16]));
                        float _sub_941;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_941) : "f"(s_705), "f"(bb_706));
                        float _sub_940;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_940) : "f"(sums_3[16]), "f"(_sub_941));
                        float _sub_942;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_942) : "f"(partner_703), "f"(bb_706));
                        float e_707;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_707) : "f"(_sub_940), "f"(_sub_942));
                        sums_3[16] = s_705;
                        float _add_944;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_944) : "f"(errs_1[16]), "f"(partner_err_704));
                        float _add_943;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_943) : "f"(_add_944), "f"(e_707));
                        errs_1[16] = _add_943;
                        float partner_708 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 16);
                        float partner_err_709 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 16);
                        float s_710;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_710) : "f"(sums_3[17]), "f"(partner_708));
                        float bb_711;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_711) : "f"(s_710), "f"(sums_3[17]));
                        float _sub_946;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_946) : "f"(s_710), "f"(bb_711));
                        float _sub_945;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_945) : "f"(sums_3[17]), "f"(_sub_946));
                        float _sub_947;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_947) : "f"(partner_708), "f"(bb_711));
                        float e_712;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_712) : "f"(_sub_945), "f"(_sub_947));
                        sums_3[17] = s_710;
                        float _add_949;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_949) : "f"(errs_1[17]), "f"(partner_err_709));
                        float _add_948;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_948) : "f"(_add_949), "f"(e_712));
                        errs_1[17] = _add_948;
                        float partner_713 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 16);
                        float partner_err_714 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 16);
                        float s_715;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_715) : "f"(sums_3[18]), "f"(partner_713));
                        float bb_716;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_716) : "f"(s_715), "f"(sums_3[18]));
                        float _sub_951;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_951) : "f"(s_715), "f"(bb_716));
                        float _sub_950;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_950) : "f"(sums_3[18]), "f"(_sub_951));
                        float _sub_952;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_952) : "f"(partner_713), "f"(bb_716));
                        float e_717;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_717) : "f"(_sub_950), "f"(_sub_952));
                        sums_3[18] = s_715;
                        float _add_954;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_954) : "f"(errs_1[18]), "f"(partner_err_714));
                        float _add_953;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_953) : "f"(_add_954), "f"(e_717));
                        errs_1[18] = _add_953;
                        float partner_718 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 16);
                        float partner_err_719 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 16);
                        float s_720;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_720) : "f"(sums_3[19]), "f"(partner_718));
                        float bb_721;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_721) : "f"(s_720), "f"(sums_3[19]));
                        float _sub_956;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_956) : "f"(s_720), "f"(bb_721));
                        float _sub_955;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_955) : "f"(sums_3[19]), "f"(_sub_956));
                        float _sub_957;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_957) : "f"(partner_718), "f"(bb_721));
                        float e_722;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_722) : "f"(_sub_955), "f"(_sub_957));
                        sums_3[19] = s_720;
                        float _add_959;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_959) : "f"(errs_1[19]), "f"(partner_err_719));
                        float _add_958;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_958) : "f"(_add_959), "f"(e_722));
                        errs_1[19] = _add_958;
                        float partner_723 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 16);
                        float partner_err_724 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 16);
                        float s_725;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_725) : "f"(sums_3[20]), "f"(partner_723));
                        float bb_726;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_726) : "f"(s_725), "f"(sums_3[20]));
                        float _sub_961;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_961) : "f"(s_725), "f"(bb_726));
                        float _sub_960;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_960) : "f"(sums_3[20]), "f"(_sub_961));
                        float _sub_962;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_962) : "f"(partner_723), "f"(bb_726));
                        float e_727;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_727) : "f"(_sub_960), "f"(_sub_962));
                        sums_3[20] = s_725;
                        float _add_964;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_964) : "f"(errs_1[20]), "f"(partner_err_724));
                        float _add_963;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_963) : "f"(_add_964), "f"(e_727));
                        errs_1[20] = _add_963;
                        float partner_728 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 16);
                        float partner_err_729 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 16);
                        float s_730;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_730) : "f"(sums_3[21]), "f"(partner_728));
                        float bb_731;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_731) : "f"(s_730), "f"(sums_3[21]));
                        float _sub_966;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_966) : "f"(s_730), "f"(bb_731));
                        float _sub_965;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_965) : "f"(sums_3[21]), "f"(_sub_966));
                        float _sub_967;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_967) : "f"(partner_728), "f"(bb_731));
                        float e_732;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_732) : "f"(_sub_965), "f"(_sub_967));
                        sums_3[21] = s_730;
                        float _add_969;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_969) : "f"(errs_1[21]), "f"(partner_err_729));
                        float _add_968;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_968) : "f"(_add_969), "f"(e_732));
                        errs_1[21] = _add_968;
                        float partner_733 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 16);
                        float partner_err_734 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 16);
                        float s_735;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_735) : "f"(sums_3[22]), "f"(partner_733));
                        float bb_736;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_736) : "f"(s_735), "f"(sums_3[22]));
                        float _sub_971;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_971) : "f"(s_735), "f"(bb_736));
                        float _sub_970;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_970) : "f"(sums_3[22]), "f"(_sub_971));
                        float _sub_972;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_972) : "f"(partner_733), "f"(bb_736));
                        float e_737;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_737) : "f"(_sub_970), "f"(_sub_972));
                        sums_3[22] = s_735;
                        float _add_974;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_974) : "f"(errs_1[22]), "f"(partner_err_734));
                        float _add_973;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_973) : "f"(_add_974), "f"(e_737));
                        errs_1[22] = _add_973;
                        float partner_738 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 16);
                        float partner_err_739 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 16);
                        float s_740;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_740) : "f"(sums_3[23]), "f"(partner_738));
                        float bb_741;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_741) : "f"(s_740), "f"(sums_3[23]));
                        float _sub_976;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_976) : "f"(s_740), "f"(bb_741));
                        float _sub_975;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_975) : "f"(sums_3[23]), "f"(_sub_976));
                        float _sub_977;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_977) : "f"(partner_738), "f"(bb_741));
                        float e_742;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_742) : "f"(_sub_975), "f"(_sub_977));
                        sums_3[23] = s_740;
                        float _add_979;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_979) : "f"(errs_1[23]), "f"(partner_err_739));
                        float _add_978;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_978) : "f"(_add_979), "f"(e_742));
                        errs_1[23] = _add_978;
                        float partner_743 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 16);
                        float partner_err_744 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 16);
                        float s_745;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_745) : "f"(sums_3[24]), "f"(partner_743));
                        float bb_746;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_746) : "f"(s_745), "f"(sums_3[24]));
                        float _sub_981;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_981) : "f"(s_745), "f"(bb_746));
                        float _sub_980;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_980) : "f"(sums_3[24]), "f"(_sub_981));
                        float _sub_982;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_982) : "f"(partner_743), "f"(bb_746));
                        float e_747;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_747) : "f"(_sub_980), "f"(_sub_982));
                        sums_3[24] = s_745;
                        float _add_984;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_984) : "f"(errs_1[24]), "f"(partner_err_744));
                        float _add_983;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_983) : "f"(_add_984), "f"(e_747));
                        errs_1[24] = _add_983;
                        float s_748 = 0.0f;
                        float total_n;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n) : "f"(sums_3[0]), "f"(errs_1[0]));
                        s_748 = ((lane_in_warp == 0) ? total_n : s_748);
                        float total_n_749;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_749) : "f"(sums_3[1]), "f"(errs_1[1]));
                        s_748 = ((lane_in_warp == 1) ? total_n_749 : s_748);
                        float total_n_750;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_750) : "f"(sums_3[2]), "f"(errs_1[2]));
                        s_748 = ((lane_in_warp == 2) ? total_n_750 : s_748);
                        float total_n_751;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_751) : "f"(sums_3[3]), "f"(errs_1[3]));
                        s_748 = ((lane_in_warp == 3) ? total_n_751 : s_748);
                        float total_n_752;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_752) : "f"(sums_3[4]), "f"(errs_1[4]));
                        s_748 = ((lane_in_warp == 4) ? total_n_752 : s_748);
                        float total_n_753;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_753) : "f"(sums_3[5]), "f"(errs_1[5]));
                        s_748 = ((lane_in_warp == 5) ? total_n_753 : s_748);
                        float total_n_754;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_754) : "f"(sums_3[6]), "f"(errs_1[6]));
                        s_748 = ((lane_in_warp == 6) ? total_n_754 : s_748);
                        float total_n_755;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_755) : "f"(sums_3[7]), "f"(errs_1[7]));
                        s_748 = ((lane_in_warp == 7) ? total_n_755 : s_748);
                        float total_n_756;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_756) : "f"(sums_3[8]), "f"(errs_1[8]));
                        s_748 = ((lane_in_warp == 8) ? total_n_756 : s_748);
                        float total_n_757;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_757) : "f"(sums_3[9]), "f"(errs_1[9]));
                        s_748 = ((lane_in_warp == 9) ? total_n_757 : s_748);
                        float total_n_758;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_758) : "f"(sums_3[10]), "f"(errs_1[10]));
                        s_748 = ((lane_in_warp == 10) ? total_n_758 : s_748);
                        float total_n_759;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_759) : "f"(sums_3[11]), "f"(errs_1[11]));
                        s_748 = ((lane_in_warp == 11) ? total_n_759 : s_748);
                        float total_n_760;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_760) : "f"(sums_3[12]), "f"(errs_1[12]));
                        s_748 = ((lane_in_warp == 12) ? total_n_760 : s_748);
                        float total_n_761;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_761) : "f"(sums_3[13]), "f"(errs_1[13]));
                        s_748 = ((lane_in_warp == 13) ? total_n_761 : s_748);
                        float total_n_762;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_762) : "f"(sums_3[14]), "f"(errs_1[14]));
                        s_748 = ((lane_in_warp == 14) ? total_n_762 : s_748);
                        float total_n_763;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_763) : "f"(sums_3[15]), "f"(errs_1[15]));
                        s_748 = ((lane_in_warp == 15) ? total_n_763 : s_748);
                        float total_n_764;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_764) : "f"(sums_3[16]), "f"(errs_1[16]));
                        s_748 = ((lane_in_warp == 16) ? total_n_764 : s_748);
                        float total_n_765;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_765) : "f"(sums_3[17]), "f"(errs_1[17]));
                        s_748 = ((lane_in_warp == 17) ? total_n_765 : s_748);
                        float total_n_766;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_766) : "f"(sums_3[18]), "f"(errs_1[18]));
                        s_748 = ((lane_in_warp == 18) ? total_n_766 : s_748);
                        float total_n_767;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_767) : "f"(sums_3[19]), "f"(errs_1[19]));
                        s_748 = ((lane_in_warp == 19) ? total_n_767 : s_748);
                        float total_n_768;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_768) : "f"(sums_3[20]), "f"(errs_1[20]));
                        s_748 = ((lane_in_warp == 20) ? total_n_768 : s_748);
                        float total_n_769;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_769) : "f"(sums_3[21]), "f"(errs_1[21]));
                        s_748 = ((lane_in_warp == 21) ? total_n_769 : s_748);
                        float total_n_770;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_770) : "f"(sums_3[22]), "f"(errs_1[22]));
                        s_748 = ((lane_in_warp == 22) ? total_n_770 : s_748);
                        float total_n_771;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_771) : "f"(sums_3[23]), "f"(errs_1[23]));
                        s_748 = ((lane_in_warp == 23) ? total_n_771 : s_748);
                        float q_772;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(q_772) : "f"(sums_3[24]), "f"(errs_1[24]));
                        s_0 = s_748;
                        q_1 = q_772;
                    } else {
                        unsigned int row_15 = token % 64;
                        unsigned int first_task_4 = mb_1 * num_splits;
                        float sums_4[25];
                        unsigned int split_4 = lane_in_warp * 2;
                        unsigned int safe_split_4 = ((split_4 < num_splits) ? split_4 : 0);
                        unsigned long long record_4 = ((unsigned long long)first_task_4 + (unsigned long long)safe_split_4) * 1536 + (unsigned long long)(row_15 * 24);
                        float _vec_load_36[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_4 + 0);
                            _vec_load_36[0 + 0] = _v4.x;
                            _vec_load_36[0 + 1] = _v4.y;
                            _vec_load_36[0 + 2] = _v4.z;
                            _vec_load_36[0 + 3] = _v4.w;
                        }
                        float value_51 = ((split_4 < num_splits) ? _vec_load_36[0] : 0.0f);
                        sums_4[0] = value_51;
                        float value_0_4 = ((split_4 < num_splits) ? _vec_load_36[1] : 0.0f);
                        sums_4[1] = value_0_4;
                        float value_1_5 = ((split_4 < num_splits) ? _vec_load_36[2] : 0.0f);
                        sums_4[2] = value_1_5;
                        float value_2_4 = ((split_4 < num_splits) ? _vec_load_36[3] : 0.0f);
                        sums_4[3] = value_2_4;
                        float _vec_load_37[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 4) + 0);
                            _vec_load_37[0 + 0] = _v4.x;
                            _vec_load_37[0 + 1] = _v4.y;
                            _vec_load_37[0 + 2] = _v4.z;
                            _vec_load_37[0 + 3] = _v4.w;
                        }
                        float value_3_4 = ((split_4 < num_splits) ? _vec_load_37[0] : 0.0f);
                        sums_4[4] = value_3_4;
                        float value_4_4 = ((split_4 < num_splits) ? _vec_load_37[1] : 0.0f);
                        sums_4[5] = value_4_4;
                        float value_5_4 = ((split_4 < num_splits) ? _vec_load_37[2] : 0.0f);
                        sums_4[6] = value_5_4;
                        float value_6_4 = ((split_4 < num_splits) ? _vec_load_37[3] : 0.0f);
                        sums_4[7] = value_6_4;
                        float _vec_load_38[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 8) + 0);
                            _vec_load_38[0 + 0] = _v4.x;
                            _vec_load_38[0 + 1] = _v4.y;
                            _vec_load_38[0 + 2] = _v4.z;
                            _vec_load_38[0 + 3] = _v4.w;
                        }
                        float value_7_4 = ((split_4 < num_splits) ? _vec_load_38[0] : 0.0f);
                        sums_4[8] = value_7_4;
                        float value_8_5 = ((split_4 < num_splits) ? _vec_load_38[1] : 0.0f);
                        sums_4[9] = value_8_5;
                        float value_9_4 = ((split_4 < num_splits) ? _vec_load_38[2] : 0.0f);
                        sums_4[10] = value_9_4;
                        float value_10_4 = ((split_4 < num_splits) ? _vec_load_38[3] : 0.0f);
                        sums_4[11] = value_10_4;
                        float _vec_load_39[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 12) + 0);
                            _vec_load_39[0 + 0] = _v4.x;
                            _vec_load_39[0 + 1] = _v4.y;
                            _vec_load_39[0 + 2] = _v4.z;
                            _vec_load_39[0 + 3] = _v4.w;
                        }
                        float value_11_3 = ((split_4 < num_splits) ? _vec_load_39[0] : 0.0f);
                        sums_4[12] = value_11_3;
                        float value_12_3 = ((split_4 < num_splits) ? _vec_load_39[1] : 0.0f);
                        sums_4[13] = value_12_3;
                        float value_13_2 = ((split_4 < num_splits) ? _vec_load_39[2] : 0.0f);
                        sums_4[14] = value_13_2;
                        float value_14_4 = ((split_4 < num_splits) ? _vec_load_39[3] : 0.0f);
                        sums_4[15] = value_14_4;
                        float _vec_load_40[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 16) + 0);
                            _vec_load_40[0 + 0] = _v4.x;
                            _vec_load_40[0 + 1] = _v4.y;
                            _vec_load_40[0 + 2] = _v4.z;
                            _vec_load_40[0 + 3] = _v4.w;
                        }
                        float value_15_3 = ((split_4 < num_splits) ? _vec_load_40[0] : 0.0f);
                        sums_4[16] = value_15_3;
                        float value_16_4 = ((split_4 < num_splits) ? _vec_load_40[1] : 0.0f);
                        sums_4[17] = value_16_4;
                        float value_17_4 = ((split_4 < num_splits) ? _vec_load_40[2] : 0.0f);
                        sums_4[18] = value_17_4;
                        float value_18_3 = ((split_4 < num_splits) ? _vec_load_40[3] : 0.0f);
                        sums_4[19] = value_18_3;
                        float _vec_load_41[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 20) + 0);
                            _vec_load_41[0 + 0] = _v4.x;
                            _vec_load_41[0 + 1] = _v4.y;
                            _vec_load_41[0 + 2] = _v4.z;
                            _vec_load_41[0 + 3] = _v4.w;
                        }
                        float value_19_3 = ((split_4 < num_splits) ? _vec_load_41[0] : 0.0f);
                        sums_4[20] = value_19_3;
                        float value_20_4 = ((split_4 < num_splits) ? _vec_load_41[1] : 0.0f);
                        sums_4[21] = value_20_4;
                        float value_21_3 = ((split_4 < num_splits) ? _vec_load_41[2] : 0.0f);
                        sums_4[22] = value_21_3;
                        float value_22_3 = ((split_4 < num_splits) ? _vec_load_41[3] : 0.0f);
                        sums_4[23] = value_22_3;
                        float q_word_4 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_4 * 64 + (unsigned long long)(safe_split_4 * 64) + (unsigned long long)row_15];
                        float q_value_4 = ((split_4 < num_splits) ? q_word_4 : 0.0f);
                        sums_4[24] = q_value_4;
                        unsigned int split_23_2 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_24_2 = ((split_23_2 < num_splits) ? split_23_2 : 0);
                        unsigned long long record_25_2 = ((unsigned long long)first_task_4 + (unsigned long long)safe_split_24_2) * 1536 + (unsigned long long)(row_15 * 24);
                        float _vec_load_42[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25_2 + 0);
                            _vec_load_42[0 + 0] = _v4.x;
                            _vec_load_42[0 + 1] = _v4.y;
                            _vec_load_42[0 + 2] = _v4.z;
                            _vec_load_42[0 + 3] = _v4.w;
                        }
                        float value_26_3 = ((split_23_2 < num_splits) ? _vec_load_42[0] : 0.0f);
                        float _add_992;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_992) : "f"(sums_4[0]), "f"(value_26_3));
                        sums_4[0] = _add_992;
                        float value_27_1 = ((split_23_2 < num_splits) ? _vec_load_42[1] : 0.0f);
                        float _add_993;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_993) : "f"(sums_4[1]), "f"(value_27_1));
                        sums_4[1] = _add_993;
                        float value_28_3 = ((split_23_2 < num_splits) ? _vec_load_42[2] : 0.0f);
                        float _add_994;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_994) : "f"(sums_4[2]), "f"(value_28_3));
                        sums_4[2] = _add_994;
                        float value_29_1 = ((split_23_2 < num_splits) ? _vec_load_42[3] : 0.0f);
                        float _add_995;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_995) : "f"(sums_4[3]), "f"(value_29_1));
                        sums_4[3] = _add_995;
                        float _vec_load_43[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 4) + 0);
                            _vec_load_43[0 + 0] = _v4.x;
                            _vec_load_43[0 + 1] = _v4.y;
                            _vec_load_43[0 + 2] = _v4.z;
                            _vec_load_43[0 + 3] = _v4.w;
                        }
                        float value_30_1 = ((split_23_2 < num_splits) ? _vec_load_43[0] : 0.0f);
                        float _add_997;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_997) : "f"(sums_4[4]), "f"(value_30_1));
                        sums_4[4] = _add_997;
                        float value_31_1 = ((split_23_2 < num_splits) ? _vec_load_43[1] : 0.0f);
                        float _add_998;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_998) : "f"(sums_4[5]), "f"(value_31_1));
                        sums_4[5] = _add_998;
                        float value_32_3 = ((split_23_2 < num_splits) ? _vec_load_43[2] : 0.0f);
                        float _add_999;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_999) : "f"(sums_4[6]), "f"(value_32_3));
                        sums_4[6] = _add_999;
                        float value_33_1 = ((split_23_2 < num_splits) ? _vec_load_43[3] : 0.0f);
                        float _add_1000;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1000) : "f"(sums_4[7]), "f"(value_33_1));
                        sums_4[7] = _add_1000;
                        float _vec_load_44[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 8) + 0);
                            _vec_load_44[0 + 0] = _v4.x;
                            _vec_load_44[0 + 1] = _v4.y;
                            _vec_load_44[0 + 2] = _v4.z;
                            _vec_load_44[0 + 3] = _v4.w;
                        }
                        float value_34_1 = ((split_23_2 < num_splits) ? _vec_load_44[0] : 0.0f);
                        float _add_1002;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1002) : "f"(sums_4[8]), "f"(value_34_1));
                        sums_4[8] = _add_1002;
                        float value_35_1 = ((split_23_2 < num_splits) ? _vec_load_44[1] : 0.0f);
                        float _add_1003;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1003) : "f"(sums_4[9]), "f"(value_35_1));
                        sums_4[9] = _add_1003;
                        float value_36_3 = ((split_23_2 < num_splits) ? _vec_load_44[2] : 0.0f);
                        float _add_1004;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1004) : "f"(sums_4[10]), "f"(value_36_3));
                        sums_4[10] = _add_1004;
                        float value_37_1 = ((split_23_2 < num_splits) ? _vec_load_44[3] : 0.0f);
                        float _add_1005;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1005) : "f"(sums_4[11]), "f"(value_37_1));
                        sums_4[11] = _add_1005;
                        float _vec_load_45[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 12) + 0);
                            _vec_load_45[0 + 0] = _v4.x;
                            _vec_load_45[0 + 1] = _v4.y;
                            _vec_load_45[0 + 2] = _v4.z;
                            _vec_load_45[0 + 3] = _v4.w;
                        }
                        float value_38_1 = ((split_23_2 < num_splits) ? _vec_load_45[0] : 0.0f);
                        float _add_1007;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1007) : "f"(sums_4[12]), "f"(value_38_1));
                        sums_4[12] = _add_1007;
                        float value_39_1 = ((split_23_2 < num_splits) ? _vec_load_45[1] : 0.0f);
                        float _add_1008;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1008) : "f"(sums_4[13]), "f"(value_39_1));
                        sums_4[13] = _add_1008;
                        float value_40_3 = ((split_23_2 < num_splits) ? _vec_load_45[2] : 0.0f);
                        float _add_1009;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1009) : "f"(sums_4[14]), "f"(value_40_3));
                        sums_4[14] = _add_1009;
                        float value_41_1 = ((split_23_2 < num_splits) ? _vec_load_45[3] : 0.0f);
                        float _add_1010;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1010) : "f"(sums_4[15]), "f"(value_41_1));
                        sums_4[15] = _add_1010;
                        float _vec_load_46[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 16) + 0);
                            _vec_load_46[0 + 0] = _v4.x;
                            _vec_load_46[0 + 1] = _v4.y;
                            _vec_load_46[0 + 2] = _v4.z;
                            _vec_load_46[0 + 3] = _v4.w;
                        }
                        float value_42_1 = ((split_23_2 < num_splits) ? _vec_load_46[0] : 0.0f);
                        float _add_1012;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1012) : "f"(sums_4[16]), "f"(value_42_1));
                        sums_4[16] = _add_1012;
                        float value_43_1 = ((split_23_2 < num_splits) ? _vec_load_46[1] : 0.0f);
                        float _add_1013;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1013) : "f"(sums_4[17]), "f"(value_43_1));
                        sums_4[17] = _add_1013;
                        float value_44_4 = ((split_23_2 < num_splits) ? _vec_load_46[2] : 0.0f);
                        float _add_1014;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1014) : "f"(sums_4[18]), "f"(value_44_4));
                        sums_4[18] = _add_1014;
                        float value_45_1 = ((split_23_2 < num_splits) ? _vec_load_46[3] : 0.0f);
                        float _add_1015;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1015) : "f"(sums_4[19]), "f"(value_45_1));
                        sums_4[19] = _add_1015;
                        float _vec_load_47[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 20) + 0);
                            _vec_load_47[0 + 0] = _v4.x;
                            _vec_load_47[0 + 1] = _v4.y;
                            _vec_load_47[0 + 2] = _v4.z;
                            _vec_load_47[0 + 3] = _v4.w;
                        }
                        float value_46_1 = ((split_23_2 < num_splits) ? _vec_load_47[0] : 0.0f);
                        float _add_1017;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1017) : "f"(sums_4[20]), "f"(value_46_1));
                        sums_4[20] = _add_1017;
                        float value_47_1 = ((split_23_2 < num_splits) ? _vec_load_47[1] : 0.0f);
                        float _add_1018;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1018) : "f"(sums_4[21]), "f"(value_47_1));
                        sums_4[21] = _add_1018;
                        float value_48_3 = ((split_23_2 < num_splits) ? _vec_load_47[2] : 0.0f);
                        float _add_1019;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1019) : "f"(sums_4[22]), "f"(value_48_3));
                        sums_4[22] = _add_1019;
                        float value_49_1 = ((split_23_2 < num_splits) ? _vec_load_47[3] : 0.0f);
                        float _add_1020;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1020) : "f"(sums_4[23]), "f"(value_49_1));
                        sums_4[23] = _add_1020;
                        float q_word_50_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_4 * 64 + (unsigned long long)(safe_split_24_2 * 64) + (unsigned long long)row_15];
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
                        float s_52 = 0.0f;
                        s_52 = ((lane_in_warp == 0) ? sums_4[0] : s_52);
                        s_52 = ((lane_in_warp == 1) ? sums_4[1] : s_52);
                        s_52 = ((lane_in_warp == 2) ? sums_4[2] : s_52);
                        s_52 = ((lane_in_warp == 3) ? sums_4[3] : s_52);
                        s_52 = ((lane_in_warp == 4) ? sums_4[4] : s_52);
                        s_52 = ((lane_in_warp == 5) ? sums_4[5] : s_52);
                        s_52 = ((lane_in_warp == 6) ? sums_4[6] : s_52);
                        s_52 = ((lane_in_warp == 7) ? sums_4[7] : s_52);
                        s_52 = ((lane_in_warp == 8) ? sums_4[8] : s_52);
                        s_52 = ((lane_in_warp == 9) ? sums_4[9] : s_52);
                        s_52 = ((lane_in_warp == 10) ? sums_4[10] : s_52);
                        s_52 = ((lane_in_warp == 11) ? sums_4[11] : s_52);
                        s_52 = ((lane_in_warp == 12) ? sums_4[12] : s_52);
                        s_52 = ((lane_in_warp == 13) ? sums_4[13] : s_52);
                        s_52 = ((lane_in_warp == 14) ? sums_4[14] : s_52);
                        s_52 = ((lane_in_warp == 15) ? sums_4[15] : s_52);
                        s_52 = ((lane_in_warp == 16) ? sums_4[16] : s_52);
                        s_52 = ((lane_in_warp == 17) ? sums_4[17] : s_52);
                        s_52 = ((lane_in_warp == 18) ? sums_4[18] : s_52);
                        s_52 = ((lane_in_warp == 19) ? sums_4[19] : s_52);
                        s_52 = ((lane_in_warp == 20) ? sums_4[20] : s_52);
                        s_52 = ((lane_in_warp == 21) ? sums_4[21] : s_52);
                        s_52 = ((lane_in_warp == 22) ? sums_4[22] : s_52);
                        s_52 = ((lane_in_warp == 23) ? sums_4[23] : s_52);
                        float q_53 = sums_4[24];
                        s_0 = s_52;
                        q_1 = q_53;
#endif
                    }
#if __CUDA_ARCH__ == 1070
                } else if (lane_in_warp >= 4) {
                    new_post_mix[(unsigned long long)token * 4 + (unsigned long long)(lane_in_warp - 4)] = _mul_0;
#else
                    s = s_0;
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
                    float _add_48_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48_1) : "f"(1.0f), "f"(_exp_0));
                    float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_48_1);
                    float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_48_1);
                    float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
                    float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
                    float _add_49_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49_1) : "f"(sig), "f"(hc_pre_eps));
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
                unsigned int mb_2 = token_1 / 64;
                float sc_1 = ((lane_in_warp < 4) ? mix_scales[0] : ((lane_in_warp < 8) ? mix_scales[1] : mix_scales[2]));
                float bs_1 = mix_bases[((lane_in_warp < 24) ? lane_in_warp : 23)];
                unsigned long long target_1 = (launch_epochs[bid] + 1) * 128 + (unsigned long long)num_splits;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_2) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target_1)) >= static_cast<unsigned long long>(1));
                }
#if __CUDA_ARCH__ == 1070
                unsigned int row_12 = token_1 % 64;
                unsigned int first_task_2 = mb_2 * num_splits;
                float sums_2[5];
                unsigned int split_2 = lane_in_warp * 2;
                unsigned int safe_split_2 = ((split_2 < num_splits) ? split_2 : 0);
                unsigned long long record_2 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_2) * 1536 + (unsigned long long)(row_12 * 24);
                float _vec_load_24[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_2 + 0);
                    _vec_load_24[0 + 0] = _v4.x;
                    _vec_load_24[0 + 1] = _v4.y;
                    _vec_load_24[0 + 2] = _v4.z;
                    _vec_load_24[0 + 3] = _v4.w;
#else
                float s_2 = 0.0f;
                float q_3 = 0.0f;
                if (num_splits > 22) {
                    unsigned int row_16 = token_1 % 64;
                    unsigned int first_task_5 = mb_2 * num_splits;
                    float sums_5[5];
                    float errs_2[5];
                    unsigned int split_5 = lane_in_warp * 2;
                    unsigned int safe_split_5 = ((split_5 < num_splits) ? split_5 : 0);
                    unsigned long long record_5 = ((unsigned long long)first_task_5 + (unsigned long long)safe_split_5) * 1536 + (unsigned long long)(row_16 * 24);
                    float _vec_load_48[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5 + 0);
                        _vec_load_48[0 + 0] = _v4.x;
                        _vec_load_48[0 + 1] = _v4.y;
                        _vec_load_48[0 + 2] = _v4.z;
                        _vec_load_48[0 + 3] = _v4.w;
                    }
                    float value_54 = ((split_5 < num_splits) ? _vec_load_48[0] : 0.0f);
                    sums_5[0] = value_54;
                    float value_0_5 = ((split_5 < num_splits) ? _vec_load_48[1] : 0.0f);
                    sums_5[1] = value_0_5;
                    float value_1_6 = ((split_5 < num_splits) ? _vec_load_48[2] : 0.0f);
                    sums_5[2] = value_1_6;
                    float value_2_5 = ((split_5 < num_splits) ? _vec_load_48[3] : 0.0f);
                    sums_5[3] = value_2_5;
                    float q_word_5 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_5 * 64 + (unsigned long long)(safe_split_5 * 64) + (unsigned long long)row_16];
                    float q_value_5 = ((split_5 < num_splits) ? q_word_5 : 0.0f);
                    sums_5[4] = q_value_5;
                    unsigned int split_3_1 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_4_1 = ((split_3_1 < num_splits) ? split_3_1 : 0);
                    unsigned long long record_5_1 = ((unsigned long long)first_task_5 + (unsigned long long)safe_split_4_1) * 1536 + (unsigned long long)(row_16 * 24);
                    float _vec_load_49[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5_1 + 0);
                        _vec_load_49[0 + 0] = _v4.x;
                        _vec_load_49[0 + 1] = _v4.y;
                        _vec_load_49[0 + 2] = _v4.z;
                        _vec_load_49[0 + 3] = _v4.w;
                    }
                    float value_6_5 = ((split_3_1 < num_splits) ? _vec_load_49[0] : 0.0f);
                    float s_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(sums_5[0]), "f"(value_6_5));
                    float bb_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_2) : "f"(s_7), "f"(sums_5[0]));
                    float _sub_7_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7_3) : "f"(s_7), "f"(bb_2));
                    float _sub_6_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_4) : "f"(sums_5[0]), "f"(_sub_7_3));
                    float _sub_8_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8_3) : "f"(value_6_5), "f"(bb_2));
                    float e_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_6_4), "f"(_sub_8_3));
                    sums_5[0] = s_7;
                    errs_2[0] = e_2;
                    float value_8_6 = ((split_3_1 < num_splits) ? _vec_load_49[1] : 0.0f);
                    float s_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(sums_5[1]), "f"(value_8_6));
                    float bb_10;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_10) : "f"(s_9), "f"(sums_5[1]));
                    float _sub_10_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10_3) : "f"(s_9), "f"(bb_10));
                    float _sub_9_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9_3) : "f"(sums_5[1]), "f"(_sub_10_3));
                    float _sub_11_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11_3) : "f"(value_8_6), "f"(bb_10));
                    float e_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_9_3), "f"(_sub_11_3));
                    sums_5[1] = s_9;
                    errs_2[1] = e_11;
                    float value_12_4 = ((split_3_1 < num_splits) ? _vec_load_49[2] : 0.0f);
                    float s_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(sums_5[2]), "f"(value_12_4));
                    float bb_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_14) : "f"(s_13), "f"(sums_5[2]));
                    float _sub_13_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_4) : "f"(s_13), "f"(bb_14));
                    float _sub_12_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_4) : "f"(sums_5[2]), "f"(_sub_13_4));
                    float _sub_14_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14_3) : "f"(value_12_4), "f"(bb_14));
                    float e_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_12_4), "f"(_sub_14_3));
                    sums_5[2] = s_13;
                    errs_2[2] = e_15;
                    float value_16_5 = ((split_3_1 < num_splits) ? _vec_load_49[3] : 0.0f);
                    float s_17_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17_1) : "f"(sums_5[3]), "f"(value_16_5));
                    float bb_18_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_18_1) : "f"(s_17_1), "f"(sums_5[3]));
                    float _sub_16_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16_2) : "f"(s_17_1), "f"(bb_18_1));
                    float _sub_15_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15_3) : "f"(sums_5[3]), "f"(_sub_16_2));
                    float _sub_17_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17_3) : "f"(value_16_5), "f"(bb_18_1));
                    float e_19_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19_1) : "f"(_sub_15_3), "f"(_sub_17_3));
                    sums_5[3] = s_17_1;
                    errs_2[3] = e_19_1;
                    float q_word_20 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_5 * 64 + (unsigned long long)(safe_split_4_1 * 64) + (unsigned long long)row_16];
                    float q_value_21 = ((split_3_1 < num_splits) ? q_word_20 : 0.0f);
                    float s_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(sums_5[4]), "f"(q_value_21));
                    float bb_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_23) : "f"(s_22), "f"(sums_5[4]));
                    float _sub_19_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_4) : "f"(s_22), "f"(bb_23));
                    float _sub_18_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18_3) : "f"(sums_5[4]), "f"(_sub_19_4));
                    float _sub_20_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_4) : "f"(q_value_21), "f"(bb_23));
                    float e_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_24) : "f"(_sub_18_3), "f"(_sub_20_4));
                    sums_5[4] = s_22;
                    errs_2[4] = e_24;
                    float partner_2 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 1);
                    float partner_err_1 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 1);
                    float s_25_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25_1) : "f"(sums_5[0]), "f"(partner_2));
                    float bb_26_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_26_1) : "f"(s_25_1), "f"(sums_5[0]));
                    float _sub_22_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_3) : "f"(s_25_1), "f"(bb_26_1));
                    float _sub_21_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21_3) : "f"(sums_5[0]), "f"(_sub_22_3));
                    float _sub_23_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_3) : "f"(partner_2), "f"(bb_26_1));
                    float e_27_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_27_1) : "f"(_sub_21_3), "f"(_sub_23_3));
                    sums_5[0] = s_25_1;
                    float _add_25_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_2) : "f"(errs_2[0]), "f"(partner_err_1));
                    float _add_24_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24_1) : "f"(_add_25_2), "f"(e_27_1));
                    errs_2[0] = _add_24_1;
                    float partner_28 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 1);
                    float partner_err_29 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 1);
                    float s_30;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(sums_5[1]), "f"(partner_28));
                    float bb_31;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_31) : "f"(s_30), "f"(sums_5[1]));
                    float _sub_27_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_4) : "f"(s_30), "f"(bb_31));
                    float _sub_26_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_4) : "f"(sums_5[1]), "f"(_sub_27_4));
                    float _sub_28_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_3) : "f"(partner_28), "f"(bb_31));
                    float e_32;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_32) : "f"(_sub_26_4), "f"(_sub_28_3));
                    sums_5[1] = s_30;
                    float _add_30_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30_1) : "f"(errs_2[1]), "f"(partner_err_29));
                    float _add_29_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_1) : "f"(_add_30_1), "f"(e_32));
                    errs_2[1] = _add_29_1;
                    float partner_33 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 1);
                    float partner_err_34 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 1);
                    float s_35;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(sums_5[2]), "f"(partner_33));
                    float bb_36;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_36) : "f"(s_35), "f"(sums_5[2]));
                    float _sub_32_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32_3) : "f"(s_35), "f"(bb_36));
                    float _sub_31_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_3) : "f"(sums_5[2]), "f"(_sub_32_3));
                    float _sub_33_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_4) : "f"(partner_33), "f"(bb_36));
                    float e_37;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_37) : "f"(_sub_31_3), "f"(_sub_33_4));
                    sums_5[2] = s_35;
                    float _add_35_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35_1) : "f"(errs_2[2]), "f"(partner_err_34));
                    float _add_34_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34_1) : "f"(_add_35_1), "f"(e_37));
                    errs_2[2] = _add_34_1;
                    float partner_38 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 1);
                    float partner_err_39 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 1);
                    float s_40;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(sums_5[3]), "f"(partner_38));
                    float bb_41;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_41) : "f"(s_40), "f"(sums_5[3]));
                    float _sub_37_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37_3) : "f"(s_40), "f"(bb_41));
                    float _sub_36_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36_3) : "f"(sums_5[3]), "f"(_sub_37_3));
                    float _sub_38_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38_3) : "f"(partner_38), "f"(bb_41));
                    float e_42;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_42) : "f"(_sub_36_3), "f"(_sub_38_3));
                    sums_5[3] = s_40;
                    float _add_40_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40_1) : "f"(errs_2[3]), "f"(partner_err_39));
                    float _add_39_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_2) : "f"(_add_40_1), "f"(e_42));
                    errs_2[3] = _add_39_2;
                    float partner_43 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 1);
                    float partner_err_44 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 1);
                    float s_45_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45_2) : "f"(sums_5[4]), "f"(partner_43));
                    float bb_46_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_46_2) : "f"(s_45_2), "f"(sums_5[4]));
                    float _sub_42_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42_3) : "f"(s_45_2), "f"(bb_46_2));
                    float _sub_41_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_4) : "f"(sums_5[4]), "f"(_sub_42_3));
                    float _sub_43_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43_3) : "f"(partner_43), "f"(bb_46_2));
                    float e_47_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_47_2) : "f"(_sub_41_4), "f"(_sub_43_3));
                    sums_5[4] = s_45_2;
                    float _add_45_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45_1) : "f"(errs_2[4]), "f"(partner_err_44));
                    float _add_44_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44_1) : "f"(_add_45_1), "f"(e_47_2));
                    errs_2[4] = _add_44_1;
                    float partner_48 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 2);
                    float partner_err_49 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 2);
                    float s_50;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(sums_5[0]), "f"(partner_48));
                    float bb_51;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_51) : "f"(s_50), "f"(sums_5[0]));
                    float _sub_47_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_3) : "f"(s_50), "f"(bb_51));
                    float _sub_46_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46_2) : "f"(sums_5[0]), "f"(_sub_47_3));
                    float _sub_48_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48_2) : "f"(partner_48), "f"(bb_51));
                    float e_52;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_52) : "f"(_sub_46_2), "f"(_sub_48_2));
                    sums_5[0] = s_50;
                    float _add_50;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(errs_2[0]), "f"(partner_err_49));
                    float _add_49_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49_2) : "f"(_add_50), "f"(e_52));
                    errs_2[0] = _add_49_2;
                    float partner_53 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 2);
                    float partner_err_54 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 2);
                    float s_55;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(sums_5[1]), "f"(partner_53));
                    float bb_56;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_56) : "f"(s_55), "f"(sums_5[1]));
                    float _sub_52_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52_2) : "f"(s_55), "f"(bb_56));
                    float _sub_51_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51_2) : "f"(sums_5[1]), "f"(_sub_52_2));
                    float _sub_53_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53_1) : "f"(partner_53), "f"(bb_56));
                    float e_57;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_57) : "f"(_sub_51_2), "f"(_sub_53_1));
                    sums_5[1] = s_55;
                    float _add_55;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(errs_2[1]), "f"(partner_err_54));
                    float _add_54_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54_1) : "f"(_add_55), "f"(e_57));
                    errs_2[1] = _add_54_1;
                    float partner_58 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 2);
                    float partner_err_59 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 2);
                    float s_60;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(sums_5[2]), "f"(partner_58));
                    float bb_61;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_61) : "f"(s_60), "f"(sums_5[2]));
                    float _sub_57_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57_2) : "f"(s_60), "f"(bb_61));
                    float _sub_56_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56_2) : "f"(sums_5[2]), "f"(_sub_57_2));
                    float _sub_58_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58_1) : "f"(partner_58), "f"(bb_61));
                    float e_62;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_62) : "f"(_sub_56_2), "f"(_sub_58_1));
                    sums_5[2] = s_60;
                    float _add_60;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(errs_2[2]), "f"(partner_err_59));
                    float _add_59_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59_1) : "f"(_add_60), "f"(e_62));
                    errs_2[2] = _add_59_1;
                    float partner_63 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 2);
                    float partner_err_64 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 2);
                    float s_65_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65_2) : "f"(sums_5[3]), "f"(partner_63));
                    float bb_66_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_66_2) : "f"(s_65_2), "f"(sums_5[3]));
                    float _sub_62_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62_2) : "f"(s_65_2), "f"(bb_66_2));
                    float _sub_61_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61_2) : "f"(sums_5[3]), "f"(_sub_62_2));
                    float _sub_63_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63_1) : "f"(partner_63), "f"(bb_66_2));
                    float e_67_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_67_2) : "f"(_sub_61_2), "f"(_sub_63_1));
                    sums_5[3] = s_65_2;
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(errs_2[3]), "f"(partner_err_64));
                    float _add_64_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64_1) : "f"(_add_65), "f"(e_67_2));
                    errs_2[3] = _add_64_1;
                    float partner_68 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 2);
                    float partner_err_69 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 2);
                    float s_70;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_70) : "f"(sums_5[4]), "f"(partner_68));
                    float bb_71;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_71) : "f"(s_70), "f"(sums_5[4]));
                    float _sub_67_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67_2) : "f"(s_70), "f"(bb_71));
                    float _sub_66_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66_2) : "f"(sums_5[4]), "f"(_sub_67_2));
                    float _sub_68_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68_1) : "f"(partner_68), "f"(bb_71));
                    float e_72;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_72) : "f"(_sub_66_2), "f"(_sub_68_1));
                    sums_5[4] = s_70;
                    float _add_70;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(errs_2[4]), "f"(partner_err_69));
                    float _add_69_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69_1) : "f"(_add_70), "f"(e_72));
                    errs_2[4] = _add_69_1;
                    float partner_73 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 4);
                    float partner_err_74 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 4);
                    float s_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_75) : "f"(sums_5[0]), "f"(partner_73));
                    float bb_76;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_76) : "f"(s_75), "f"(sums_5[0]));
                    float _sub_72_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72_1) : "f"(s_75), "f"(bb_76));
                    float _sub_71_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71_2) : "f"(sums_5[0]), "f"(_sub_72_1));
                    float _sub_73;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(partner_73), "f"(bb_76));
                    float e_77;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_77) : "f"(_sub_71_2), "f"(_sub_73));
                    sums_5[0] = s_75;
                    float _add_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(errs_2[0]), "f"(partner_err_74));
                    float _add_74_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74_1) : "f"(_add_75), "f"(e_77));
                    errs_2[0] = _add_74_1;
                    float partner_78 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 4);
                    float partner_err_79 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 4);
                    float s_80;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_80) : "f"(sums_5[1]), "f"(partner_78));
                    float bb_81;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_81) : "f"(s_80), "f"(sums_5[1]));
                    float _sub_77_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77_1) : "f"(s_80), "f"(bb_81));
                    float _sub_76_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76_1) : "f"(sums_5[1]), "f"(_sub_77_1));
                    float _sub_78;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(partner_78), "f"(bb_81));
                    float e_82;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_82) : "f"(_sub_76_1), "f"(_sub_78));
                    sums_5[1] = s_80;
                    float _add_80;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(errs_2[1]), "f"(partner_err_79));
                    float _add_79_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79_1) : "f"(_add_80), "f"(e_82));
                    errs_2[1] = _add_79_1;
                    float partner_83 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 4);
                    float partner_err_84 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 4);
                    float s_85_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_85_1) : "f"(sums_5[2]), "f"(partner_83));
                    float bb_86_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_86_1) : "f"(s_85_1), "f"(sums_5[2]));
                    float _sub_82_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82_1) : "f"(s_85_1), "f"(bb_86_1));
                    float _sub_81_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81_1) : "f"(sums_5[2]), "f"(_sub_82_1));
                    float _sub_83;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(partner_83), "f"(bb_86_1));
                    float e_87_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_87_1) : "f"(_sub_81_1), "f"(_sub_83));
                    sums_5[2] = s_85_1;
                    float _add_85;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(errs_2[2]), "f"(partner_err_84));
                    float _add_84_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84_1) : "f"(_add_85), "f"(e_87_1));
                    errs_2[2] = _add_84_1;
                    float partner_88 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 4);
                    float partner_err_89 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 4);
                    float s_90;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_90) : "f"(sums_5[3]), "f"(partner_88));
                    float bb_91;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_91) : "f"(s_90), "f"(sums_5[3]));
                    float _sub_87_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87_1) : "f"(s_90), "f"(bb_91));
                    float _sub_86_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86_1) : "f"(sums_5[3]), "f"(_sub_87_1));
                    float _sub_88;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(partner_88), "f"(bb_91));
                    float e_92;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_92) : "f"(_sub_86_1), "f"(_sub_88));
                    sums_5[3] = s_90;
                    float _add_90;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(errs_2[3]), "f"(partner_err_89));
                    float _add_89_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89_1) : "f"(_add_90), "f"(e_92));
                    errs_2[3] = _add_89_1;
                    float partner_93 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 4);
                    float partner_err_94 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 4);
                    float s_95;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_95) : "f"(sums_5[4]), "f"(partner_93));
                    float bb_96;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_96) : "f"(s_95), "f"(sums_5[4]));
                    float _sub_92_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92_1) : "f"(s_95), "f"(bb_96));
                    float _sub_91_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91_1) : "f"(sums_5[4]), "f"(_sub_92_1));
                    float _sub_93;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(partner_93), "f"(bb_96));
                    float e_97;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_97) : "f"(_sub_91_1), "f"(_sub_93));
                    sums_5[4] = s_95;
                    float _add_95;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(errs_2[4]), "f"(partner_err_94));
                    float _add_94_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94_1) : "f"(_add_95), "f"(e_97));
                    errs_2[4] = _add_94_1;
                    float partner_98 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 8);
                    float partner_err_99 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 8);
                    float s_100;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_100) : "f"(sums_5[0]), "f"(partner_98));
                    float bb_101;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_101) : "f"(s_100), "f"(sums_5[0]));
                    float _sub_97_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97_1) : "f"(s_100), "f"(bb_101));
                    float _sub_96_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96_1) : "f"(sums_5[0]), "f"(_sub_97_1));
                    float _sub_98;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_98) : "f"(partner_98), "f"(bb_101));
                    float e_102;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_102) : "f"(_sub_96_1), "f"(_sub_98));
                    sums_5[0] = s_100;
                    float _add_100;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(errs_2[0]), "f"(partner_err_99));
                    float _add_99_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99_1) : "f"(_add_100), "f"(e_102));
                    errs_2[0] = _add_99_1;
                    float partner_103 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 8);
                    float partner_err_104 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 8);
                    float s_105_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_105_1) : "f"(sums_5[1]), "f"(partner_103));
                    float bb_106_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_106_1) : "f"(s_105_1), "f"(sums_5[1]));
                    float _sub_102_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102_1) : "f"(s_105_1), "f"(bb_106_1));
                    float _sub_101_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101_1) : "f"(sums_5[1]), "f"(_sub_102_1));
                    float _sub_103;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(partner_103), "f"(bb_106_1));
                    float e_107_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_107_1) : "f"(_sub_101_1), "f"(_sub_103));
                    sums_5[1] = s_105_1;
                    float _add_105;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(errs_2[1]), "f"(partner_err_104));
                    float _add_104_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104_1) : "f"(_add_105), "f"(e_107_1));
                    errs_2[1] = _add_104_1;
                    float partner_108 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 8);
                    float partner_err_109 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 8);
                    float s_110_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_110_1) : "f"(sums_5[2]), "f"(partner_108));
                    float bb_111_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_111_1) : "f"(s_110_1), "f"(sums_5[2]));
                    float _sub_107_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_107_1) : "f"(s_110_1), "f"(bb_111_1));
                    float _sub_106_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106_1) : "f"(sums_5[2]), "f"(_sub_107_1));
                    float _sub_108;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_108) : "f"(partner_108), "f"(bb_111_1));
                    float e_112_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_112_1) : "f"(_sub_106_1), "f"(_sub_108));
                    sums_5[2] = s_110_1;
                    float _add_110;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(errs_2[2]), "f"(partner_err_109));
                    float _add_109_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109_1) : "f"(_add_110), "f"(e_112_1));
                    errs_2[2] = _add_109_1;
                    float partner_113 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 8);
                    float partner_err_114 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 8);
                    float s_115;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_115) : "f"(sums_5[3]), "f"(partner_113));
                    float bb_116;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_116) : "f"(s_115), "f"(sums_5[3]));
                    float _sub_112_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112_1) : "f"(s_115), "f"(bb_116));
                    float _sub_111_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111_1) : "f"(sums_5[3]), "f"(_sub_112_1));
                    float _sub_113;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(partner_113), "f"(bb_116));
                    float e_117;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_117) : "f"(_sub_111_1), "f"(_sub_113));
                    sums_5[3] = s_115;
                    float _add_115;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(errs_2[3]), "f"(partner_err_114));
                    float _add_114_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114_1) : "f"(_add_115), "f"(e_117));
                    errs_2[3] = _add_114_1;
                    float partner_118 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 8);
                    float partner_err_119 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 8);
                    float s_120;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_120) : "f"(sums_5[4]), "f"(partner_118));
                    float bb_121;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_121) : "f"(s_120), "f"(sums_5[4]));
                    float _sub_117_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_117_1) : "f"(s_120), "f"(bb_121));
                    float _sub_116_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_116_1) : "f"(sums_5[4]), "f"(_sub_117_1));
                    float _sub_118;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(partner_118), "f"(bb_121));
                    float e_122;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_122) : "f"(_sub_116_1), "f"(_sub_118));
                    sums_5[4] = s_120;
                    float _add_120;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(errs_2[4]), "f"(partner_err_119));
                    float _add_119_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119_1) : "f"(_add_120), "f"(e_122));
                    errs_2[4] = _add_119_1;
                    float partner_123 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 16);
                    float partner_err_124 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 16);
                    float s_125_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_125_1) : "f"(sums_5[0]), "f"(partner_123));
                    float bb_126_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_126_1) : "f"(s_125_1), "f"(sums_5[0]));
                    float _sub_122_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_122_1) : "f"(s_125_1), "f"(bb_126_1));
                    float _sub_121_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121_1) : "f"(sums_5[0]), "f"(_sub_122_1));
                    float _sub_123;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_123) : "f"(partner_123), "f"(bb_126_1));
                    float e_127_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_127_1) : "f"(_sub_121_1), "f"(_sub_123));
                    sums_5[0] = s_125_1;
                    float _add_125;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(errs_2[0]), "f"(partner_err_124));
                    float _add_124;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(_add_125), "f"(e_127_1));
                    errs_2[0] = _add_124;
                    float partner_128_1 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 16);
                    float partner_err_129_1 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 16);
                    float s_130_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_130_1) : "f"(sums_5[1]), "f"(partner_128_1));
                    float bb_131_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_131_1) : "f"(s_130_1), "f"(sums_5[1]));
                    float _sub_127;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_127) : "f"(s_130_1), "f"(bb_131_1));
                    float _sub_126;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_126) : "f"(sums_5[1]), "f"(_sub_127));
                    float _sub_128;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_128) : "f"(partner_128_1), "f"(bb_131_1));
                    float e_132_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_132_1) : "f"(_sub_126), "f"(_sub_128));
                    sums_5[1] = s_130_1;
                    float _add_130_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130_1) : "f"(errs_2[1]), "f"(partner_err_129_1));
                    float _add_129_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129_1) : "f"(_add_130_1), "f"(e_132_1));
                    errs_2[1] = _add_129_1;
                    float partner_133_1 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 16);
                    float partner_err_134_1 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 16);
                    float s_135_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_135_1) : "f"(sums_5[2]), "f"(partner_133_1));
                    float bb_136_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_136_1) : "f"(s_135_1), "f"(sums_5[2]));
                    float _sub_132;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_132) : "f"(s_135_1), "f"(bb_136_1));
                    float _sub_131;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(sums_5[2]), "f"(_sub_132));
                    float _sub_133;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_133) : "f"(partner_133_1), "f"(bb_136_1));
                    float e_137_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_137_1) : "f"(_sub_131), "f"(_sub_133));
                    sums_5[2] = s_135_1;
                    float _add_135_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135_1) : "f"(errs_2[2]), "f"(partner_err_134_1));
                    float _add_134_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134_1) : "f"(_add_135_1), "f"(e_137_1));
                    errs_2[2] = _add_134_1;
                    float partner_138_1 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 16);
                    float partner_err_139_1 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 16);
                    float s_140_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_140_1) : "f"(sums_5[3]), "f"(partner_138_1));
                    float bb_141_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_141_1) : "f"(s_140_1), "f"(sums_5[3]));
                    float _sub_137;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_137) : "f"(s_140_1), "f"(bb_141_1));
                    float _sub_136;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(sums_5[3]), "f"(_sub_137));
                    float _sub_138;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(partner_138_1), "f"(bb_141_1));
                    float e_142_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_142_1) : "f"(_sub_136), "f"(_sub_138));
                    sums_5[3] = s_140_1;
                    float _add_140_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140_1) : "f"(errs_2[3]), "f"(partner_err_139_1));
                    float _add_139_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139_1) : "f"(_add_140_1), "f"(e_142_1));
                    errs_2[3] = _add_139_1;
                    float partner_143_1 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 16);
                    float partner_err_144_1 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 16);
                    float s_145_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_145_1) : "f"(sums_5[4]), "f"(partner_143_1));
                    float bb_146_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_146_1) : "f"(s_145_1), "f"(sums_5[4]));
                    float _sub_142;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_142) : "f"(s_145_1), "f"(bb_146_1));
                    float _sub_141;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_141) : "f"(sums_5[4]), "f"(_sub_142));
                    float _sub_143;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_143) : "f"(partner_143_1), "f"(bb_146_1));
                    float e_147_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_147_1) : "f"(_sub_141), "f"(_sub_143));
                    sums_5[4] = s_145_1;
                    float _add_145_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145_1) : "f"(errs_2[4]), "f"(partner_err_144_1));
                    float _add_144;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_add_145_1), "f"(e_147_1));
                    errs_2[4] = _add_144;
                    float s_148 = 0.0f;
                    float total_n_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1) : "f"(sums_5[0]), "f"(errs_2[0]));
                    s_148 = ((lane_in_warp == 0) ? total_n_1 : s_148);
                    float total_n_149;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_149) : "f"(sums_5[1]), "f"(errs_2[1]));
                    s_148 = ((lane_in_warp == 1) ? total_n_149 : s_148);
                    float total_n_150;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_150) : "f"(sums_5[2]), "f"(errs_2[2]));
                    s_148 = ((lane_in_warp == 2) ? total_n_150 : s_148);
                    float total_n_151;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_151) : "f"(sums_5[3]), "f"(errs_2[3]));
                    s_148 = ((lane_in_warp == 3) ? total_n_151 : s_148);
                    float q_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(q_152) : "f"(sums_5[4]), "f"(errs_2[4]));
                    s_2 = s_148;
                    q_3 = q_152;
                } else {
                    unsigned int row_18 = token_1 % 64;
                    unsigned int first_task_6 = mb_2 * num_splits;
                    float sums_6[5];
                    unsigned int split_6 = lane_in_warp * 2;
                    unsigned int safe_split_6 = ((split_6 < num_splits) ? split_6 : 0);
                    unsigned long long record_6 = ((unsigned long long)first_task_6 + (unsigned long long)safe_split_6) * 1536 + (unsigned long long)(row_18 * 24);
                    float _vec_load_50[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_6 + 0);
                        _vec_load_50[0 + 0] = _v4.x;
                        _vec_load_50[0 + 1] = _v4.y;
                        _vec_load_50[0 + 2] = _v4.z;
                        _vec_load_50[0 + 3] = _v4.w;
                    }
                    float value_55 = ((split_6 < num_splits) ? _vec_load_50[0] : 0.0f);
                    sums_6[0] = value_55;
                    float value_0_6 = ((split_6 < num_splits) ? _vec_load_50[1] : 0.0f);
                    sums_6[1] = value_0_6;
                    float value_1_7 = ((split_6 < num_splits) ? _vec_load_50[2] : 0.0f);
                    sums_6[2] = value_1_7;
                    float value_2_6 = ((split_6 < num_splits) ? _vec_load_50[3] : 0.0f);
                    sums_6[3] = value_2_6;
                    float q_word_6 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_6 * 64 + (unsigned long long)(safe_split_6 * 64) + (unsigned long long)row_18];
                    float q_value_6 = ((split_6 < num_splits) ? q_word_6 : 0.0f);
                    sums_6[4] = q_value_6;
                    unsigned int split_3_2 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_4_2 = ((split_3_2 < num_splits) ? split_3_2 : 0);
                    unsigned long long record_5_2 = ((unsigned long long)first_task_6 + (unsigned long long)safe_split_4_2) * 1536 + (unsigned long long)(row_18 * 24);
                    float _vec_load_51[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5_2 + 0);
                        _vec_load_51[0 + 0] = _v4.x;
                        _vec_load_51[0 + 1] = _v4.y;
                        _vec_load_51[0 + 2] = _v4.z;
                        _vec_load_51[0 + 3] = _v4.w;
                    }
                    float value_6_6 = ((split_3_2 < num_splits) ? _vec_load_51[0] : 0.0f);
                    float _add_148;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(sums_6[0]), "f"(value_6_6));
                    sums_6[0] = _add_148;
                    float value_7_5 = ((split_3_2 < num_splits) ? _vec_load_51[1] : 0.0f);
                    float _add_149_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149_1) : "f"(sums_6[1]), "f"(value_7_5));
                    sums_6[1] = _add_149_1;
                    float value_8_7 = ((split_3_2 < num_splits) ? _vec_load_51[2] : 0.0f);
                    float _add_150;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(sums_6[2]), "f"(value_8_7));
                    sums_6[2] = _add_150;
                    float value_9_5 = ((split_3_2 < num_splits) ? _vec_load_51[3] : 0.0f);
                    float _add_151_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151_1) : "f"(sums_6[3]), "f"(value_9_5));
                    sums_6[3] = _add_151_1;
                    float q_word_10 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_6 * 64 + (unsigned long long)(safe_split_4_2 * 64) + (unsigned long long)row_18];
                    float q_value_11 = ((split_3_2 < num_splits) ? q_word_10 : 0.0f);
                    float _add_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(sums_6[4]), "f"(q_value_11));
                    sums_6[4] = _add_152;
                    float _shfl_xor_sync_154_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 1);
                    float _add_153_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_154_1));
                    sums_6[0] = _add_153_1;
                    float _shfl_xor_sync_156_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 1);
                    float _add_155_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155_1) : "f"(sums_6[1]), "f"(_shfl_xor_sync_156_1));
                    sums_6[1] = _add_155_1;
                    float _shfl_xor_sync_158_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 1);
                    float _add_157_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157_1) : "f"(sums_6[2]), "f"(_shfl_xor_sync_158_1));
                    sums_6[2] = _add_157_1;
                    float _shfl_xor_sync_160_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 1);
                    float _add_159_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_160_1));
                    sums_6[3] = _add_159_1;
                    float _shfl_xor_sync_162_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 1);
                    float _add_161_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_162_1));
                    sums_6[4] = _add_161_1;
                    float _shfl_xor_sync_164_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 2);
                    float _add_163_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_164_1));
                    sums_6[0] = _add_163_1;
                    float _shfl_xor_sync_166_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 2);
                    float _add_165_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165_1) : "f"(sums_6[1]), "f"(_shfl_xor_sync_166_1));
                    sums_6[1] = _add_165_1;
                    float _shfl_xor_sync_168_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 2);
                    float _add_167_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167_1) : "f"(sums_6[2]), "f"(_shfl_xor_sync_168_1));
                    sums_6[2] = _add_167_1;
                    float _shfl_xor_sync_170_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 2);
                    float _add_169_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_170_1));
                    sums_6[3] = _add_169_1;
                    float _shfl_xor_sync_172_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 2);
                    float _add_171_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_172_1));
                    sums_6[4] = _add_171_1;
                    float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 4);
                    float _add_173;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(sums_6[0]), "f"(_shfl_xor_sync_174));
                    sums_6[0] = _add_173;
                    float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 4);
                    float _add_175;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(sums_6[1]), "f"(_shfl_xor_sync_176));
                    sums_6[1] = _add_175;
                    float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 4);
                    float _add_177;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(sums_6[2]), "f"(_shfl_xor_sync_178));
                    sums_6[2] = _add_177;
                    float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 4);
                    float _add_179;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(sums_6[3]), "f"(_shfl_xor_sync_180));
                    sums_6[3] = _add_179;
                    float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 4);
                    float _add_181_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_182));
                    sums_6[4] = _add_181_1;
                    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 8);
                    float _add_183_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_184));
                    sums_6[0] = _add_183_1;
                    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 8);
                    float _add_185;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(sums_6[1]), "f"(_shfl_xor_sync_186));
                    sums_6[1] = _add_185;
                    float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 8);
                    float _add_187_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187_1) : "f"(sums_6[2]), "f"(_shfl_xor_sync_188));
                    sums_6[2] = _add_187_1;
                    float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 8);
                    float _add_189_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_190));
                    sums_6[3] = _add_189_1;
                    float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 8);
                    float _add_191_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_192));
                    sums_6[4] = _add_191_1;
                    float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 16);
                    float _add_193_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_194));
                    sums_6[0] = _add_193_1;
                    float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 16);
                    float _add_195;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(sums_6[1]), "f"(_shfl_xor_sync_196));
                    sums_6[1] = _add_195;
                    float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 16);
                    float _add_197_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197_1) : "f"(sums_6[2]), "f"(_shfl_xor_sync_198));
                    sums_6[2] = _add_197_1;
                    float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 16);
                    float _add_199_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_200));
                    sums_6[3] = _add_199_1;
                    float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 16);
                    float _add_201_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_202));
                    sums_6[4] = _add_201_1;
                    float s_12 = 0.0f;
                    s_12 = ((lane_in_warp == 0) ? sums_6[0] : s_12);
                    s_12 = ((lane_in_warp == 1) ? sums_6[1] : s_12);
                    s_12 = ((lane_in_warp == 2) ? sums_6[2] : s_12);
                    s_12 = ((lane_in_warp == 3) ? sums_6[3] : s_12);
                    float q_13 = sums_6[4];
                    s_2 = s_12;
                    q_3 = q_13;
#endif
                }
#if __CUDA_ARCH__ == 1070
                float value_24 = ((split_2 < num_splits) ? _vec_load_24[0] : 0.0f);
                sums_2[0] = value_24;
                float value_0_2 = ((split_2 < num_splits) ? _vec_load_24[1] : 0.0f);
                sums_2[1] = value_0_2;
                float value_1_3 = ((split_2 < num_splits) ? _vec_load_24[2] : 0.0f);
                sums_2[2] = value_1_3;
                float value_2_2 = ((split_2 < num_splits) ? _vec_load_24[3] : 0.0f);
                sums_2[3] = value_2_2;
                float q_word_2 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_2 * 64) + (unsigned long long)row_12];
                float q_value_2 = ((split_2 < num_splits) ? q_word_2 : 0.0f);
                sums_2[4] = q_value_2;
                unsigned int split_3 = lane_in_warp * 2 + 1;
                unsigned int safe_split_4 = ((split_3 < num_splits) ? split_3 : 0);
                unsigned long long record_5 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_4) * 1536 + (unsigned long long)(row_12 * 24);
                float _vec_load_25[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5 + 0);
                    _vec_load_25[0 + 0] = _v4.x;
                    _vec_load_25[0 + 1] = _v4.y;
                    _vec_load_25[0 + 2] = _v4.z;
                    _vec_load_25[0 + 3] = _v4.w;
                }
                float value_6_2 = ((split_3 < num_splits) ? _vec_load_25[0] : 0.0f);
                float _add_6_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6_1) : "f"(sums_2[0]), "f"(value_6_2));
                sums_2[0] = _add_6_1;
                float value_7_2 = ((split_3 < num_splits) ? _vec_load_25[1] : 0.0f);
                float _add_7_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7_2) : "f"(sums_2[1]), "f"(value_7_2));
                sums_2[1] = _add_7_2;
                float value_8_3 = ((split_3 < num_splits) ? _vec_load_25[2] : 0.0f);
                float _add_8_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8_2) : "f"(sums_2[2]), "f"(value_8_3));
                sums_2[2] = _add_8_2;
                float value_9_2 = ((split_3 < num_splits) ? _vec_load_25[3] : 0.0f);
                float _add_9_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9_2) : "f"(sums_2[3]), "f"(value_9_2));
                sums_2[3] = _add_9_2;
                float q_word_10 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_4 * 64) + (unsigned long long)row_12];
                float q_value_11 = ((split_3 < num_splits) ? q_word_10 : 0.0f);
                float _add_10_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10_2) : "f"(sums_2[4]), "f"(q_value_11));
                sums_2[4] = _add_10_2;
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 1);
                float _add_11_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11_2) : "f"(sums_2[0]), "f"(_shfl_xor_sync_12));
                sums_2[0] = _add_11_2;
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 1);
                float _add_13_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13_2) : "f"(sums_2[1]), "f"(_shfl_xor_sync_14));
                sums_2[1] = _add_13_2;
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 1);
                float _add_15_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15_2) : "f"(sums_2[2]), "f"(_shfl_xor_sync_16));
                sums_2[2] = _add_15_2;
                float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 1);
                float _add_17_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17_2) : "f"(sums_2[3]), "f"(_shfl_xor_sync_18));
                sums_2[3] = _add_17_2;
                float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 1);
                float _add_19_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19_2) : "f"(sums_2[4]), "f"(_shfl_xor_sync_20));
                sums_2[4] = _add_19_2;
                float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 2);
                float _add_21_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_22));
                sums_2[0] = _add_21_1;
                float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 2);
                float _add_23_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23_2) : "f"(sums_2[1]), "f"(_shfl_xor_sync_24));
                sums_2[1] = _add_23_2;
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 2);
                float _add_25_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_3) : "f"(sums_2[2]), "f"(_shfl_xor_sync_26));
                sums_2[2] = _add_25_3;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 2);
                float _add_27_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27_2) : "f"(sums_2[3]), "f"(_shfl_xor_sync_28));
                sums_2[3] = _add_27_2;
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 2);
                float _add_29_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_2) : "f"(sums_2[4]), "f"(_shfl_xor_sync_30));
                sums_2[4] = _add_29_2;
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 4);
                float _add_31_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_32));
                sums_2[0] = _add_31_1;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 4);
                float _add_33_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33_2) : "f"(sums_2[1]), "f"(_shfl_xor_sync_34));
                sums_2[1] = _add_33_2;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 4);
                float _add_35_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35_2) : "f"(sums_2[2]), "f"(_shfl_xor_sync_36));
                sums_2[2] = _add_35_2;
                float _shfl_xor_sync_38_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 4);
                float _add_37_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37_2) : "f"(sums_2[3]), "f"(_shfl_xor_sync_38_1));
                sums_2[3] = _add_37_2;
                float _shfl_xor_sync_40_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 4);
                float _add_39_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_2) : "f"(sums_2[4]), "f"(_shfl_xor_sync_40_1));
                sums_2[4] = _add_39_2;
                float _shfl_xor_sync_42_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 8);
                float _add_41_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_42_1));
                sums_2[0] = _add_41_1;
                float _shfl_xor_sync_44_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 8);
                float _add_43_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43_1) : "f"(sums_2[1]), "f"(_shfl_xor_sync_44_1));
                sums_2[1] = _add_43_1;
                float _shfl_xor_sync_46_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 8);
                float _add_45_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45_1) : "f"(sums_2[2]), "f"(_shfl_xor_sync_46_1));
                sums_2[2] = _add_45_1;
                float _shfl_xor_sync_48_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 8);
                float _add_47_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47_1) : "f"(sums_2[3]), "f"(_shfl_xor_sync_48_1));
                sums_2[3] = _add_47_1;
                float _shfl_xor_sync_50_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 8);
                float _add_49_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_50_1));
                sums_2[4] = _add_49_1;
                float _shfl_xor_sync_52_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 16);
                float _add_51_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_52_1));
                sums_2[0] = _add_51_1;
                float _shfl_xor_sync_54_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 16);
                float _add_53_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_2) : "f"(sums_2[1]), "f"(_shfl_xor_sync_54_1));
                sums_2[1] = _add_53_2;
                float _shfl_xor_sync_56_2 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 16);
                float _add_55_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55_1) : "f"(sums_2[2]), "f"(_shfl_xor_sync_56_2));
                sums_2[2] = _add_55_1;
                float _shfl_xor_sync_58_2 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 16);
                float _add_57_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57_1) : "f"(sums_2[3]), "f"(_shfl_xor_sync_58_2));
                sums_2[3] = _add_57_1;
                float _shfl_xor_sync_60_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 16);
                float _add_59_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_60_1));
                sums_2[4] = _add_59_1;
                float s_1 = 0.0f;
                s_1 = ((lane_in_warp == 0) ? sums_2[0] : s_1);
                s_1 = ((lane_in_warp == 1) ? sums_2[1] : s_1);
                s_1 = ((lane_in_warp == 2) ? sums_2[2] : s_1);
                s_1 = ((lane_in_warp == 3) ? sums_2[3] : s_1);
                float q_1 = sums_2[4];
                float _fma_62;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(q_1), "f"(6.1035156e-05f), "f"(hc_norm_eps));
                float _sqrt_61;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_61) : "f"(_fma_62));
                float r_1 = __frcp_rn(_sqrt_61);
#else
                float _fma_204;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(q_3), "f"(6.1035156e-05f), "f"(hc_norm_eps));
                float _sqrt_203;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_203) : "f"(_fma_204));
                float r_1 = __frcp_rn(_sqrt_203);
#endif
                float z_1;
#if __CUDA_ARCH__ == 1070
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z_1) : "f"(r_1), "f"(s_1));
#else
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z_1) : "f"(r_1), "f"(s_2));
#endif
                float a_map_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map_1) : "f"(z_1), "f"(sc_1), "f"(bs_1));
                float a_1 = ((lane_in_warp < 24) ? a_map_1 : 0.0f);
#if __CUDA_ARCH__ == 1070
                float _sub_63;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(0.0f), "f"(fabsf(a_1)));
                float _exp_2 = __nv_expf(_sub_63);
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(1.0f), "f"(_exp_2));
                float _fdiv_rn_2 = __fdiv_rn(1.0f, _add_38);
                float _fdiv_rn_3 = __fdiv_rn(_exp_2, _add_38);
#else
                float _sub_205;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_205) : "f"(0.0f), "f"(fabsf(a_1)));
                float _exp_2 = __nv_expf(_sub_205);
                float _add_50_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50_1) : "f"(1.0f), "f"(_exp_2));
                float _fdiv_rn_2 = __fdiv_rn(1.0f, _add_50_1);
                float _fdiv_rn_3 = __fdiv_rn(_exp_2, _add_50_1);
#endif
                float sig_1 = ((a_1 >= 0.0f) ? _fdiv_rn_2 : _fdiv_rn_3);
                float sneg_1 = ((a_1 >= 0.0f) ? _fdiv_rn_3 : _fdiv_rn_2);
                float pre;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(pre) : "f"(sig_1), "f"(hc_pre_eps));
                __syncwarp();
                if (lane_in_warp < 4) {
                    tail_scratch[30 + warp / 4 * 4 + lane_in_warp] = pre;
                }
            }
        }
        asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
        token_1 = tail_control[24 + warp / 4];
        if (token_1 < num_tokens) {
            float coeff_1 = ((lane_in_warp < 4) ? tail_scratch[30 + warp / 4 * 4 + lane_in_warp] : 0.0f);
            unsigned int retained[16];
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            #pragma unroll
            for (int pack = 0; pack < 4; pack++) {
                unsigned int hidden_pack = warp % 4 * 4 + (unsigned int)pack;
                unsigned long long off = (unsigned long long)token_1 * 16384 + (unsigned long long)(hidden_pack * 256) + (unsigned long long)(lane_in_warp * 8);
                float values_1[8];
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, coeff_1, 0);
#if __CUDA_ARCH__ == 1070
                float _vec_load_26[8];
#else
                float _vec_load_52[8];
#endif
                {
#if __CUDA_ARCH__ == 1070
                    const uint4* _vptr_64 = reinterpret_cast<const uint4*>(new_residual + off + 0);
                    uint4 _vld_64[1];
#else
                    const uint4* _vptr_206 = reinterpret_cast<const uint4*>(new_residual + off + 0);
                    uint4 _vld_206[1];
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_vld_64[_blk].x), "=r"(_vld_64[_blk].y), "=r"(_vld_64[_blk].z), "=r"(_vld_64[_blk].w) : "l"((const void*)(_vptr_64 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
#else
                            : "=r"(_vld_206[_blk].x), "=r"(_vld_206[_blk].y), "=r"(_vld_206[_blk].z), "=r"(_vld_206[_blk].w) : "l"((const void*)(_vptr_206 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_206 = reinterpret_cast<uint32_t*>(&_vld_206[_blk]);
#endif
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                            (&_vec_load_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                            (&_vec_load_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
#else
                            (&_vec_load_52[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_206[_pair]) << 16);
                            (&_vec_load_52[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_206[_pair]) & 0xffff0000u);
#endif
                        }
                    }
                }
#if __CUDA_ARCH__ == 1070
                float2 _f32x2_mul_a_65 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_66 = make_float2(_vec_load_26[0], _vec_load_26[1]);
#else
                float2 _f32x2_mul_a_207 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_208 = make_float2(_vec_load_52[0], _vec_load_52[1]);
#endif
                float2 pair_2;
#if __CUDA_ARCH__ == 1070
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
#else
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_207), "l"(*(const unsigned long long*)&_f32x2_mul_b_208));
#endif
                values_1[0] = pair_2.x;
                values_1[1] = pair_2.y;
#if __CUDA_ARCH__ == 1070
                float2 _f32x2_mul_a_67 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_68 = make_float2(_vec_load_26[2], _vec_load_26[3]);
#else
                float2 _f32x2_mul_a_209 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_210 = make_float2(_vec_load_52[2], _vec_load_52[3]);
#endif
                float2 pair_0;
#if __CUDA_ARCH__ == 1070
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
#else
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_209), "l"(*(const unsigned long long*)&_f32x2_mul_b_210));
#endif
                values_1[2] = pair_0.x;
                values_1[3] = pair_0.y;
#if __CUDA_ARCH__ == 1070
                float2 _f32x2_mul_a_69 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_70 = make_float2(_vec_load_26[4], _vec_load_26[5]);
#else
                float2 _f32x2_mul_a_211 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_212 = make_float2(_vec_load_52[4], _vec_load_52[5]);
#endif
                float2 pair_1_1;
#if __CUDA_ARCH__ == 1070
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
#else
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_211), "l"(*(const unsigned long long*)&_f32x2_mul_b_212));
#endif
                values_1[4] = pair_1_1.x;
                values_1[5] = pair_1_1.y;
#if __CUDA_ARCH__ == 1070
                float2 _f32x2_mul_a_71 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_72 = make_float2(_vec_load_26[6], _vec_load_26[7]);
#else
                float2 _f32x2_mul_a_213 = make_float2(_shfl_0, _shfl_0);
                float2 _f32x2_mul_b_214 = make_float2(_vec_load_52[6], _vec_load_52[7]);
#endif
                float2 pair_2_1;
#if __CUDA_ARCH__ == 1070
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
#else
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&_f32x2_mul_b_214));
#endif
                values_1[6] = pair_2_1.x;
                values_1[7] = pair_2_1.y;
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, coeff_1, 1);
#if __CUDA_ARCH__ == 1070
                float _vec_load_27[8];
#else
                float _vec_load_53[8];
#endif
                {
#if __CUDA_ARCH__ == 1070
                    const uint4* _vptr_73 = reinterpret_cast<const uint4*>(new_residual + (off + 4096) + 0);
                    uint4 _vld_73[1];
#else
                    const uint4* _vptr_215 = reinterpret_cast<const uint4*>(new_residual + (off + 4096) + 0);
                    uint4 _vld_215[1];
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_vld_73[_blk].x), "=r"(_vld_73[_blk].y), "=r"(_vld_73[_blk].z), "=r"(_vld_73[_blk].w) : "l"((const void*)(_vptr_73 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_73 = reinterpret_cast<uint32_t*>(&_vld_73[_blk]);
#else
                            : "=r"(_vld_215[_blk].x), "=r"(_vld_215[_blk].y), "=r"(_vld_215[_blk].z), "=r"(_vld_215[_blk].w) : "l"((const void*)(_vptr_215 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_215 = reinterpret_cast<uint32_t*>(&_vld_215[_blk]);
#endif
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                            (&_vec_load_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) << 16);
                            (&_vec_load_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) & 0xffff0000u);
#else
                            (&_vec_load_53[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_215[_pair]) << 16);
                            (&_vec_load_53[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_215[_pair]) & 0xffff0000u);
#endif
                        }
                    }
                }
#if __CUDA_ARCH__ == 1070
                float2 pair_3 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[0], _vec_load_27[1]), make_float2(values_1[0], values_1[1]));
#else
                float2 pair_3 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[0], _vec_load_53[1]), make_float2(values_1[0], values_1[1]));
#endif
                values_1[0] = pair_3.x;
                values_1[1] = pair_3.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_4 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[2], _vec_load_27[3]), make_float2(values_1[2], values_1[3]));
#else
                float2 pair_4 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[2], _vec_load_53[3]), make_float2(values_1[2], values_1[3]));
#endif
                values_1[2] = pair_4.x;
                values_1[3] = pair_4.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_5_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[4], _vec_load_27[5]), make_float2(values_1[4], values_1[5]));
#else
                float2 pair_5_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[4], _vec_load_53[5]), make_float2(values_1[4], values_1[5]));
#endif
                values_1[4] = pair_5_1.x;
                values_1[5] = pair_5_1.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_6 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[6], _vec_load_27[7]), make_float2(values_1[6], values_1[7]));
#else
                float2 pair_6 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[6], _vec_load_53[7]), make_float2(values_1[6], values_1[7]));
#endif
                values_1[6] = pair_6.x;
                values_1[7] = pair_6.y;
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, coeff_1, 2);
#if __CUDA_ARCH__ == 1070
                float _vec_load_28[8];
#else
                float _vec_load_54[8];
#endif
                {
#if __CUDA_ARCH__ == 1070
                    const uint4* _vptr_74 = reinterpret_cast<const uint4*>(new_residual + (off + 8192) + 0);
                    uint4 _vld_74[1];
#else
                    const uint4* _vptr_216 = reinterpret_cast<const uint4*>(new_residual + (off + 8192) + 0);
                    uint4 _vld_216[1];
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_vld_74[_blk].x), "=r"(_vld_74[_blk].y), "=r"(_vld_74[_blk].z), "=r"(_vld_74[_blk].w) : "l"((const void*)(_vptr_74 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_74 = reinterpret_cast<uint32_t*>(&_vld_74[_blk]);
#else
                            : "=r"(_vld_216[_blk].x), "=r"(_vld_216[_blk].y), "=r"(_vld_216[_blk].z), "=r"(_vld_216[_blk].w) : "l"((const void*)(_vptr_216 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_216 = reinterpret_cast<uint32_t*>(&_vld_216[_blk]);
#endif
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                            (&_vec_load_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) << 16);
                            (&_vec_load_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) & 0xffff0000u);
#else
                            (&_vec_load_54[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_216[_pair]) << 16);
                            (&_vec_load_54[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_216[_pair]) & 0xffff0000u);
#endif
                        }
                    }
                }
#if __CUDA_ARCH__ == 1070
                float2 pair_7 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[0], _vec_load_28[1]), make_float2(values_1[0], values_1[1]));
#else
                float2 pair_7 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[0], _vec_load_54[1]), make_float2(values_1[0], values_1[1]));
#endif
                values_1[0] = pair_7.x;
                values_1[1] = pair_7.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_8_1 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[2], _vec_load_28[3]), make_float2(values_1[2], values_1[3]));
#else
                float2 pair_8_1 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[2], _vec_load_54[3]), make_float2(values_1[2], values_1[3]));
#endif
                values_1[2] = pair_8_1.x;
                values_1[3] = pair_8_1.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_9 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[4], _vec_load_28[5]), make_float2(values_1[4], values_1[5]));
#else
                float2 pair_9 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[4], _vec_load_54[5]), make_float2(values_1[4], values_1[5]));
#endif
                values_1[4] = pair_9.x;
                values_1[5] = pair_9.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_10 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[6], _vec_load_28[7]), make_float2(values_1[6], values_1[7]));
#else
                float2 pair_10 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[6], _vec_load_54[7]), make_float2(values_1[6], values_1[7]));
#endif
                values_1[6] = pair_10.x;
                values_1[7] = pair_10.y;
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, coeff_1, 3);
#if __CUDA_ARCH__ == 1070
                float _vec_load_29[8];
#else
                float _vec_load_55[8];
#endif
                {
#if __CUDA_ARCH__ == 1070
                    const uint4* _vptr_75 = reinterpret_cast<const uint4*>(new_residual + (off + 12288) + 0);
                    uint4 _vld_75[1];
#else
                    const uint4* _vptr_217 = reinterpret_cast<const uint4*>(new_residual + (off + 12288) + 0);
                    uint4 _vld_217[1];
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_vld_75[_blk].x), "=r"(_vld_75[_blk].y), "=r"(_vld_75[_blk].z), "=r"(_vld_75[_blk].w) : "l"((const void*)(_vptr_75 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_75 = reinterpret_cast<uint32_t*>(&_vld_75[_blk]);
#else
                            : "=r"(_vld_217[_blk].x), "=r"(_vld_217[_blk].y), "=r"(_vld_217[_blk].z), "=r"(_vld_217[_blk].w) : "l"((const void*)(_vptr_217 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_217 = reinterpret_cast<uint32_t*>(&_vld_217[_blk]);
#endif
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                            (&_vec_load_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) << 16);
                            (&_vec_load_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) & 0xffff0000u);
#else
                            (&_vec_load_55[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_217[_pair]) << 16);
                            (&_vec_load_55[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_217[_pair]) & 0xffff0000u);
#endif
                        }
                    }
                }
#if __CUDA_ARCH__ == 1070
                float2 pair_11 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[0], _vec_load_29[1]), make_float2(values_1[0], values_1[1]));
#else
                float2 pair_11 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[0], _vec_load_55[1]), make_float2(values_1[0], values_1[1]));
#endif
                values_1[0] = pair_11.x;
                values_1[1] = pair_11.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_12 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[2], _vec_load_29[3]), make_float2(values_1[2], values_1[3]));
#else
                float2 pair_12 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[2], _vec_load_55[3]), make_float2(values_1[2], values_1[3]));
#endif
                values_1[2] = pair_12.x;
                values_1[3] = pair_12.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_13 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[4], _vec_load_29[5]), make_float2(values_1[4], values_1[5]));
#else
                float2 pair_13 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[4], _vec_load_55[5]), make_float2(values_1[4], values_1[5]));
#endif
                values_1[4] = pair_13.x;
                values_1[5] = pair_13.y;
#if __CUDA_ARCH__ == 1070
                float2 pair_14 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[6], _vec_load_29[7]), make_float2(values_1[6], values_1[7]));
#else
                float2 pair_14 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[6], _vec_load_55[7]), make_float2(values_1[6], values_1[7]));
#endif
                values_1[6] = pair_14.x;
                values_1[7] = pair_14.y;
                float2 w_pair = make_float2(values_1[0], values_1[1]);
                float2 squares_1 = fma_f32x2_rn_noftz(w_pair, w_pair, make_float2(sum0, sum1));
                sum0 = squares_1.x;
                sum1 = squares_1.y;
                retained[pack * 4] = __as_u32(__float22bfloat162_rn(make_float2(values_1[0], values_1[1])));
                float2 w_pair_15 = make_float2(values_1[2], values_1[3]);
                float2 squares_16 = fma_f32x2_rn_noftz(w_pair_15, w_pair_15, make_float2(sum0, sum1));
                sum0 = squares_16.x;
                sum1 = squares_16.y;
                retained[pack * 4 + 1] = __as_u32(__float22bfloat162_rn(make_float2(values_1[2], values_1[3])));
                float2 w_pair_17 = make_float2(values_1[4], values_1[5]);
                float2 squares_18 = fma_f32x2_rn_noftz(w_pair_17, w_pair_17, make_float2(sum0, sum1));
                sum0 = squares_18.x;
                sum1 = squares_18.y;
                retained[pack * 4 + 2] = __as_u32(__float22bfloat162_rn(make_float2(values_1[4], values_1[5])));
                float2 w_pair_19 = make_float2(values_1[6], values_1[7]);
                float2 squares_20 = fma_f32x2_rn_noftz(w_pair_19, w_pair_19, make_float2(sum0, sum1));
                sum0 = squares_20.x;
                sum1 = squares_20.y;
                retained[pack * 4 + 3] = __as_u32(__float22bfloat162_rn(make_float2(values_1[6], values_1[7])));
            }
#if __CUDA_ARCH__ == 1070
            float _add_76;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(sum0), "f"(sum1));
            float _warp_reduce_0 = _add_76;
#else
            float _add_218;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(sum0), "f"(sum1));
            float _warp_reduce_0 = _add_218;
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
            float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            float _add_39_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_3) : "f"(row_sum), "f"(_shfl_xor_sync_77));
            float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, _add_39_3, 1);
            float _add_78;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_39_3), "f"(_shfl_xor_sync_79));
            row_sum = _add_78;
            float _shfl_sync_80 = __shfl_sync(0xFFFFFFFF, row_sum, 0);
            row_sum = _shfl_sync_80;
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(row_sum), "f"(0.00024414062f), "f"(rmsnorm_eps));
            float _sqrt_82;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_82) : "f"(_fma_83));
            float _rcp_81 = __frcp_rn(_sqrt_82);
#else
            float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            float _add_51;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(row_sum), "f"(_shfl_xor_sync_219));
            float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, _add_51, 1);
            float _add_220;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(_add_51), "f"(_shfl_xor_sync_221));
            row_sum = _add_220;
            float _shfl_sync_222 = __shfl_sync(0xFFFFFFFF, row_sum, 0);
            row_sum = _shfl_sync_222;
            float _fma_225;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(row_sum), "f"(0.00024414062f), "f"(rmsnorm_eps));
            float _sqrt_224;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_224) : "f"(_fma_225));
            float _rcp_223 = __frcp_rn(_sqrt_224);
#endif
            float rstd;
#if __CUDA_ARCH__ == 1070
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rstd) : "f"(_rcp_81), "f"(rmsnorm_scale));
#else
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rstd) : "f"(_rcp_223), "f"(rmsnorm_scale));
#endif
            unsigned int first_hidden_pack = warp % 4 * 4;
            unsigned int weight_cache[4];
            {
#if __CUDA_ARCH__ == 1070
                const uint4* _vptr_84 = reinterpret_cast<const uint4*>(rmsnorm_weight + (first_hidden_pack * 256 + lane_in_warp * 8) + 0);
                uint4* _vdst_84 = reinterpret_cast<uint4*>(&weight_cache[0]);
#else
                const uint4* _vptr_226 = reinterpret_cast<const uint4*>(rmsnorm_weight + (first_hidden_pack * 256 + lane_in_warp * 8) + 0);
                uint4* _vdst_226 = reinterpret_cast<uint4*>(&weight_cache[0]);
#endif
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1070
                    _vdst_84[_blk] = _vptr_84[_blk];
#else
                    _vdst_226[_blk] = _vptr_226[_blk];
#endif
                }
            }
            #pragma unroll
            for (int pack_1 = 0; pack_1 < 4; pack_1++) {
                unsigned int hidden_pack_1 = warp % 4 * 4 + (unsigned int)pack_1;
                unsigned int next_weight[4];
                if (pack_1 + 1 < 4) {
                    {
#if __CUDA_ARCH__ == 1070
                        const uint4* _vptr_85 = reinterpret_cast<const uint4*>(rmsnorm_weight + ((hidden_pack_1 + 1) * 256 + lane_in_warp * 8) + 0);
                        uint4* _vdst_85 = reinterpret_cast<uint4*>(&next_weight[0]);
#else
                        const uint4* _vptr_227 = reinterpret_cast<const uint4*>(rmsnorm_weight + ((hidden_pack_1 + 1) * 256 + lane_in_warp * 8) + 0);
                        uint4* _vdst_227 = reinterpret_cast<uint4*>(&next_weight[0]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1070
                            _vdst_85[_blk] = _vptr_85[_blk];
#else
                            _vdst_227[_blk] = _vptr_227[_blk];
#endif
                        }
                    }
                }
                unsigned int normalized_words[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned int word_3 = retained[pack_1 * 4 + j];
                    float lo = __uint_as_float(word_3 << 16);
                    float hi = __uint_as_float(word_3 & 4294901760u);
                    float weight_lo = __uint_as_float(weight_cache[j] << 16);
                    float weight_hi = __uint_as_float(weight_cache[j] & 4294901760u);
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_86 = make_float2(lo, hi);
                    float2 _f32x2_mul_b_87 = make_float2(rstd, rstd);
#else
                    float2 _f32x2_mul_a_228 = make_float2(lo, hi);
                    float2 _f32x2_mul_b_229 = make_float2(rstd, rstd);
#endif
                    float2 scaled;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
                    float2 _f32x2_mul_b_88 = make_float2(weight_lo, weight_hi);
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
                    float2 _f32x2_mul_b_230 = make_float2(weight_lo, weight_hi);
#endif
                    float2 weighted;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
#endif
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
#if __CUDA_ARCH__ == 1070
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
#else
                float _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
#endif
                float amax1;
#if __CUDA_ARCH__ == 1070
                asm("max.f32 %0, %1, %2;" : "=f"(amax1) : "f"(amax), "f"(_shfl_xor_sync_89));
                float _shfl_xor_sync_90_1 = __shfl_xor_sync(0xFFFFFFFF, amax1, 2);
#else
                asm("max.f32 %0, %1, %2;" : "=f"(amax1) : "f"(amax), "f"(_shfl_xor_sync_231));
                float _shfl_xor_sync_232_1 = __shfl_xor_sync(0xFFFFFFFF, amax1, 2);
#endif
                float amax2;
#if __CUDA_ARCH__ == 1070
                asm("max.f32 %0, %1, %2;" : "=f"(amax2) : "f"(amax1), "f"(_shfl_xor_sync_90_1));
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, amax2, 4);
#else
                asm("max.f32 %0, %1, %2;" : "=f"(amax2) : "f"(amax1), "f"(_shfl_xor_sync_232_1));
                float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, amax2, 4);
#endif
                float amax4;
#if __CUDA_ARCH__ == 1070
                asm("max.f32 %0, %1, %2;" : "=f"(amax4) : "f"(amax2), "f"(_shfl_xor_sync_91));
                float _shfl_xor_sync_92_1 = __shfl_xor_sync(0xFFFFFFFF, amax4, 8);
#else
                asm("max.f32 %0, %1, %2;" : "=f"(amax4) : "f"(amax2), "f"(_shfl_xor_sync_233));
                float _shfl_xor_sync_234_1 = __shfl_xor_sync(0xFFFFFFFF, amax4, 8);
#endif
                float amax8;
#if __CUDA_ARCH__ == 1070
                asm("max.f32 %0, %1, %2;" : "=f"(amax8) : "f"(amax4), "f"(_shfl_xor_sync_92_1));
#else
                asm("max.f32 %0, %1, %2;" : "=f"(amax8) : "f"(amax4), "f"(_shfl_xor_sync_234_1));
#endif
                float absmax;
                asm("max.f32 %0, %1, %2;" : "=f"(absmax) : "f"(amax8), "f"(1e-10f));
                float scale_raw = __fdiv_rn(absmax, 448.0f);
                unsigned int floored_bits = __as_u32(max_noftz(scale_raw, 1e-10f));
                float pow2 = __uint_as_float(floored_bits + 8388607 & 2139095040);
                float scale = ((fp8_sf_ue8m0 != 0) ? pow2 : scale_raw);
#if __CUDA_ARCH__ == 1070
                float q_2[8];
#else
                float q_4[8];
#endif
                #pragma unroll
                for (int j_1 = 0; j_1 < 4; j_1++) {
#if __CUDA_ARCH__ == 1070
                    float _fdiv_93 = __fdiv_rn(__uint_as_float(normalized_words[j_1] << 16), scale);
                    q_2[2 * j_1] = _fdiv_93;
                    float _fdiv_94 = __fdiv_rn(__uint_as_float(normalized_words[j_1] & 4294901760u), scale);
                    q_2[2 * j_1 + 1] = _fdiv_94;
#else
                    float _fdiv_235 = __fdiv_rn(__uint_as_float(normalized_words[j_1] << 16), scale);
                    q_4[2 * j_1] = _fdiv_235;
                    float _fdiv_236 = __fdiv_rn(__uint_as_float(normalized_words[j_1] & 4294901760u), scale);
                    q_4[2 * j_1 + 1] = _fdiv_236;
#endif
                }
                unsigned long long off_0 = (unsigned long long)token_1 * 4096 + (unsigned long long)(hidden_pack_1 * 256) + (unsigned long long)(lane_in_warp * 8);
                unsigned int _packed_0[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
#if __CUDA_ARCH__ == 1070
                        : "=r"(_packed) : "f"(q_2[0]), "f"(q_2[1]),
                                           "f"(q_2[2]), "f"(q_2[3]));
#else
                        : "=r"(_packed) : "f"(q_4[0]), "f"(q_4[1]),
                                           "f"(q_4[2]), "f"(q_4[3]));
#endif
                    _packed_0[0] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
#if __CUDA_ARCH__ == 1070
                        : "=r"(_packed) : "f"(q_2[4]), "f"(q_2[5]),
                                           "f"(q_2[6]), "f"(q_2[7]));
#else
                        : "=r"(_packed) : "f"(q_4[4]), "f"(q_4[5]),
                                           "f"(q_4[6]), "f"(q_4[7]));
#endif
                    _packed_0[1] = _packed;
                }
                reinterpret_cast<int2*>(y_fp8 + off_0)[0] = reinterpret_cast<int2*>(_packed_0)[0];
                unsigned long long sf_idx = (unsigned long long)token_1 * 32 + (unsigned long long)(hidden_pack_1 * 2 + lane_in_warp / 16);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(y_fp8_sf + sf_idx)), "f"(scale), "r"((unsigned int)(lane_in_warp % 16 == 0)) : "memory");
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
#if __CUDA_ARCH__ == 1070
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
#else
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
#endif
    }
}

} // extern "C"
