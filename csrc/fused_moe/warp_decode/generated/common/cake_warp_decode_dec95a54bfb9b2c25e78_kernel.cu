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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 176
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 175104
#define SMEM_EPI_STAGING_STAGE_BYTES 256
#define SMEM_EPI_STAGING_STRIDE 256
#define SMEM_SMEM_SFA_OFF 177152
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 197632
#define SMEM_SMEM_SFB_STAGE_BYTES 256
#define SMEM_SMEM_SFB_STRIDE 256
#define SMEM_WORK_RESPONSE_OFF 198912
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_SMEM_V6_OFF 198960
#define SMEM_SMEM_V6_STAGE_BYTES 4096
#define SMEM_SMEM_V6_STRIDE 4096
#define SMEM_SMEM_V7_OFF 203056
#define SMEM_SMEM_V7_STAGE_BYTES 256
#define SMEM_SMEM_V7_STRIDE 256
#define SMEM_SMEM_V8_OFF 203312
#define SMEM_SMEM_V8_STAGE_BYTES 16
#define SMEM_SMEM_V8_STRIDE 16
#define SMEM_SMEM_V9_OFF 203328
#define SMEM_SMEM_V9_STAGE_BYTES 2048
#define SMEM_SMEM_V9_STRIDE 2048
#define SMEM_SMEM_V10_OFF 205376
#define SMEM_SMEM_V10_STAGE_BYTES 2048
#define SMEM_SMEM_V10_STRIDE 2048
#define SMEM_SMEM_V11_OFF 207424
#define SMEM_SMEM_V11_STAGE_BYTES 4096
#define SMEM_SMEM_V11_STRIDE 4096
#define SMEM_SMEM_V12_OFF 211520
#define SMEM_SMEM_V12_STAGE_BYTES 4096
#define SMEM_SMEM_V12_STRIDE 4096
#define SMEM_SMEM_V13_OFF 215616
#define SMEM_SMEM_V13_STAGE_BYTES 2048
#define SMEM_SMEM_V13_STRIDE 2048
#define SMEM_SMEM_V14_OFF 217664
#define SMEM_SMEM_V14_STAGE_BYTES 2048
#define SMEM_SMEM_V14_STRIDE 2048
#define SMEM_SMEM_V15_OFF 203328
#define SMEM_SMEM_V15_STAGE_BYTES 16384
#define SMEM_SMEM_V15_STRIDE 16384
#define SMEM_SMEM_V16_OFF 203328
#define SMEM_SMEM_V16_STAGE_BYTES 16384
#define SMEM_SMEM_V16_STRIDE 16384
#define SMEM_TOTAL 219776
#define THREADS 512
#define BLOCK_M 128
#define BLOCK_N 8
#define BLOCK_K 512
#define WEIGHTS_SHUFFLED 1
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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






__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_warp_decode_dec95a54bfb9b2c25e78(const __grid_constant__ CUtensorMap A, uint8_t* __restrict__ B, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int* __restrict__ work_counter, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles, int* __restrict__ route_experts, int* __restrict__ route_slots, float* __restrict__ pack_ready, int* __restrict__ done_counter, int route_count, int top_k, int local_expert_offset, int num_experts, int initial_work, int launch_ctas)
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
    #define a_full_addr (mbar_base + 0)
    #define b_full_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 80)
    #define sfb_full_addr (mbar_base + 120)
    #define sfa_free_addr (mbar_base + 160)
    #define sfb_free_addr (mbar_base + 200)
    #define tmem_sfa_full_addr (mbar_base + 240)
    #define tmem_sfb_full_addr (mbar_base + 280)
    #define k_done_addr (mbar_base + 320)
    #define mma_full_addr (mbar_base + 360)
    #define mma_free_addr (mbar_base + 376)
    #define work_full_addr (mbar_base + 392)
    #define work_empty_addr (mbar_base + 416)
    #define throttle_full_addr (mbar_base + 440)
    #define throttle_empty_addr (mbar_base + 464)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 175104);
    const int epi_staging_addr = smem + 175104;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 177152);
    const int smem_sfa_addr = smem + 177152;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_sfb_addr = smem + 197632;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 198912);
    const int work_response_addr = smem + 198912;
    int* smem_v6 = reinterpret_cast<int*>(smem_raw + 198960);
    const int smem_v6_addr = smem + 198960;
    int* smem_v7 = reinterpret_cast<int*>(smem_raw + 203056);
    const int smem_v7_addr = smem + 203056;
    int* smem_v8 = reinterpret_cast<int*>(smem_raw + 203312);
    const int smem_v8_addr = smem + 203312;
    int* smem_v9 = reinterpret_cast<int*>(smem_raw + 203328);
    const int smem_v9_addr = smem + 203328;
    int* smem_v10 = reinterpret_cast<int*>(smem_raw + 205376);
    const int smem_v10_addr = smem + 205376;
    int* smem_v11 = reinterpret_cast<int*>(smem_raw + 207424);
    const int smem_v11_addr = smem + 207424;
    int* smem_v12 = reinterpret_cast<int*>(smem_raw + 211520);
    const int smem_v12_addr = smem + 211520;
    int* smem_v13 = reinterpret_cast<int*>(smem_raw + 215616);
    const int smem_v13_addr = smem + 215616;
    int* smem_v14 = reinterpret_cast<int*>(smem_raw + 217664);
    const int smem_v14_addr = smem + 217664;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 203328);
    const int smem_v15_addr = smem + 203328;
    int* smem_v16 = reinterpret_cast<int*>(smem_raw + 203328);
    const int smem_v16_addr = smem + 203328;
    const int tid_0 = tid;
    int _min_0 = ((tid_0) < (route_count - 1) ? (tid_0) : (route_count - 1));
    int raw_expert = route_experts[_min_0];
    int expert_slot_2_0 = tid_0;
    smem_v6[expert_slot_2_0] = 0;
    int expert_slot_2_1 = tid_0 + 512;
    smem_v6[expert_slot_2_1] = 0;
    int chunk_word_1_0 = tid_0;
    smem_v16[chunk_word_1_0] = 0;
    int chunk_word_1_1 = tid_0 + 512;
    smem_v16[chunk_word_1_1] = 0;
    int chunk_word_1_2 = tid_0 + 1024;
    smem_v16[chunk_word_1_2] = 0;
    int chunk_word_1_3 = tid_0 + 1536;
    smem_v16[chunk_word_1_3] = 0;
    int chunk_word_1_4 = tid_0 + 2048;
    smem_v16[chunk_word_1_4] = 0;
    int chunk_word_1_5 = tid_0 + 2560;
    smem_v16[chunk_word_1_5] = 0;
    int chunk_word_1_6 = tid_0 + 3072;
    smem_v16[chunk_word_1_6] = 0;
    int chunk_word_1_7 = tid_0 + 3584;
    smem_v16[chunk_word_1_7] = 0;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 61 barriers)
    // Mbarriers at smem_raw[0..488)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // b_full: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // sfa_full: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // sfb_full: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // sfa_free: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // sfb_free: 5 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            // tmem_sfa_full: 5 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // tmem_sfb_full: 5 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // k_done: 5 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            // mma_free: 2 barriers, init_count=4
            mbarrier_init(smem + 376, 4);
            mbarrier_init(smem + 384, 4);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 392, 1);
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            // work_empty: 3 barriers, init_count=512
            mbarrier_init(smem + 416, 512);
            mbarrier_init(smem + 424, 512);
            mbarrier_init(smem + 432, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 440, 32);
            mbarrier_init(smem + 448, 32);
            mbarrier_init(smem + 456, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 464, 32);
            mbarrier_init(smem + 472, 32);
            mbarrier_init(smem + 480, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 488);
    if (warp == 0) {
        int _tmem_hold = smem + 488;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 16;
    const int tmem_sfb = taddr + 176;
    const int tid_9 = tid;
    const int lane_10 = lane;
    const int warp_11 = warp;
    int publishing = 0;
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        publishing = 1;
    }
    int my_expert = -1;
    int local_expert = raw_expert - local_expert_offset;
    if (tid_9 < route_count && local_expert >= 0 && local_expert < num_experts) {
        my_expert = local_expert;
    }
    __syncthreads();
    unsigned int _warp_vote_0 = __ballot_sync(0xFFFFFFFF, my_expert >= 0);
    int same = _warp_vote_0;
    unsigned int _warp_vote_1 = __ballot_sync(0xFFFFFFFF, (my_expert & 1) == 1);
    int bit_lanes_1_0 = _warp_vote_1;
    int my_bit_1_0 = my_expert & 1;
    same = same & (bit_lanes_1_0 ^ my_bit_1_0 - 1);
    unsigned int _warp_vote_2 = __ballot_sync(0xFFFFFFFF, (my_expert >> 1 & 1) == 1);
    int bit_lanes_1_1 = _warp_vote_2;
    int my_bit_1_1 = my_expert >> 1 & 1;
    same = same & (bit_lanes_1_1 ^ my_bit_1_1 - 1);
    unsigned int _warp_vote_3 = __ballot_sync(0xFFFFFFFF, (my_expert >> 2 & 1) == 1);
    int bit_lanes_1_2 = _warp_vote_3;
    int my_bit_1_2 = my_expert >> 2 & 1;
    same = same & (bit_lanes_1_2 ^ my_bit_1_2 - 1);
    unsigned int _warp_vote_4 = __ballot_sync(0xFFFFFFFF, (my_expert >> 3 & 1) == 1);
    int bit_lanes_1_3 = _warp_vote_4;
    int my_bit_1_3 = my_expert >> 3 & 1;
    same = same & (bit_lanes_1_3 ^ my_bit_1_3 - 1);
    unsigned int _warp_vote_5 = __ballot_sync(0xFFFFFFFF, (my_expert >> 4 & 1) == 1);
    int bit_lanes_1_4 = _warp_vote_5;
    int my_bit_1_4 = my_expert >> 4 & 1;
    same = same & (bit_lanes_1_4 ^ my_bit_1_4 - 1);
    unsigned int _warp_vote_6 = __ballot_sync(0xFFFFFFFF, (my_expert >> 5 & 1) == 1);
    int bit_lanes_1_5 = _warp_vote_6;
    int my_bit_1_5 = my_expert >> 5 & 1;
    same = same & (bit_lanes_1_5 ^ my_bit_1_5 - 1);
    unsigned int _warp_vote_7 = __ballot_sync(0xFFFFFFFF, (my_expert >> 6 & 1) == 1);
    int bit_lanes_1_6 = _warp_vote_7;
    int my_bit_1_6 = my_expert >> 6 & 1;
    same = same & (bit_lanes_1_6 ^ my_bit_1_6 - 1);
    unsigned int _warp_vote_8 = __ballot_sync(0xFFFFFFFF, (my_expert >> 7 & 1) == 1);
    int bit_lanes_1_7 = _warp_vote_8;
    int my_bit_1_7 = my_expert >> 7 & 1;
    same = same & (bit_lanes_1_7 ^ my_bit_1_7 - 1);
    unsigned int _warp_vote_9 = __ballot_sync(0xFFFFFFFF, (my_expert >> 8 & 1) == 1);
    int bit_lanes_1_8 = _warp_vote_9;
    int my_bit_1_8 = my_expert >> 8 & 1;
    same = same & (bit_lanes_1_8 ^ my_bit_1_8 - 1);
    unsigned int _warp_vote_10 = __ballot_sync(0xFFFFFFFF, (my_expert >> 9 & 1) == 1);
    int bit_lanes_1_9 = _warp_vote_10;
    int my_bit_1_9 = my_expert >> 9 & 1;
    same = same & (bit_lanes_1_9 ^ my_bit_1_9 - 1);
    int lane_below = (1 << lane_10) - 1;
    int in_cnt = (int)(unsigned int)__popc(same);
    int in_rank = (int)(unsigned int)__popc(same & lane_below);
    int leader = 0;
    if (my_expert >= 0 && in_rank == 0) {
        leader = 1;
    }
    int rank = 0;
    if (leader == 1) {
        atomicAdd(&smem_v6[my_expert], in_cnt);
        smem_v15[warp_11 * 1024 + my_expert] = in_cnt;
    }
    __syncthreads();
    int _find_first_set_11 = __ffs(same);
    int first = _find_first_set_11 - 1;
    int prefix = 0;
    if (leader == 1) {
        for (int chunk = 0; chunk < warp_11; chunk++) {
            prefix = prefix + (int)smem_v15[chunk * 1024 + my_expert];
        }
    }
    int _max_13 = ((first) > (0) ? (first) : (0));
    int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, prefix, _max_13);
    rank = _shfl_sync_12 + in_rank;
    int expert0 = 2 * tid_9;
    int expert1 = expert0 + 1;
    int count0 = 0;
    int count1 = 0;
    if (expert0 < num_experts) {
        count0 = smem_v6[expert0];
    }
    if (expert1 < num_experts) {
        count1 = smem_v6[expert1];
    }
    int tiles0 = (count0 + 8 - 1) / 8;
    int tiles1 = (count1 + 8 - 1) / 8;
    int own_tiles = tiles0 + tiles1;
    int own_count = count0 + count1;
    int tiles_incl = own_tiles;
    {
        uint32_t tiles_incl_scan = (uint32_t)tiles_incl;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tiles_incl_scan, 1, 32); if (lane >= 1) { tiles_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tiles_incl_scan, 2, 32); if (lane >= 2) { tiles_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tiles_incl_scan, 4, 32); if (lane >= 4) { tiles_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tiles_incl_scan, 8, 32); if (lane >= 8) { tiles_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tiles_incl_scan, 16, 32); if (lane >= 16) { tiles_incl_scan += peer; } }
        tiles_incl = (int)tiles_incl_scan;
    }
    int count_incl = own_count;
    {
        uint32_t count_incl_scan = (uint32_t)count_incl;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, count_incl_scan, 1, 32); if (lane >= 1) { count_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, count_incl_scan, 2, 32); if (lane >= 2) { count_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, count_incl_scan, 4, 32); if (lane >= 4) { count_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, count_incl_scan, 8, 32); if (lane >= 8) { count_incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, count_incl_scan, 16, 32); if (lane >= 16) { count_incl_scan += peer; } }
        count_incl = (int)count_incl_scan;
    }
    if (lane_10 == 31) {
        smem_v7[warp_11] = tiles_incl;
        smem_v7[16 + warp_11] = count_incl;
    }
    __syncthreads();
    if (warp_11 == 0) {
        int warp_tiles = 0;
        int warp_count = 0;
        if (lane_10 < 16) {
            warp_tiles = smem_v7[lane_10];
            warp_count = smem_v7[16 + lane_10];
        }
        int _warp_scan_sum_14 = warp_tiles;
        {
            uint32_t _warp_scan_sum_14_scan = (uint32_t)_warp_scan_sum_14;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_14_scan, 1, 32); if (lane >= 1) { _warp_scan_sum_14_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_14_scan, 2, 32); if (lane >= 2) { _warp_scan_sum_14_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_14_scan, 4, 32); if (lane >= 4) { _warp_scan_sum_14_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_14_scan, 8, 32); if (lane >= 8) { _warp_scan_sum_14_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_14_scan, 16, 32); if (lane >= 16) { _warp_scan_sum_14_scan += peer; } }
            _warp_scan_sum_14 = (int)_warp_scan_sum_14_scan;
        }
        warp_tiles = _warp_scan_sum_14;
        int _warp_scan_sum_15 = warp_count;
        {
            uint32_t _warp_scan_sum_15_scan = (uint32_t)_warp_scan_sum_15;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_15_scan, 1, 32); if (lane >= 1) { _warp_scan_sum_15_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_15_scan, 2, 32); if (lane >= 2) { _warp_scan_sum_15_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_15_scan, 4, 32); if (lane >= 4) { _warp_scan_sum_15_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_15_scan, 8, 32); if (lane >= 8) { _warp_scan_sum_15_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_15_scan, 16, 32); if (lane >= 16) { _warp_scan_sum_15_scan += peer; } }
            _warp_scan_sum_15 = (int)_warp_scan_sum_15_scan;
        }
        warp_count = _warp_scan_sum_15;
        if (lane_10 < 16) {
            smem_v7[32 + lane_10] = warp_tiles;
            smem_v7[48 + lane_10] = warp_count;
        }
    }
    __syncthreads();
    int warp_base_tiles = 0;
    int warp_base_count = 0;
    if (warp_11 > 0) {
        warp_base_tiles = smem_v7[32 + warp_11 - 1];
        warp_base_count = smem_v7[48 + warp_11 - 1];
    }
    int excl_tiles = warp_base_tiles + tiles_incl - own_tiles;
    int excl_count = warp_base_count + count_incl - own_count;
    int start0 = excl_tiles;
    int start1 = excl_tiles + tiles0;
    int total_tiles = smem_v7[47];
    if (tid_9 == 0) {
        smem_v8[0] = total_tiles;
        if (publishing == 1) {
            num_non_exiting_ctas[0] = total_tiles;
        }
    }
    if (expert0 < num_experts) {
        for (int tile_i0 = 0; tile_i0 < tiles0; tile_i0++) {
            int tile0 = start0 + tile_i0;
            int valid_rows0 = ((count0 - tile_i0 * 8) < (8) ? (count0 - tile_i0 * 8) : (8));
            smem_v9[tile0] = expert0;
            smem_v10[tile0] = tile0 * 8 + valid_rows0;
            if (publishing == 1) {
                tile_expert[tile0] = expert0;
                tile_mn_limit[tile0] = tile0 * 8 + valid_rows0;
            }
        }
    }
    if (expert1 < num_experts) {
        for (int tile_i1 = 0; tile_i1 < tiles1; tile_i1++) {
            int tile1 = start1 + tile_i1;
            int valid_rows1 = ((count1 - tile_i1 * 8) < (8) ? (count1 - tile_i1 * 8) : (8));
            smem_v9[tile1] = expert1;
            smem_v10[tile1] = tile1 * 8 + valid_rows1;
            if (publishing == 1) {
                tile_expert[tile1] = expert1;
                tile_mn_limit[tile1] = tile1 * 8 + valid_rows1;
            }
        }
    }
    if (expert0 < num_experts) {
        smem_v11[expert0] = start0;
        smem_v12[expert0] = excl_count;
    }
    if (expert1 < num_experts) {
        smem_v11[expert1] = start1;
        smem_v12[expert1] = excl_count + count0;
    }
    __syncthreads();
    if (my_expert >= 0) {
        int position = smem_v12[my_expert] + rank;
        int route_tile = smem_v11[my_expert] + rank / 8;
        int row = rank % 8;
        int token = tid_9 / top_k;
        smem_v13[position] = token;
        if (row == 0) {
            smem_v14[route_tile] = position;
        }
        if (publishing == 1) {
            int slot = route_tile * 8 + row;
            route_map[slot] = token;
            route_slots[tid_9] = slot;
        }
    }
    if (publishing == 1) {
        __threadfence();
    }
    __syncthreads();
    if (publishing == 1 && tid_9 == 0) {
        reinterpret_cast<volatile float*>(pack_ready)[0] = 1.0f;
    }
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            const int tmem_lane_base = warp % 4 * 32;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            float quant_pair[8] = {0};
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)smem_v8[0]) {
                    break;
                }
                unsigned int tile_work_id = m_tile * (unsigned int)grid_n + n_tile;
                bool tile_profile = _tile_iter < 2;
                int expert = smem_v9[n_tile];
                int valid_rows = (unsigned int)smem_v10[n_tile] - n_tile * (unsigned int)BLOCK_N;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float al = act_alpha[expert];
                float be = act_beta[expert];
                float neg_cl;
                asm("neg.ftz.f32 %0, %1;" : "=f"(neg_cl) : "f"(cl));
                float beta_sg = be * sg;
                float alpha_sg = al * sg;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 8;
                float _tmem_load_0[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                    : "r"(taddr + (unsigned int)(tmem_lane_base << 16) + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                    : "r"(taddr + (unsigned int)(tmem_lane_base + 16 << 16) + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                for (int token_group = 0; token_group < 1; token_group++) {
                    int token0 = lane_1 % 4 * 2 + token_group * 8;
                    int token1 = token0 + 1;
                    float value00 = 0.0f;
                    float value01 = 0.0f;
                    float value10 = 0.0f;
                    float value11 = 0.0f;
                    float oa_up00 = fminf(max_noftz(_tmem_load_0[token_group * 4] * sg, neg_cl), cl);
                    float oa_up01 = fminf(max_noftz(_tmem_load_0[token_group * 4 + 1] * sg, neg_cl), cl);
                    float oa_up10 = fminf(max_noftz(_tmem_load_1[token_group * 4] * sg, neg_cl), cl);
                    float oa_up11 = fminf(max_noftz(_tmem_load_1[token_group * 4 + 1] * sg, neg_cl), cl);
                    float oa_gate00 = fminf(_tmem_load_0[token_group * 4 + 2] * sg, cl);
                    float oa_gate01 = fminf(_tmem_load_0[token_group * 4 + 3] * sg, cl);
                    float oa_gate10 = fminf(_tmem_load_1[token_group * 4 + 2] * sg, cl);
                    float oa_gate11 = fminf(_tmem_load_1[token_group * 4 + 3] * sg, cl);
                    float _exp_0 = __expf(-(al * oa_gate00));
                    value00 = oa_gate00 / (1.0f + _exp_0) * (oa_up00 + be);
                    float _exp_1 = __expf(-(al * oa_gate01));
                    value01 = oa_gate01 / (1.0f + _exp_1) * (oa_up01 + be);
                    float _exp_2 = __expf(-(al * oa_gate10));
                    value10 = oa_gate10 / (1.0f + _exp_2) * (oa_up10 + be);
                    float _exp_3 = __expf(-(al * oa_gate11));
                    value11 = oa_gate11 / (1.0f + _exp_3) * (oa_up11 + be);
                    float block_max0;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(block_max0) : "f"(fabsf(value00)), "f"(fabsf(value10)));
                    float block_max1;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(block_max1) : "f"(fabsf(value01)), "f"(fabsf(value11)));
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 4);
                    float _max_4;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(block_max0), "f"(_shfl_xor_sync_5));
                    block_max0 = _max_4;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 4);
                    float _max_6;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(block_max1), "f"(_shfl_xor_sync_7));
                    block_max1 = _max_6;
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 8);
                    float _max_8;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(block_max0), "f"(_shfl_xor_sync_9));
                    block_max0 = _max_8;
                    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 8);
                    float _max_10;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(block_max1), "f"(_shfl_xor_sync_11));
                    block_max1 = _max_10;
                    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 16);
                    float _max_12;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(block_max0), "f"(_shfl_xor_sync_13));
                    block_max0 = _max_12;
                    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 16);
                    float _max_14;
                    asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(block_max1), "f"(_shfl_xor_sync_15));
                    block_max1 = _max_14;
                    float scale0 = 0.0f;
                    float scale1 = 0.0f;
                    float oa_quant_scale_max = sc / sg;
                    float2 _f32x2_mul_a_16 = make_float2(block_max0, block_max1);
                    float2 _f32x2_mul_b_17 = make_float2(oa_quant_scale_max, oa_quant_scale_max);
                    float2 oa_scaled_max;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&oa_scaled_max) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
                    oa_scaled_max = mul_f32x2(oa_scaled_max, make_float2(0.16666667f, 0.16666667f));
                    uint16_t _e4m3x2_20;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_20) : "f"(0.0f), "f"(oa_scaled_max.x));
                    uint8_t _cvt_satfinite_e4m3_f32_19 = (uint8_t)_e4m3x2_20;
                    uint32_t _f16x2_decode_21;
                    float _fp8_e4m3_decode_18;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_21) : "h"(_e4m3x2_20));
                    uint16_t _f16_decode_21 = (uint16_t)_f16x2_decode_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_18) : "h"(_f16_decode_21));
                    scale0 = _fp8_e4m3_decode_18;
                    uint16_t _e4m3x2_24;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_24) : "f"(0.0f), "f"(oa_scaled_max.y));
                    uint8_t _cvt_satfinite_e4m3_f32_23 = (uint8_t)_e4m3x2_24;
                    uint32_t _f16x2_decode_25;
                    float _fp8_e4m3_decode_22;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_25) : "h"(_e4m3x2_24));
                    uint16_t _f16_decode_25 = (uint16_t)_f16x2_decode_25;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_22) : "h"(_f16_decode_25));
                    scale1 = _fp8_e4m3_decode_22;
                    float inv_scale0 = 0.0f;
                    float inv_scale1 = 0.0f;
                    uint32_t _setp_26_bits;
                    asm("{ .reg .pred p; setp.neu.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_26_bits) : "f"(scale0), "f"(0.0f));
                    bool _setp_26 = _setp_26_bits != 0;
                    if (_setp_26) {
                        float _rcp_27 = approx_rcp(scale0);
                        inv_scale0 = _rcp_27;
                    }
                    uint32_t _setp_28_bits;
                    asm("{ .reg .pred p; setp.neu.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_28_bits) : "f"(scale1), "f"(0.0f));
                    bool _setp_28 = _setp_28_bits != 0;
                    if (_setp_28) {
                        float _rcp_29 = approx_rcp(scale1);
                        inv_scale1 = _rcp_29;
                    }
                    float2 oa_quant0 = make_float2(0.0f, 0.0f);
                    float2 oa_quant1 = make_float2(0.0f, 0.0f);
                    oa_quant0 = mul_f32x2(make_float2(value00, value10), make_float2(inv_scale0, inv_scale0));
                    oa_quant1 = mul_f32x2(make_float2(value01, value11), make_float2(inv_scale1, inv_scale1));
                    float oa_quant_scale_values = sc / sg;
                    float2 oa_quant_scale2 = make_float2(oa_quant_scale_values, oa_quant_scale_values);
                    oa_quant0 = mul_f32x2(oa_quant0, oa_quant_scale2);
                    oa_quant1 = mul_f32x2(oa_quant1, oa_quant_scale2);
                    quant_pair[0] = oa_quant0.x;
                    quant_pair[1] = oa_quant0.y;
                    uint32_t _thresh_mask_30;
                    {
                        int _lim_31 = 2;
                        if (_lim_31 <= 0) { _thresh_mask_30 = 0u; }
                        else if (_lim_31 >= 8) { _thresh_mask_30 = ((1u << 8) - 1u); }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_30) : "r"(_lim_31));
                        }
                    }
                    uint32_t _thresh_mask_32;
                    {
                        int _lim_33 = 8;
                        if (_lim_33 <= 0) { _thresh_mask_32 = 0u; }
                        else if (_lim_33 >= 8) { _thresh_mask_32 = ((1u << 8) - 1u); }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_32) : "r"(_lim_33));
                        }
                    }
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 0))) quant_pair[0] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 1))) quant_pair[1] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 2))) quant_pair[2] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 3))) quant_pair[3] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 4))) quant_pair[4] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 5))) quant_pair[5] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 6))) quant_pair[6] = 0.0f;
                    if (!((_thresh_mask_30 | ~_thresh_mask_32) & (1u << 7))) quant_pair[7] = 0.0f;
                    unsigned int _fp4_0[1];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                    quant_pair[0] = oa_quant1.x;
                    quant_pair[1] = oa_quant1.y;
                    unsigned int _fp4_1[1];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                    if (lane_1 < 4) {
                        int sf_feature = m_tile * 4 + (unsigned int)warp_0;
                        int sf_token_group = token0 / 8;
                        int sf_tile_stride = M_out / 64 * 32;
                        int sf_base = n_tile * (unsigned int)sf_tile_stride + (unsigned int)(sf_token_group * (M_out / 64) * 32) + (unsigned int)(sf_feature / 4 * 32) + (unsigned int)(token0 % 8 * 4) + (unsigned int)(sf_feature % 4);
                        if (token0 < valid_rows) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_34;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_34) : "f"(0.0f), "f"(scale0));
                            *reinterpret_cast<unsigned char*>(SFC + sf_base) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_34));
                        }
                        if (token1 < valid_rows) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_35;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_35) : "f"(0.0f), "f"(scale1));
                            *reinterpret_cast<unsigned char*>(SFC + (sf_base + 4)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_35));
                        }
                    }
                    int smem_flat0 = token0 * 64 + base_row;
                    int smem_flat1 = token1 * 64 + base_row;
                    int smem_index0 = smem_flat0 / 2 ^ smem_flat0 / 256 % 2 * 16;
                    int smem_index1 = smem_flat1 / 2 ^ smem_flat1 / 256 % 2 * 16;
                    epi_staging[smem_index0] = _fp4_0[0];
                    epi_staging[smem_index1] = _fp4_1[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (8 - valid_rows % 8) % 8;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 8 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(mma_free_addr + (acc_stage) * 8);
                }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
                int response_index = work_stage * 4;
                valid = work_response[response_index];
                next_x = work_response[response_index + 1];
                next_y = work_response[response_index + 2];
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_0 = valid;
                m_tile = next_x;
                n_tile = next_y;
                if (valid_0 == 0) {
                    break;
                }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            const int lane_0 = lane;
            const int tmem_lane_base_1 = warp % 4 * 32;
            unsigned int word[1];
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int q = 0; q < 8; q++) {
                        word[0] = 0;
                        if (lane_0 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[0])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)(lane_0 / 8 * 256) + (unsigned int)(q * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(tmem_lane_base_1 << 16) + 16 + 160 + stage * 16 + (unsigned int)(q * 2)), "r"(word[0]));
                    }
                    mbarrier_arrive(sfb_free_addr + (stage) * 8);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
                        }
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_sfb_full ^= 1; _phase_k_done ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
                int response_index_1 = work_stage_1 * 4;
                valid_1 = work_response[response_index_1];
                next_x_1 = work_response[response_index_1 + 1];
                next_y_1 = work_response[response_index_1 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_0_1 = valid_1;
                m_tile_1 = next_x_1;
                n_tile_1 = next_y_1;
                if (valid_0_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 8 && warp <= 9) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            int local_thread = (warp - 8) * 32 + lane;
            int row_stride_bytes = K / 2;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                int valid_rows_1 = (unsigned int)smem_v10[n_tile_2] - n_tile_2 * (unsigned int)BLOCK_N;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                    int dst_base = smem_b_addr + stage_1 * 2048;
                    for (int row_group = 0; row_group < 1; row_group++) {
                        int elt_offset = local_thread * 32 + row_group * 2048;
                        int row_1 = elt_offset / 256;
                        int col = elt_offset % 256;
                        int routed = 0;
                        if (row_1 < valid_rows_1) {
                            routed = smem_v13[smem_v14[n_tile_2] + row_1];
                        }
                        int src_base = routed * row_stride_bytes + iter_k * 256 + col / 2;
                        int dst_chunk = elt_offset / 2 ^ row_1 % 8 * 16;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(dst_base + dst_chunk), "l"(B + src_base), "r"((row_1 < valid_rows_1) ? 16 : 0));
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(dst_base + 1024 + dst_chunk), "l"(B + (src_base + 128)), "r"((row_1 < valid_rows_1) ? 16 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(b_full_addr + (stage_1) * 8) : "memory");
                    asm volatile("barrier.sync 1, 64;" ::: "memory");
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive(b_full_addr + (stage_1) * 8);
                        }
                    }
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                unsigned int valid_2 = 0;
                unsigned int next_x_2 = 0;
                unsigned int next_y_2 = 0;
                int response_index_2 = work_stage_2 * 4;
                valid_2 = work_response[response_index_2];
                next_x_2 = work_response[response_index_2 + 1];
                next_y_2 = work_response[response_index_2 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_2 = valid_2;
                m_tile_2 = next_x_2;
                n_tile_2 = next_y_2;
                if (valid_0_2 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 10) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            const int lane_0_1 = lane;
            int block4 = lane_0_1 % 8;
            int row0 = lane_0_1 / 8;
            int sf_stride = K / 16;
            unsigned int _phase_sfb_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m * grid_n; _tile_iter_3++) {
                if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                int valid_rows_2 = (unsigned int)smem_v10[n_tile_3] - n_tile_3 * (unsigned int)BLOCK_N;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
                    int sf_col = iter_k_1 * 32 + block4 * 4;
                    for (int row_group_1 = 0; row_group_1 < 2; row_group_1++) {
                        int row_2 = row0 + row_group_1 * 4;
                        int routed_1 = 0;
                        if (row_2 < valid_rows_2) {
                            routed_1 = smem_v13[smem_v14[n_tile_3] + row_2];
                        }
                        int dst_offset = row_2 / 8 * 256 + block4 * 32 + row_2 % 8 * 4;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_2 * 256 + (unsigned int)dst_offset), "l"(SFB + (routed_1 * sf_stride + sf_col)), "r"((row_2 < valid_rows_2) ? 4 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(sfb_full_addr + (stage_2) * 8) : "memory");
                    asm volatile("barrier.sync 2, 32;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
                    }
                    stage_2 += 1;
                    if (stage_2 == 5) { stage_2 = 0; _phase_sfb_free ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
                int response_index_3 = work_stage_3 * 4;
                valid_3 = work_response[response_index_3];
                next_x_3 = work_response[response_index_3 + 1];
                next_y_3 = work_response[response_index_3 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_3 = valid_3;
                m_tile_3 = next_x_3;
                n_tile_3 = next_y_3;
                if (valid_0_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 11) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_2 = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m * grid_n; _tile_iter_4++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                int expert_1 = smem_v9[n_tile_4];
                mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                    mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_a_addr + stage_3 * 32768), "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(iter_k_2 * 2), "r"(expert_1),
                               "r"(a_full_addr + (stage_3) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_a_addr + stage_3 * 32768 + 16384), "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(iter_k_2 * 2 + 1), "r"(expert_1),
                               "r"(a_full_addr + (stage_3) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 32768);
                    }
                    stage_3 += 1;
                    if (stage_3 == 5) { stage_3 = 0; _phase_k_done_2 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                unsigned int valid_4 = 0;
                unsigned int next_x_4 = 0;
                unsigned int next_y_4 = 0;
                int response_index_4 = work_stage_4 * 4;
                valid_4 = work_response[response_index_4];
                next_x_4 = work_response[response_index_4 + 1];
                next_y_4 = work_response[response_index_4 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                unsigned int valid_0_4 = valid_4;
                m_tile_4 = next_x_4;
                n_tile_4 = next_y_4;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 12) {
        { // load_sfa_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_sfa_free = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m * grid_n; _tile_iter_5++) {
                if (m_tile_5 >= (unsigned int)grid_m || n_tile_5 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                int expert_2 = smem_v9[n_tile_5];
                #pragma unroll 1
                for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                    mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
                    int sf_tile = ((unsigned int)(expert_2 * grid_m) + m_tile_5) * (unsigned int)K_tiles + (unsigned int)iter_k_3;
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_sfa_addr + stage_4 * 4096), "l"((&SFA)), "r"(0), "r"(0), "r"(iter_k_3 * 8), "r"((unsigned int)(expert_2 * grid_m) + m_tile_5),
                               "r"(sfa_full_addr + (stage_4) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 4096);
                    }
                    stage_4 += 1;
                    if (stage_4 == 5) { stage_4 = 0; _phase_sfa_free ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
                int response_index_5 = work_stage_5 * 4;
                valid_5 = work_response[response_index_5];
                next_x_5 = work_response[response_index_5 + 1];
                next_y_5 = work_response[response_index_5 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_0_5 = valid_5;
                m_tile_5 = next_x_5;
                n_tile_5 = next_y_5;
                if (valid_0_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 13) {
        { // copy_sfa_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_5 = 0;
            unsigned int work_stage_6 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_3 = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m * grid_n; _tile_iter_6++) {
                if (m_tile_6 >= (unsigned int)grid_m || n_tile_6 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                #pragma unroll 1
                for (int _iter_k_1 = 0; _iter_k_1 < K_tiles; _iter_k_1++) {
                    mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
                    mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        for (int k_set = 0; k_set < 8; k_set++) {
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + (unsigned int)(k_set * 512))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + (unsigned int)(k_set * 4)))), "l"(_tcgen05_cp_desc_0)
                                    : "memory");
                            }
                        }
                    }
                    elect_commit2(tmem_sfa_full_addr + (stage_5) * 8, sfa_free_addr + (stage_5) * 8);
                    stage_5 += 1;
                    if (stage_5 == 5) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                unsigned int valid_6 = 0;
                unsigned int next_x_6 = 0;
                unsigned int next_y_6 = 0;
                int response_index_6 = work_stage_6 * 4;
                valid_6 = work_response[response_index_6];
                next_x_6 = work_response[response_index_6 + 1];
                next_y_6 = work_response[response_index_6 + 2];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_6 = valid_6;
                m_tile_6 = next_x_6;
                n_tile_6 = next_y_6;
                if (valid_0_6 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 14) {
        { // mma_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_6 = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_7 = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                if (m_tile_7 >= (unsigned int)grid_m || n_tile_7 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                unsigned int tile_work_id_1 = m_tile_7 * (unsigned int)grid_n + n_tile_7;
                bool tile_profile_1 = _tile_iter_7 < 2;
                mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                #pragma unroll 1
                for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                    unsigned int stage_record = _tile_iter_7 * (unsigned int)K_tiles + (unsigned int)iter_k_4;
                    unsigned int stage_work_id = tile_work_id_1 * (unsigned int)K_tiles + (unsigned int)iter_k_4;
                    mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
                    mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_a_full);
                    mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_a_full);
                    mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_a_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + stage_6 * 32 + 0, (unsigned int)tmem_sfb + stage_6 * 16 + 0, ((((iter_k_4 == 0) ? 1 : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 4) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 2) + 0, 1);
                        }
                    }
                    int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 8) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 4) + 0, 1);
                        }
                    }
                    int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 12) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 6) + 0, 1);
                        }
                    }
                    int _mma_a_lo_4 = make_warp_uniform((((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 16) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 8) + 0, 1);
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform((((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_b_addr + 1056) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 20) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 10) + 0, 1);
                        }
                    }
                    int _mma_a_lo_6 = make_warp_uniform((((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_b_addr + 1088) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 24) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 12) + 0, 1);
                        }
                    }
                    int _mma_a_lo_7 = make_warp_uniform((((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_b_addr + 1120) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 28) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 14) + 0, 1);
                        }
                    }
                    if (iter_k_4 + 1 == K_tiles) {
                        elect_commit2(k_done_addr + (stage_6) * 8, mma_full_addr + (acc_stage_1) * 8);
                    } else {
                        elect_commit(k_done_addr + (stage_6) * 8);
                    }
                    stage_6 += 1;
                    if (stage_6 == 5) { stage_6 = 0; _phase_a_full ^= 1; }
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
                int response_index_7 = work_stage_7 * 4;
                valid_7 = work_response[response_index_7];
                next_x_7 = work_response[response_index_7 + 1];
                next_y_7 = work_response[response_index_7 + 2];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                unsigned int valid_0_7 = valid_7;
                m_tile_7 = next_x_7;
                n_tile_7 = next_y_7;
                if (valid_0_7 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 15) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < grid_m * grid_n; _tile_iter_8++) {
                if (m_tile_8 >= (unsigned int)grid_m || n_tile_8 >= (unsigned int)smem_v8[0]) {
                    break;
                }
                mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                    int next_linear = atomicAdd(work_counter, 1);
                    int response_index_8 = work_stage_8 * 4;
                    work_response[response_index_8] = next_linear < grid_m * smem_v8[0];
                    work_response[response_index_8 + 1] = next_linear % grid_m;
                    work_response[response_index_8 + 2] = next_linear / grid_m;
                    work_response[response_index_8 + 3] = 0;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_full_addr + (work_stage_8) * 8);
                }
                __syncwarp();
                mbarrier_wait(work_full_addr + (work_stage_8) * 8, _phase_work_full_8);
                unsigned int valid_8 = 0;
                unsigned int next_x_8 = 0;
                unsigned int next_y_8 = 0;
                int response_index_9 = work_stage_8 * 4;
                valid_8 = work_response[response_index_9];
                next_x_8 = work_response[response_index_9 + 1];
                next_y_8 = work_response[response_index_9 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                unsigned int valid_0_8 = valid_8;
                m_tile_8 = next_x_8;
                n_tile_8 = next_y_8;
                if (valid_0_8 == 0) {
                    break;
                }
            }
            if (elect_sync()) {
                __threadfence();
                int finished = atomicAdd(done_counter, 1);
                if (finished == launch_ctas - 1) {
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    work_counter[0] = initial_work;
                    done_counter[0] = 0;
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
