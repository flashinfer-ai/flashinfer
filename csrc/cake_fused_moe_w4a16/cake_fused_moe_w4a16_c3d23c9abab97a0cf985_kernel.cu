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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define TMEM_A_BF16_OFFSET 128
#define NUM_RAW_RING_STAGES 4
#define NUM_B_RING_STAGES 4
#define NUM_A_RING_STAGES 3
#define NUM_ACC_RING_STAGES 2
#define NUM_TILE_RING_STAGES 3
#define NUM_STORE_RING_STAGES 2
#define SMEM_RAW_SMEM_OFF 1024
#define SMEM_RAW_SMEM_STAGE_BYTES 16384
#define SMEM_RAW_SMEM_STRIDE 16384
#define SMEM_SF_SMEM_OFF 66560
#define SMEM_SF_SMEM_STAGE_BYTES 2048
#define SMEM_SF_SMEM_STRIDE 2048
#define SMEM_SMEM_B_OFF 74752
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_STAGING_OFF 205824
#define SMEM_STAGING_STAGE_BYTES 4096
#define SMEM_STAGING_STRIDE 4096
#define SMEM_TILE_INFO_OFF 214016
#define SMEM_TILE_INFO_STAGE_BYTES 16
#define SMEM_TILE_INFO_STRIDE 16
#define SMEM_TOTAL 214144
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

__device__ __forceinline__ uint32_t cvt_bf16x2_e4m3x2_p112(uint16_t src) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
#else
    uint32_t _f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(src));
    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
    float _f0;
    float _f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(dst) : "f"(_f1), "f"(_f0));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(dst), "r"(0x77807780u));
#endif
    return dst;
}

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2_prescaled(uint32_t src) {
    uint16_t _fp4_u16 = (uint16_t)(src & 0xFFu);
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
        : "=r"(dst) : "h"(_fp4_u16));
#else
    uint32_t _fp4_f16x2;
    asm("{ .reg .b8 _fp4b, _fp4z0, _fp4z1, _fp4z2;      \n\t"
        "  mov.b32 {_fp4b, _fp4z0, _fp4z1, _fp4z2}, %1;   \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;              }"
        : "=r"(_fp4_f16x2) : "r"((uint32_t)_fp4_u16));
    dst = ((_fp4_f16x2 & 0x7FFF7FFFu) >> 3) | (_fp4_f16x2 & 0x80008000u);
#endif
    return dst;
}

__device__ __forceinline__ void cvt_bf16x8_e2m1x16_e4m3(uint32_t packed0, uint32_t packed1, uint32_t scale_byte, uint32_t (&dst)[8]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7, _sb, _sz0, _sz1, _sz2; \n\t"
        "  .reg .b16 _sx2;                            \n\t"
        "  .reg .b32 _sc;                             \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %8;          \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %9;          \n\t"
        "  mov.b32 {_sb, _sz0, _sz1, _sz2}, %10;      \n\t"
        "  mov.b16 _sx2, {_sb, _sb};                  \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _sx2;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _sc;            \n\t"
        "  mul.rn.bf16x2 %1, %1, _sc;            \n\t"
        "  mul.rn.bf16x2 %2, %2, _sc;            \n\t"
        "  mul.rn.bf16x2 %3, %3, _sc;            \n\t"
        "  mul.rn.bf16x2 %4, %4, _sc;            \n\t"
        "  mul.rn.bf16x2 %5, %5, _sc;            \n\t"
        "  mul.rn.bf16x2 %6, %6, _sc;            \n\t"
        "  mul.rn.bf16x2 %7, %7, _sc;            \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(packed0), "r"(packed1), "r"(scale_byte));
#else
    uint32_t _scale_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(scale_byte | (scale_byte << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _lo = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _hi = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_lo), "r"(_scale_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_hi), "r"(_scale_p112));
    }
#endif
}

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







__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fused_moe_w4a16_c3d23c9abab97a0cf985(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap S, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, float* __restrict__ alpha, float swiglu_limit, int K, int num_m_tiles, CakeFastDivmod m_tile_div)
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
    #define raw_full_addr (mbar_base + 0)
    #define raw_empty_addr (mbar_base + 32)
    #define b_full_addr (mbar_base + 64)
    #define b_empty_addr (mbar_base + 96)
    #define a_full_addr (mbar_base + 128)
    #define a_empty_addr (mbar_base + 152)
    #define acc_full_addr (mbar_base + 176)
    #define acc_empty_addr (mbar_base + 192)
    #define tile_full_addr (mbar_base + 208)
    #define tile_empty_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* raw_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int raw_smem_addr = smem + 1024;
    uint8_t* sf_smem = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int sf_smem_addr = smem + 66560;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_b_addr = smem + 74752;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int staging_addr = smem + 205824;
    int* tile_info = reinterpret_cast<int*>(smem_raw + 214016);
    const int tile_info_addr = smem + 214016;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'raw_ring' ---
            // raw_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // raw_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'b_ring' ---
            // b_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // b_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'a_ring' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // --- pipeline 'acc_ring' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // acc_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 192, 4);
            mbarrier_init(smem + 200, 4);
            // --- pipeline 'tile_ring' ---
            // tile_full: 3 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // tile_empty: 3 barriers, init_count=5
            mbarrier_init(smem + 232, 5);
            mbarrier_init(smem + 240, 5);
            mbarrier_init(smem + 248, 5);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_a_bf16 = taddr + 128;
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // epilogue_main
            int valid_tiles = num_non_exiting_tiles[0];
            int end = valid_tiles * num_m_tiles;
            const int quarter = warp % 4;
            const unsigned int gate_lanes = (unsigned int)(quarter * 32) << 16;
            const unsigned int up_lanes = (unsigned int)(quarter * 32 + 16) << 16;
            const int thread_off = lane / 2 % 2 * 128 + (quarter * 16 + lane / 4 + lane % 2 * 8) * 2;
            float _neg_0;
            asm("neg.f32 %0, %1;" : "=f"(_neg_0) : "f"(swiglu_limit));
            const float neg_limit = _neg_0;
            unsigned int tile_cur = 0;
            unsigned int acc_cur = 0;
            unsigned int out_cur = 0;
            int record[2];
            unsigned int _phase_tile_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int w = bid; w < end; w += num_bids) {
                uint32_t _fast_div_q_3 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w);
                uint32_t _fast_div_r_3 = (uint32_t)(w) - _fast_div_q_3 * (uint32_t)(m_tile_div.divisor);
                int t = (int)_fast_div_q_3;
                int m_cta = (int)_fast_div_r_3;
                mbarrier_wait(tile_full_addr + (tile_cur) * 8, _phase_tile_full);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&record[0])), "=r"(*reinterpret_cast<uint32_t*>(&record[(0) + 1]))
                    : "r"(tile_info_addr + tile_cur * 16));
                int expert = record[0];
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tile_empty_addr + (tile_cur) * 8);
                }
                tile_cur += 1;
                if (tile_cur == 3) { tile_cur = 0; _phase_tile_full ^= 1; }
                float alpha_e = alpha[expert];
                int feature0 = m_cta * 64;
                int row0 = t * 64;
                mbarrier_wait(acc_full_addr + (acc_cur) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int sub = 0; sub < 2; sub++) {
                    unsigned int col = acc_cur * 64 + (unsigned int)(sub * 32);
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(taddr + gate_lanes + col));
                    float _tmem_load_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                        : "r"(taddr + up_lanes + col));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (sub == 1) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            mbarrier_arrive(acc_empty_addr + (acc_cur) * 8);
                        }
                    }
                    unsigned int out_base = staging_addr + out_cur * 4096 + (unsigned int)thread_off;
                    float g;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g) : "f"(_tmem_load_0[0]), "f"(alpha_e));
                    float u;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(_tmem_load_1[0]), "f"(alpha_e));
                    float _min_1;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_1) : "f"(g), "f"(swiglu_limit));
                    g = _min_1;
                    float _min_3;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(u), "f"(swiglu_limit));
                    float _max_2;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(_min_3), "f"(neg_limit));
                    u = _max_2;
                    float _exp2_6 = approx_exp2(mul_f32_rn_noftz(g, -1.442695f));
                    float _add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(_exp2_6), "f"(1.0f));
                    float _rcp_4 = approx_rcp(_add_5);
                    {
                        __nv_bfloat16 _bval_7 = __float2bfloat16_rn(mul_f32_rn_noftz(u, mul_f32_rn_noftz(g, _rcp_4)));
                        uint16_t _bits_7 = *(uint16_t*)&_bval_7;
                        uint32_t _addr_7 = static_cast<uint32_t>(out_base);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7) : "memory");
                    }
                    float g_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_0) : "f"(_tmem_load_0[1]), "f"(alpha_e));
                    float u_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_1) : "f"(_tmem_load_1[1]), "f"(alpha_e));
                    float _min_8;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_8) : "f"(g_0), "f"(swiglu_limit));
                    g_0 = _min_8;
                    float _min_10;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_10) : "f"(u_1), "f"(swiglu_limit));
                    float _max_9;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(_min_10), "f"(neg_limit));
                    u_1 = _max_9;
                    float _exp2_13 = approx_exp2(mul_f32_rn_noftz(g_0, -1.442695f));
                    float _add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_exp2_13), "f"(1.0f));
                    float _rcp_11 = approx_rcp(_add_12);
                    {
                        __nv_bfloat16 _bval_14 = __float2bfloat16_rn(mul_f32_rn_noftz(u_1, mul_f32_rn_noftz(g_0, _rcp_11)));
                        uint16_t _bits_14 = *(uint16_t*)&_bval_14;
                        uint32_t _addr_14 = static_cast<uint32_t>(out_base + 256);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14) : "memory");
                    }
                    float g_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_2) : "f"(_tmem_load_0[2]), "f"(alpha_e));
                    float u_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_3) : "f"(_tmem_load_1[2]), "f"(alpha_e));
                    float _min_15;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(g_2), "f"(swiglu_limit));
                    g_2 = _min_15;
                    float _min_17;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_17) : "f"(u_3), "f"(swiglu_limit));
                    float _max_16;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(_min_17), "f"(neg_limit));
                    u_3 = _max_16;
                    float _exp2_20 = approx_exp2(mul_f32_rn_noftz(g_2, -1.442695f));
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_exp2_20), "f"(1.0f));
                    float _rcp_18 = approx_rcp(_add_19);
                    {
                        __nv_bfloat16 _bval_21 = __float2bfloat16_rn(mul_f32_rn_noftz(u_3, mul_f32_rn_noftz(g_2, _rcp_18)));
                        uint16_t _bits_21 = *(uint16_t*)&_bval_21;
                        uint32_t _addr_21 = static_cast<uint32_t>(out_base + 512);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_21), "h"(_bits_21) : "memory");
                    }
                    float g_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_4) : "f"(_tmem_load_0[3]), "f"(alpha_e));
                    float u_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_5) : "f"(_tmem_load_1[3]), "f"(alpha_e));
                    float _min_22;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_22) : "f"(g_4), "f"(swiglu_limit));
                    g_4 = _min_22;
                    float _min_24;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(u_5), "f"(swiglu_limit));
                    float _max_23;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_23) : "f"(_min_24), "f"(neg_limit));
                    u_5 = _max_23;
                    float _exp2_27 = approx_exp2(mul_f32_rn_noftz(g_4, -1.442695f));
                    float _add_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(_exp2_27), "f"(1.0f));
                    float _rcp_25 = approx_rcp(_add_26);
                    {
                        __nv_bfloat16 _bval_28 = __float2bfloat16_rn(mul_f32_rn_noftz(u_5, mul_f32_rn_noftz(g_4, _rcp_25)));
                        uint16_t _bits_28 = *(uint16_t*)&_bval_28;
                        uint32_t _addr_28 = static_cast<uint32_t>(out_base + 768);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_28), "h"(_bits_28) : "memory");
                    }
                    float g_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_6) : "f"(_tmem_load_0[4]), "f"(alpha_e));
                    float u_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_7) : "f"(_tmem_load_1[4]), "f"(alpha_e));
                    float _min_29;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_29) : "f"(g_6), "f"(swiglu_limit));
                    g_6 = _min_29;
                    float _min_31;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_31) : "f"(u_7), "f"(swiglu_limit));
                    float _max_30;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(_min_31), "f"(neg_limit));
                    u_7 = _max_30;
                    float _exp2_34 = approx_exp2(mul_f32_rn_noftz(g_6, -1.442695f));
                    float _add_33;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_exp2_34), "f"(1.0f));
                    float _rcp_32 = approx_rcp(_add_33);
                    {
                        __nv_bfloat16 _bval_35 = __float2bfloat16_rn(mul_f32_rn_noftz(u_7, mul_f32_rn_noftz(g_6, _rcp_32)));
                        uint16_t _bits_35 = *(uint16_t*)&_bval_35;
                        uint32_t _addr_35 = static_cast<uint32_t>(out_base + 1024);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_35), "h"(_bits_35) : "memory");
                    }
                    float g_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_8) : "f"(_tmem_load_0[5]), "f"(alpha_e));
                    float u_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_9) : "f"(_tmem_load_1[5]), "f"(alpha_e));
                    float _min_36;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(g_8), "f"(swiglu_limit));
                    g_8 = _min_36;
                    float _min_38;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_38) : "f"(u_9), "f"(swiglu_limit));
                    float _max_37;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(_min_38), "f"(neg_limit));
                    u_9 = _max_37;
                    float _exp2_41 = approx_exp2(mul_f32_rn_noftz(g_8, -1.442695f));
                    float _add_40;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_exp2_41), "f"(1.0f));
                    float _rcp_39 = approx_rcp(_add_40);
                    {
                        __nv_bfloat16 _bval_42 = __float2bfloat16_rn(mul_f32_rn_noftz(u_9, mul_f32_rn_noftz(g_8, _rcp_39)));
                        uint16_t _bits_42 = *(uint16_t*)&_bval_42;
                        uint32_t _addr_42 = static_cast<uint32_t>(out_base + 1280);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_42), "h"(_bits_42) : "memory");
                    }
                    float g_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_10) : "f"(_tmem_load_0[6]), "f"(alpha_e));
                    float u_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_11) : "f"(_tmem_load_1[6]), "f"(alpha_e));
                    float _min_43;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(g_10), "f"(swiglu_limit));
                    g_10 = _min_43;
                    float _min_45;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_45) : "f"(u_11), "f"(swiglu_limit));
                    float _max_44;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(_min_45), "f"(neg_limit));
                    u_11 = _max_44;
                    float _exp2_48 = approx_exp2(mul_f32_rn_noftz(g_10, -1.442695f));
                    float _add_47;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_exp2_48), "f"(1.0f));
                    float _rcp_46 = approx_rcp(_add_47);
                    {
                        __nv_bfloat16 _bval_49 = __float2bfloat16_rn(mul_f32_rn_noftz(u_11, mul_f32_rn_noftz(g_10, _rcp_46)));
                        uint16_t _bits_49 = *(uint16_t*)&_bval_49;
                        uint32_t _addr_49 = static_cast<uint32_t>(out_base + 1536);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_49), "h"(_bits_49) : "memory");
                    }
                    float g_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_12) : "f"(_tmem_load_0[7]), "f"(alpha_e));
                    float u_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_13) : "f"(_tmem_load_1[7]), "f"(alpha_e));
                    float _min_50;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_50) : "f"(g_12), "f"(swiglu_limit));
                    g_12 = _min_50;
                    float _min_52;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_52) : "f"(u_13), "f"(swiglu_limit));
                    float _max_51;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_51) : "f"(_min_52), "f"(neg_limit));
                    u_13 = _max_51;
                    float _exp2_55 = approx_exp2(mul_f32_rn_noftz(g_12, -1.442695f));
                    float _add_54;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(_exp2_55), "f"(1.0f));
                    float _rcp_53 = approx_rcp(_add_54);
                    {
                        __nv_bfloat16 _bval_56 = __float2bfloat16_rn(mul_f32_rn_noftz(u_13, mul_f32_rn_noftz(g_12, _rcp_53)));
                        uint16_t _bits_56 = *(uint16_t*)&_bval_56;
                        uint32_t _addr_56 = static_cast<uint32_t>(out_base + 1792);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_56), "h"(_bits_56) : "memory");
                    }
                    float g_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_14) : "f"(_tmem_load_0[8]), "f"(alpha_e));
                    float u_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_15) : "f"(_tmem_load_1[8]), "f"(alpha_e));
                    float _min_57;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(g_14), "f"(swiglu_limit));
                    g_14 = _min_57;
                    float _min_59;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_59) : "f"(u_15), "f"(swiglu_limit));
                    float _max_58;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(_min_59), "f"(neg_limit));
                    u_15 = _max_58;
                    float _exp2_62 = approx_exp2(mul_f32_rn_noftz(g_14, -1.442695f));
                    float _add_61;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(_exp2_62), "f"(1.0f));
                    float _rcp_60 = approx_rcp(_add_61);
                    {
                        __nv_bfloat16 _bval_63 = __float2bfloat16_rn(mul_f32_rn_noftz(u_15, mul_f32_rn_noftz(g_14, _rcp_60)));
                        uint16_t _bits_63 = *(uint16_t*)&_bval_63;
                        uint32_t _addr_63 = static_cast<uint32_t>(out_base + 2048);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_63), "h"(_bits_63) : "memory");
                    }
                    float g_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_16) : "f"(_tmem_load_0[9]), "f"(alpha_e));
                    float u_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_17) : "f"(_tmem_load_1[9]), "f"(alpha_e));
                    float _min_64;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(g_16), "f"(swiglu_limit));
                    g_16 = _min_64;
                    float _min_66;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_66) : "f"(u_17), "f"(swiglu_limit));
                    float _max_65;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_65) : "f"(_min_66), "f"(neg_limit));
                    u_17 = _max_65;
                    float _exp2_69 = approx_exp2(mul_f32_rn_noftz(g_16, -1.442695f));
                    float _add_68;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_exp2_69), "f"(1.0f));
                    float _rcp_67 = approx_rcp(_add_68);
                    {
                        __nv_bfloat16 _bval_70 = __float2bfloat16_rn(mul_f32_rn_noftz(u_17, mul_f32_rn_noftz(g_16, _rcp_67)));
                        uint16_t _bits_70 = *(uint16_t*)&_bval_70;
                        uint32_t _addr_70 = static_cast<uint32_t>(out_base + 2304);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_70), "h"(_bits_70) : "memory");
                    }
                    float g_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_18) : "f"(_tmem_load_0[10]), "f"(alpha_e));
                    float u_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_19) : "f"(_tmem_load_1[10]), "f"(alpha_e));
                    float _min_71;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_71) : "f"(g_18), "f"(swiglu_limit));
                    g_18 = _min_71;
                    float _min_73;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_73) : "f"(u_19), "f"(swiglu_limit));
                    float _max_72;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_72) : "f"(_min_73), "f"(neg_limit));
                    u_19 = _max_72;
                    float _exp2_76 = approx_exp2(mul_f32_rn_noftz(g_18, -1.442695f));
                    float _add_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(_exp2_76), "f"(1.0f));
                    float _rcp_74 = approx_rcp(_add_75);
                    {
                        __nv_bfloat16 _bval_77 = __float2bfloat16_rn(mul_f32_rn_noftz(u_19, mul_f32_rn_noftz(g_18, _rcp_74)));
                        uint16_t _bits_77 = *(uint16_t*)&_bval_77;
                        uint32_t _addr_77 = static_cast<uint32_t>(out_base + 2560);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_77), "h"(_bits_77) : "memory");
                    }
                    float g_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_20) : "f"(_tmem_load_0[11]), "f"(alpha_e));
                    float u_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_21) : "f"(_tmem_load_1[11]), "f"(alpha_e));
                    float _min_78;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_78) : "f"(g_20), "f"(swiglu_limit));
                    g_20 = _min_78;
                    float _min_80;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_80) : "f"(u_21), "f"(swiglu_limit));
                    float _max_79;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_79) : "f"(_min_80), "f"(neg_limit));
                    u_21 = _max_79;
                    float _exp2_83 = approx_exp2(mul_f32_rn_noftz(g_20, -1.442695f));
                    float _add_82;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_exp2_83), "f"(1.0f));
                    float _rcp_81 = approx_rcp(_add_82);
                    {
                        __nv_bfloat16 _bval_84 = __float2bfloat16_rn(mul_f32_rn_noftz(u_21, mul_f32_rn_noftz(g_20, _rcp_81)));
                        uint16_t _bits_84 = *(uint16_t*)&_bval_84;
                        uint32_t _addr_84 = static_cast<uint32_t>(out_base + 2816);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_84), "h"(_bits_84) : "memory");
                    }
                    float g_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_22) : "f"(_tmem_load_0[12]), "f"(alpha_e));
                    float u_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_23) : "f"(_tmem_load_1[12]), "f"(alpha_e));
                    float _min_85;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(g_22), "f"(swiglu_limit));
                    g_22 = _min_85;
                    float _min_87;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_87) : "f"(u_23), "f"(swiglu_limit));
                    float _max_86;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_86) : "f"(_min_87), "f"(neg_limit));
                    u_23 = _max_86;
                    float _exp2_90 = approx_exp2(mul_f32_rn_noftz(g_22, -1.442695f));
                    float _add_89;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(_exp2_90), "f"(1.0f));
                    float _rcp_88 = approx_rcp(_add_89);
                    {
                        __nv_bfloat16 _bval_91 = __float2bfloat16_rn(mul_f32_rn_noftz(u_23, mul_f32_rn_noftz(g_22, _rcp_88)));
                        uint16_t _bits_91 = *(uint16_t*)&_bval_91;
                        uint32_t _addr_91 = static_cast<uint32_t>(out_base + 3072);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_91), "h"(_bits_91) : "memory");
                    }
                    float g_24;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_24) : "f"(_tmem_load_0[13]), "f"(alpha_e));
                    float u_25;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_25) : "f"(_tmem_load_1[13]), "f"(alpha_e));
                    float _min_92;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_92) : "f"(g_24), "f"(swiglu_limit));
                    g_24 = _min_92;
                    float _min_94;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_94) : "f"(u_25), "f"(swiglu_limit));
                    float _max_93;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_93) : "f"(_min_94), "f"(neg_limit));
                    u_25 = _max_93;
                    float _exp2_97 = approx_exp2(mul_f32_rn_noftz(g_24, -1.442695f));
                    float _add_96;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_exp2_97), "f"(1.0f));
                    float _rcp_95 = approx_rcp(_add_96);
                    {
                        __nv_bfloat16 _bval_98 = __float2bfloat16_rn(mul_f32_rn_noftz(u_25, mul_f32_rn_noftz(g_24, _rcp_95)));
                        uint16_t _bits_98 = *(uint16_t*)&_bval_98;
                        uint32_t _addr_98 = static_cast<uint32_t>(out_base + 3328);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_98), "h"(_bits_98) : "memory");
                    }
                    float g_26;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_26) : "f"(_tmem_load_0[14]), "f"(alpha_e));
                    float u_27;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_27) : "f"(_tmem_load_1[14]), "f"(alpha_e));
                    float _min_99;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_99) : "f"(g_26), "f"(swiglu_limit));
                    g_26 = _min_99;
                    float _min_101;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_101) : "f"(u_27), "f"(swiglu_limit));
                    float _max_100;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_100) : "f"(_min_101), "f"(neg_limit));
                    u_27 = _max_100;
                    float _exp2_104 = approx_exp2(mul_f32_rn_noftz(g_26, -1.442695f));
                    float _add_103;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_exp2_104), "f"(1.0f));
                    float _rcp_102 = approx_rcp(_add_103);
                    {
                        __nv_bfloat16 _bval_105 = __float2bfloat16_rn(mul_f32_rn_noftz(u_27, mul_f32_rn_noftz(g_26, _rcp_102)));
                        uint16_t _bits_105 = *(uint16_t*)&_bval_105;
                        uint32_t _addr_105 = static_cast<uint32_t>(out_base + 3584);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_105), "h"(_bits_105) : "memory");
                    }
                    float g_28;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_28) : "f"(_tmem_load_0[15]), "f"(alpha_e));
                    float u_29;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_29) : "f"(_tmem_load_1[15]), "f"(alpha_e));
                    float _min_106;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_106) : "f"(g_28), "f"(swiglu_limit));
                    g_28 = _min_106;
                    float _min_108;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_108) : "f"(u_29), "f"(swiglu_limit));
                    float _max_107;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_107) : "f"(_min_108), "f"(neg_limit));
                    u_29 = _max_107;
                    float _exp2_111 = approx_exp2(mul_f32_rn_noftz(g_28, -1.442695f));
                    float _add_110;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(_exp2_111), "f"(1.0f));
                    float _rcp_109 = approx_rcp(_add_110);
                    {
                        __nv_bfloat16 _bval_112 = __float2bfloat16_rn(mul_f32_rn_noftz(u_29, mul_f32_rn_noftz(g_28, _rcp_109)));
                        uint16_t _bits_112 = *(uint16_t*)&_bval_112;
                        uint32_t _addr_112 = static_cast<uint32_t>(out_base + 3840);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_112), "h"(_bits_112) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2}], [%3], %4;"
                                :: "l"((&C)), "r"(feature0), "r"(row0 + sub * 32), "r"(staging_addr + out_cur * 4096), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    out_cur += 1;
                    if (out_cur == 2) { out_cur = 0; }
                }
                acc_cur += 1;
                if (acc_cur == 2) { acc_cur = 0; _phase_acc_full ^= 1; }
            }
            if (warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int valid_tiles_1 = num_non_exiting_tiles[0];
            int end_1 = valid_tiles_1 * num_m_tiles;
            int num_k = K / 256;
            unsigned int a_cur = 0;
            unsigned int b_cur = 0;
            unsigned int acc_cur_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int _w = bid; _w < end_1; _w += num_bids) {
                    mbarrier_wait(acc_empty_addr + (acc_cur_1) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kt = 0; kt < num_k; kt++) {
                        mbarrier_wait(a_full_addr + (a_cur) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (b_cur) * 8, _phase_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (b_cur) * 2048;
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
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_acc + (acc_cur_1 * 64))), "r"(_smem_desc_lo_0), "r"((unsigned int)tmem_a_bf16 + a_cur * 128), "r"(((((kt == 0) ? 1 : 0)) ? 0 : 1)) : "memory");
                        tcgen05_commit(b_empty_addr + (b_cur) * 8);
                        tcgen05_commit(a_empty_addr + (a_cur) * 8);
                        a_cur += 1;
                        if (a_cur == 3) { a_cur = 0; _phase_a_full ^= 1; }
                        b_cur += 1;
                        if (b_cur == 4) { b_cur = 0; _phase_b_full ^= 1; }
                    }
                    tcgen05_commit(acc_full_addr + (acc_cur_1) * 8);
                    acc_cur_1 += 1;
                    if (acc_cur_1 == 2) { acc_cur_1 = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 5) {
        { // scheduler_main
            int valid_tiles_2 = num_non_exiting_tiles[0];
            int end_2 = valid_tiles_2 * num_m_tiles;
            unsigned int tile_cur_1 = 0;
            unsigned int _phase_tile_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_1 = bid; w_1 < end_2; w_1 += num_bids) {
                    uint32_t _fast_div_q_0 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w_1), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w_1);
                    uint32_t _fast_div_r_0 = (uint32_t)(w_1) - _fast_div_q_0 * (uint32_t)(m_tile_div.divisor);
                    int t_1 = (int)_fast_div_q_0;
                    int m_cta_1 = (int)_fast_div_r_0;
                    int expert_1 = tile_idx_to_expert_idx[t_1];
                    int valid_rows = tile_idx_to_mn_limit[t_1] - t_1 * 64;
                    mbarrier_wait(tile_empty_addr + (tile_cur_1) * 8, _phase_tile_empty);
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(tile_info_addr + tile_cur_1 * 16), "r"(expert_1), "r"(valid_rows) : "memory");
                    mbarrier_arrive(tile_full_addr + (tile_cur_1) * 8);
                    tile_cur_1 += 1;
                    if (tile_cur_1 == 3) { tile_cur_1 = 0; _phase_tile_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: weight_tma ----
    if (warp == 6) {
        { // weight_tma_main
            int valid_tiles_3 = num_non_exiting_tiles[0];
            int end_3 = valid_tiles_3 * num_m_tiles;
            const int features_i = num_m_tiles * 64;
            int num_k_1 = K / 256;
            unsigned int tile_cur_2 = 0;
            unsigned int raw_cur = 0;
            int record_1[2];
            unsigned int _phase_tile_full_1 = 0;
            unsigned int _phase_raw_empty = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (unsigned int w_2 = bid; w_2 < end_3; w_2 += num_bids) {
                    uint32_t _fast_div_q_1 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w_2), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w_2);
                    uint32_t _fast_div_r_1 = (uint32_t)(w_2) - _fast_div_q_1 * (uint32_t)(m_tile_div.divisor);
                    int t_2 = (int)_fast_div_q_1;
                    int m_cta_2 = (int)_fast_div_r_1;
                    mbarrier_wait(tile_full_addr + (tile_cur_2) * 8, _phase_tile_full_1);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&record_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_1[(0) + 1]))
                        : "r"(tile_info_addr + tile_cur_2 * 16));
                    int expert_2 = record_1[0];
                    mbarrier_arrive(tile_empty_addr + (tile_cur_2) * 8);
                    tile_cur_2 += 1;
                    if (tile_cur_2 == 3) { tile_cur_2 = 0; _phase_tile_full_1 ^= 1; }
                    int gate_row = m_cta_2 * 64;
                    int up_row = features_i + m_cta_2 * 64;
                    #pragma unroll 1
                    for (int kt_1 = 0; kt_1 < num_k_1; kt_1++) {
                        mbarrier_wait(raw_empty_addr + (raw_cur) * 8, _phase_raw_empty);
                        unsigned int raw_base = raw_smem_addr + raw_cur * 16384;
                        unsigned int sf_base = sf_smem_addr + raw_cur * 2048;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(raw_base), "l"((&W)), "r"(kt_1 * 128), "r"(gate_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(raw_base + 8192), "l"((&W)), "r"(kt_1 * 128), "r"(up_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sf_base), "l"((&S)), "r"(kt_1 * 16), "r"(gate_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sf_base + 1024), "l"((&S)), "r"(kt_1 * 16), "r"(up_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(raw_full_addr + (raw_cur) * 8, 18432);
                        raw_cur += 1;
                        if (raw_cur == 4) { raw_cur = 0; _phase_raw_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: token_tma ----
    if (warp == 7) {
        { // token_tma_main
            int valid_tiles_4 = num_non_exiting_tiles[0];
            int end_4 = valid_tiles_4 * num_m_tiles;
            int num_k_2 = K / 256;
            unsigned int b_cur_1 = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_b_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_3 = bid; w_3 < end_4; w_3 += num_bids) {
                    uint32_t _fast_div_q_2 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w_3), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w_3);
                    uint32_t _fast_div_r_2 = (uint32_t)(w_3) - _fast_div_q_2 * (uint32_t)(m_tile_div.divisor);
                    int t_3 = (int)_fast_div_q_2;
                    int m_cta_3 = (int)_fast_div_r_2;
                    int row0_1 = t_3 * 64;
                    #pragma unroll 1
                    for (int kt_2 = 0; kt_2 < num_k_2; kt_2++) {
                        mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_b_addr + b_cur_1 * 32768), "l"((&X)), "r"(0), "r"(row0_1), "r"(kt_2 * 4),
                               "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(b_full_addr + (b_cur_1) * 8, 32768);
                        b_cur_1 += 1;
                        if (b_cur_1 == 4) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int _ = 0; _ < 4; _++) {
                    mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                    b_cur_1 += 1;
                    if (b_cur_1 == 4) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // transform_main
            int valid_tiles_5 = num_non_exiting_tiles[0];
            int end_5 = valid_tiles_5 * num_m_tiles;
            int num_k_3 = K / 256;
            const int quarter_1 = warp % 4;
            const int half = warp / 4 - 2;
            const int src_row = ((lane < 16) ? quarter_1 * 16 + lane : 48 + quarter_1 * 16 + lane);
            const unsigned int lane_base = (unsigned int)(quarter_1 * 32) << 16;
            unsigned int raw_cur_1 = 0;
            unsigned int a_cur_1 = 0;
            unsigned int raw_words[16];
            unsigned int scale_words[2];
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_a_empty = 1;
            #pragma unroll 1
            for (unsigned int w_4 = bid; w_4 < end_5; w_4 += num_bids) {
                #pragma unroll 1
                for (int _kt = 0; _kt < num_k_3; _kt++) {
                    mbarrier_wait(raw_full_addr + (raw_cur_1) * 8, _phase_raw_full);
                    mbarrier_wait(a_empty_addr + (a_cur_1) * 8, _phase_a_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int raw_row = raw_smem_addr + raw_cur_1 * 16384 + (unsigned int)(src_row * 128);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(4) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 + 16 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(8) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 + 32 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(12) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 + 48 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&scale_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words[(0) + 1]))
                        : "r"(sf_smem_addr + raw_cur_1 * 2048 + (unsigned int)(src_row * 16 + half * 8)));
                    unsigned int a_base = taddr + 128 + lane_base + (a_cur_1 * 128 + (unsigned int)(half * 64));
                    uint8_t scale_byte = (uint8_t)(scale_words[0] & 255);
                    uint32_t _fp4_dequant_block16_bf16_0[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[0]), (uint32_t)(raw_words[1]), _scale_byte, _fp4_dequant_block16_bf16_0);
                    }
                    tmem_st_x8_u32(a_base, (const uint32_t*)_fp4_dequant_block16_bf16_0);
                    uint8_t scale_byte_0 = (uint8_t)(scale_words[0] >> 8 & 255);
                    uint32_t _fp4_dequant_block16_bf16_1[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_0)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[2]), (uint32_t)(raw_words[3]), _scale_byte, _fp4_dequant_block16_bf16_1);
                    }
                    tmem_st_x8_u32(a_base + 8, (const uint32_t*)_fp4_dequant_block16_bf16_1);
                    uint8_t scale_byte_1 = (uint8_t)(scale_words[0] >> 16 & 255);
                    uint32_t _fp4_dequant_block16_bf16_2[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_1)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[4]), (uint32_t)(raw_words[5]), _scale_byte, _fp4_dequant_block16_bf16_2);
                    }
                    tmem_st_x8_u32(a_base + 16, (const uint32_t*)_fp4_dequant_block16_bf16_2);
                    uint8_t scale_byte_2 = (uint8_t)(scale_words[0] >> 24 & 255);
                    uint32_t _fp4_dequant_block16_bf16_3[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_2)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[6]), (uint32_t)(raw_words[7]), _scale_byte, _fp4_dequant_block16_bf16_3);
                    }
                    tmem_st_x8_u32(a_base + 24, (const uint32_t*)_fp4_dequant_block16_bf16_3);
                    uint8_t scale_byte_3 = (uint8_t)(scale_words[1] & 255);
                    uint32_t _fp4_dequant_block16_bf16_4[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_3)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[8]), (uint32_t)(raw_words[9]), _scale_byte, _fp4_dequant_block16_bf16_4);
                    }
                    tmem_st_x8_u32(a_base + 32, (const uint32_t*)_fp4_dequant_block16_bf16_4);
                    uint8_t scale_byte_4 = (uint8_t)(scale_words[1] >> 8 & 255);
                    uint32_t _fp4_dequant_block16_bf16_5[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_4)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[10]), (uint32_t)(raw_words[11]), _scale_byte, _fp4_dequant_block16_bf16_5);
                    }
                    tmem_st_x8_u32(a_base + 40, (const uint32_t*)_fp4_dequant_block16_bf16_5);
                    uint8_t scale_byte_5 = (uint8_t)(scale_words[1] >> 16 & 255);
                    uint32_t _fp4_dequant_block16_bf16_6[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_5)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[12]), (uint32_t)(raw_words[13]), _scale_byte, _fp4_dequant_block16_bf16_6);
                    }
                    tmem_st_x8_u32(a_base + 48, (const uint32_t*)_fp4_dequant_block16_bf16_6);
                    uint8_t scale_byte_6 = (uint8_t)(scale_words[1] >> 24 & 255);
                    uint32_t _fp4_dequant_block16_bf16_7[8];
                    {
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_6)) & 0xFFu;
                        cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[14]), (uint32_t)(raw_words[15]), _scale_byte, _fp4_dequant_block16_bf16_7);
                    }
                    tmem_st_x8_u32(a_base + 56, (const uint32_t*)_fp4_dequant_block16_bf16_7);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive(a_full_addr + (a_cur_1) * 8);
                            mbarrier_arrive(raw_empty_addr + (raw_cur_1) * 8);
                        }
                    }
                    raw_cur_1 += 1;
                    if (raw_cur_1 == 4) { raw_cur_1 = 0; _phase_raw_full ^= 1; }
                    a_cur_1 += 1;
                    if (a_cur_1 == 3) { a_cur_1 = 0; _phase_a_empty ^= 1; }
                }
            }
            #pragma unroll 1
            for (int __1 = 0; __1 < 3; __1++) {
                mbarrier_wait(a_empty_addr + (a_cur_1) * 8, _phase_a_empty);
                a_cur_1 += 1;
                if (a_cur_1 == 3) { a_cur_1 = 0; _phase_a_empty ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
