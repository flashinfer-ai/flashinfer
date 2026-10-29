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
#define NUM_TMA_PIPE_STAGES 4
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 1024
#define SMEM_LAND_STRIDE 1024
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 3072
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 134144
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_BN_OFF 134144
#define SMEM_SMEM_BN_STAGE_BYTES 16384
#define SMEM_SMEM_BN_STRIDE 16384
#define SMEM_SMEM_AG_OFF 3072
#define SMEM_SMEM_AG_STAGE_BYTES 16384
#define SMEM_SMEM_AG_STRIDE 32768
#define SMEM_SMEM_AU_OFF 19456
#define SMEM_SMEM_AU_STAGE_BYTES 16384
#define SMEM_SMEM_AU_STRIDE 32768
#define SMEM_TOTAL 199680

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

extern "C" {

__global__ __launch_bounds__(192) void
kernel_cake_kimi_k3_latent_moe_6d29ea0ec6599d33c6e7(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_empty_addr (mbar_base + 32)
    #define acc_full_addr (mbar_base + 64)
    #define acc_empty_addr (mbar_base + 72)
    #define red_bar_addr (mbar_base + 80)
    #define issue_bar_addr (mbar_base + 88)
    #define rows_bar_addr (mbar_base + 96)
    #define bn_bar_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_a_addr = smem + 3072;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 134144);
    const int smem_b_addr = smem + 134144;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 134144);
    const int smem_bn_addr = smem + 134144;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_ag_addr = smem + 3072;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 19456);
    const int smem_au_addr = smem + 19456;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // a_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 0) {
        int _tmem_hold = smem + 112;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int g_e = bid;
            int epi_warp = warp % 4;
            int tid_1 = epi_warp * 32 + lane;
            int tl_e = g_e * 16;
            int tile = g_e;
            int slot = 0;
            int cls = ((tile < 7) ? 0 : ((tile < 35) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = (tile - 7) * 128;
            int row_s = (tile - 7 - 28) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            float red[64];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (cls == 2) {
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(lane_addr + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_0[0];
                red[32] = _tmem_load_1[0];
                red[1] = _tmem_load_0[1];
                red[33] = _tmem_load_1[1];
                red[2] = _tmem_load_0[2];
                red[34] = _tmem_load_1[2];
                red[3] = _tmem_load_0[3];
                red[35] = _tmem_load_1[3];
                red[4] = _tmem_load_0[4];
                red[36] = _tmem_load_1[4];
                red[5] = _tmem_load_0[5];
                red[37] = _tmem_load_1[5];
                red[6] = _tmem_load_0[6];
                red[38] = _tmem_load_1[6];
                red[7] = _tmem_load_0[7];
                red[39] = _tmem_load_1[7];
                red[8] = _tmem_load_0[8];
                red[40] = _tmem_load_1[8];
                red[9] = _tmem_load_0[9];
                red[41] = _tmem_load_1[9];
                red[10] = _tmem_load_0[10];
                red[42] = _tmem_load_1[10];
                red[11] = _tmem_load_0[11];
                red[43] = _tmem_load_1[11];
                red[12] = _tmem_load_0[12];
                red[44] = _tmem_load_1[12];
                red[13] = _tmem_load_0[13];
                red[45] = _tmem_load_1[13];
                red[14] = _tmem_load_0[14];
                red[46] = _tmem_load_1[14];
                red[15] = _tmem_load_0[15];
                red[47] = _tmem_load_1[15];
                red[16] = _tmem_load_0[16];
                red[48] = _tmem_load_1[16];
                red[17] = _tmem_load_0[17];
                red[49] = _tmem_load_1[17];
                red[18] = _tmem_load_0[18];
                red[50] = _tmem_load_1[18];
                red[19] = _tmem_load_0[19];
                red[51] = _tmem_load_1[19];
                red[20] = _tmem_load_0[20];
                red[52] = _tmem_load_1[20];
                red[21] = _tmem_load_0[21];
                red[53] = _tmem_load_1[21];
                red[22] = _tmem_load_0[22];
                red[54] = _tmem_load_1[22];
                red[23] = _tmem_load_0[23];
                red[55] = _tmem_load_1[23];
                red[24] = _tmem_load_0[24];
                red[56] = _tmem_load_1[24];
                red[25] = _tmem_load_0[25];
                red[57] = _tmem_load_1[25];
                red[26] = _tmem_load_0[26];
                red[58] = _tmem_load_1[26];
                red[27] = _tmem_load_0[27];
                red[59] = _tmem_load_1[27];
                red[28] = _tmem_load_0[28];
                red[60] = _tmem_load_1[28];
                red[29] = _tmem_load_0[29];
                red[61] = _tmem_load_1[29];
                red[30] = _tmem_load_0[30];
                red[62] = _tmem_load_1[30];
                red[31] = _tmem_load_0[31];
                red[63] = _tmem_load_1[31];
            } else {
                float _tmem_load_2[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(lane_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_2[0];
                red[1] = _tmem_load_2[1];
                red[2] = _tmem_load_2[2];
                red[3] = _tmem_load_2[3];
                red[4] = _tmem_load_2[4];
                red[5] = _tmem_load_2[5];
                red[6] = _tmem_load_2[6];
                red[7] = _tmem_load_2[7];
                red[8] = _tmem_load_2[8];
                red[9] = _tmem_load_2[9];
                red[10] = _tmem_load_2[10];
                red[11] = _tmem_load_2[11];
                red[12] = _tmem_load_2[12];
                red[13] = _tmem_load_2[13];
                red[14] = _tmem_load_2[14];
                red[15] = _tmem_load_2[15];
                red[16] = _tmem_load_2[16];
                red[17] = _tmem_load_2[17];
                red[18] = _tmem_load_2[18];
                red[19] = _tmem_load_2[19];
                red[20] = _tmem_load_2[20];
                red[21] = _tmem_load_2[21];
                red[22] = _tmem_load_2[22];
                red[23] = _tmem_load_2[23];
                red[24] = _tmem_load_2[24];
                red[25] = _tmem_load_2[25];
                red[26] = _tmem_load_2[26];
                red[27] = _tmem_load_2[27];
                red[28] = _tmem_load_2[28];
                red[29] = _tmem_load_2[29];
                red[30] = _tmem_load_2[30];
                red[31] = _tmem_load_2[31];
                red[32] = _tmem_load_2[32];
                red[33] = _tmem_load_2[33];
                red[34] = _tmem_load_2[34];
                red[35] = _tmem_load_2[35];
                red[36] = _tmem_load_2[36];
                red[37] = _tmem_load_2[37];
                red[38] = _tmem_load_2[38];
                red[39] = _tmem_load_2[39];
                red[40] = _tmem_load_2[40];
                red[41] = _tmem_load_2[41];
                red[42] = _tmem_load_2[42];
                red[43] = _tmem_load_2[43];
                red[44] = _tmem_load_2[44];
                red[45] = _tmem_load_2[45];
                red[46] = _tmem_load_2[46];
                red[47] = _tmem_load_2[47];
                red[48] = _tmem_load_2[48];
                red[49] = _tmem_load_2[49];
                red[50] = _tmem_load_2[50];
                red[51] = _tmem_load_2[51];
                red[52] = _tmem_load_2[52];
                red[53] = _tmem_load_2[53];
                red[54] = _tmem_load_2[54];
                red[55] = _tmem_load_2[55];
                red[56] = _tmem_load_2[56];
                red[57] = _tmem_load_2[57];
                red[58] = _tmem_load_2[58];
                red[59] = _tmem_load_2[59];
                red[60] = _tmem_load_2[60];
                red[61] = _tmem_load_2[61];
                red[62] = _tmem_load_2[62];
                red[63] = _tmem_load_2[63];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr);
                }
            }
            if (fin == 1) {
                if (cls == 2) {
                    int tok_1_0 = lane_pair * 2;
                    int frow_2_0 = row_base;
                    if (tok_1_0 < num_tokens) {
                        float gk_1_0 = (float)(__nv_bfloat16)red[0];
                        float uk_1_0 = (float)(__nv_bfloat16)red[32];
                        float _exp2_0 = approx_exp2((-gk_1_0) * 1.442695f);
                        float sig_1_0 = approx_rcp(1.0f + _exp2_0);
                        float _tanh_1 = tanhf(gk_1_0 * 0.25f);
                        float aa_1_0 = 4.0f * _tanh_1 * sig_1_0;
                        float _tanh_2 = tanhf(uk_1_0 * 0.04f);
                        float bb_1_0 = 25.0f * _tanh_2;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_0 * 6144 + row_s + frow_2_0)) + (0)) = (__nv_bfloat16)(aa_1_0 * bb_1_0);
                    }
                    int tok_1_1 = lane_pair * 2 + 1;
                    int frow_2_1 = row_base;
                    if (tok_1_1 < num_tokens) {
                        float gk_1_1 = (float)(__nv_bfloat16)red[1];
                        float uk_1_1 = (float)(__nv_bfloat16)red[33];
                        float _exp2_3 = approx_exp2((-gk_1_1) * 1.442695f);
                        float sig_1_1 = approx_rcp(1.0f + _exp2_3);
                        float _tanh_4 = tanhf(gk_1_1 * 0.25f);
                        float aa_1_1 = 4.0f * _tanh_4 * sig_1_1;
                        float _tanh_5 = tanhf(uk_1_1 * 0.04f);
                        float bb_1_1 = 25.0f * _tanh_5;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_1 * 6144 + row_s + frow_2_1)) + (0)) = (__nv_bfloat16)(aa_1_1 * bb_1_1);
                    }
                    int tok_1_2 = lane_pair * 2;
                    int frow_2_2 = row_base + 8;
                    if (tok_1_2 < num_tokens) {
                        float gk_1_2 = (float)(__nv_bfloat16)red[2];
                        float uk_1_2 = (float)(__nv_bfloat16)red[34];
                        float _exp2_6 = approx_exp2((-gk_1_2) * 1.442695f);
                        float sig_1_2 = approx_rcp(1.0f + _exp2_6);
                        float _tanh_7 = tanhf(gk_1_2 * 0.25f);
                        float aa_1_2 = 4.0f * _tanh_7 * sig_1_2;
                        float _tanh_8 = tanhf(uk_1_2 * 0.04f);
                        float bb_1_2 = 25.0f * _tanh_8;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_2 * 6144 + row_s + frow_2_2)) + (0)) = (__nv_bfloat16)(aa_1_2 * bb_1_2);
                    }
                    int tok_1_3 = lane_pair * 2 + 1;
                    int frow_2_3 = row_base + 8;
                    if (tok_1_3 < num_tokens) {
                        float gk_1_3 = (float)(__nv_bfloat16)red[3];
                        float uk_1_3 = (float)(__nv_bfloat16)red[35];
                        float _exp2_9 = approx_exp2((-gk_1_3) * 1.442695f);
                        float sig_1_3 = approx_rcp(1.0f + _exp2_9);
                        float _tanh_10 = tanhf(gk_1_3 * 0.25f);
                        float aa_1_3 = 4.0f * _tanh_10 * sig_1_3;
                        float _tanh_11 = tanhf(uk_1_3 * 0.04f);
                        float bb_1_3 = 25.0f * _tanh_11;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_3 * 6144 + row_s + frow_2_3)) + (0)) = (__nv_bfloat16)(aa_1_3 * bb_1_3);
                    }
                    int tok_1_4 = 8 + lane_pair * 2;
                    int frow_2_4 = row_base;
                    if (tok_1_4 < num_tokens) {
                        float gk_1_4 = (float)(__nv_bfloat16)red[4];
                        float uk_1_4 = (float)(__nv_bfloat16)red[36];
                        float _exp2_12 = approx_exp2((-gk_1_4) * 1.442695f);
                        float sig_1_4 = approx_rcp(1.0f + _exp2_12);
                        float _tanh_13 = tanhf(gk_1_4 * 0.25f);
                        float aa_1_4 = 4.0f * _tanh_13 * sig_1_4;
                        float _tanh_14 = tanhf(uk_1_4 * 0.04f);
                        float bb_1_4 = 25.0f * _tanh_14;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_4 * 6144 + row_s + frow_2_4)) + (0)) = (__nv_bfloat16)(aa_1_4 * bb_1_4);
                    }
                    int tok_1_5 = 8 + lane_pair * 2 + 1;
                    int frow_2_5 = row_base;
                    if (tok_1_5 < num_tokens) {
                        float gk_1_5 = (float)(__nv_bfloat16)red[5];
                        float uk_1_5 = (float)(__nv_bfloat16)red[37];
                        float _exp2_15 = approx_exp2((-gk_1_5) * 1.442695f);
                        float sig_1_5 = approx_rcp(1.0f + _exp2_15);
                        float _tanh_16 = tanhf(gk_1_5 * 0.25f);
                        float aa_1_5 = 4.0f * _tanh_16 * sig_1_5;
                        float _tanh_17 = tanhf(uk_1_5 * 0.04f);
                        float bb_1_5 = 25.0f * _tanh_17;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_5 * 6144 + row_s + frow_2_5)) + (0)) = (__nv_bfloat16)(aa_1_5 * bb_1_5);
                    }
                    int tok_1_6 = 8 + lane_pair * 2;
                    int frow_2_6 = row_base + 8;
                    if (tok_1_6 < num_tokens) {
                        float gk_1_6 = (float)(__nv_bfloat16)red[6];
                        float uk_1_6 = (float)(__nv_bfloat16)red[38];
                        float _exp2_18 = approx_exp2((-gk_1_6) * 1.442695f);
                        float sig_1_6 = approx_rcp(1.0f + _exp2_18);
                        float _tanh_19 = tanhf(gk_1_6 * 0.25f);
                        float aa_1_6 = 4.0f * _tanh_19 * sig_1_6;
                        float _tanh_20 = tanhf(uk_1_6 * 0.04f);
                        float bb_1_6 = 25.0f * _tanh_20;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_6 * 6144 + row_s + frow_2_6)) + (0)) = (__nv_bfloat16)(aa_1_6 * bb_1_6);
                    }
                    int tok_1_7 = 8 + lane_pair * 2 + 1;
                    int frow_2_7 = row_base + 8;
                    if (tok_1_7 < num_tokens) {
                        float gk_1_7 = (float)(__nv_bfloat16)red[7];
                        float uk_1_7 = (float)(__nv_bfloat16)red[39];
                        float _exp2_21 = approx_exp2((-gk_1_7) * 1.442695f);
                        float sig_1_7 = approx_rcp(1.0f + _exp2_21);
                        float _tanh_22 = tanhf(gk_1_7 * 0.25f);
                        float aa_1_7 = 4.0f * _tanh_22 * sig_1_7;
                        float _tanh_23 = tanhf(uk_1_7 * 0.04f);
                        float bb_1_7 = 25.0f * _tanh_23;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_7 * 6144 + row_s + frow_2_7)) + (0)) = (__nv_bfloat16)(aa_1_7 * bb_1_7);
                    }
                    int tok_1_8 = 16 + lane_pair * 2;
                    int frow_2_8 = row_base;
                    if (tok_1_8 < num_tokens) {
                        float gk_1_8 = (float)(__nv_bfloat16)red[8];
                        float uk_1_8 = (float)(__nv_bfloat16)red[40];
                        float _exp2_24 = approx_exp2((-gk_1_8) * 1.442695f);
                        float sig_1_8 = approx_rcp(1.0f + _exp2_24);
                        float _tanh_25 = tanhf(gk_1_8 * 0.25f);
                        float aa_1_8 = 4.0f * _tanh_25 * sig_1_8;
                        float _tanh_26 = tanhf(uk_1_8 * 0.04f);
                        float bb_1_8 = 25.0f * _tanh_26;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_8 * 6144 + row_s + frow_2_8)) + (0)) = (__nv_bfloat16)(aa_1_8 * bb_1_8);
                    }
                    int tok_1_9 = 16 + lane_pair * 2 + 1;
                    int frow_2_9 = row_base;
                    if (tok_1_9 < num_tokens) {
                        float gk_1_9 = (float)(__nv_bfloat16)red[9];
                        float uk_1_9 = (float)(__nv_bfloat16)red[41];
                        float _exp2_27 = approx_exp2((-gk_1_9) * 1.442695f);
                        float sig_1_9 = approx_rcp(1.0f + _exp2_27);
                        float _tanh_28 = tanhf(gk_1_9 * 0.25f);
                        float aa_1_9 = 4.0f * _tanh_28 * sig_1_9;
                        float _tanh_29 = tanhf(uk_1_9 * 0.04f);
                        float bb_1_9 = 25.0f * _tanh_29;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_9 * 6144 + row_s + frow_2_9)) + (0)) = (__nv_bfloat16)(aa_1_9 * bb_1_9);
                    }
                    int tok_1_10 = 16 + lane_pair * 2;
                    int frow_2_10 = row_base + 8;
                    if (tok_1_10 < num_tokens) {
                        float gk_1_10 = (float)(__nv_bfloat16)red[10];
                        float uk_1_10 = (float)(__nv_bfloat16)red[42];
                        float _exp2_30 = approx_exp2((-gk_1_10) * 1.442695f);
                        float sig_1_10 = approx_rcp(1.0f + _exp2_30);
                        float _tanh_31 = tanhf(gk_1_10 * 0.25f);
                        float aa_1_10 = 4.0f * _tanh_31 * sig_1_10;
                        float _tanh_32 = tanhf(uk_1_10 * 0.04f);
                        float bb_1_10 = 25.0f * _tanh_32;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_10 * 6144 + row_s + frow_2_10)) + (0)) = (__nv_bfloat16)(aa_1_10 * bb_1_10);
                    }
                    int tok_1_11 = 16 + lane_pair * 2 + 1;
                    int frow_2_11 = row_base + 8;
                    if (tok_1_11 < num_tokens) {
                        float gk_1_11 = (float)(__nv_bfloat16)red[11];
                        float uk_1_11 = (float)(__nv_bfloat16)red[43];
                        float _exp2_33 = approx_exp2((-gk_1_11) * 1.442695f);
                        float sig_1_11 = approx_rcp(1.0f + _exp2_33);
                        float _tanh_34 = tanhf(gk_1_11 * 0.25f);
                        float aa_1_11 = 4.0f * _tanh_34 * sig_1_11;
                        float _tanh_35 = tanhf(uk_1_11 * 0.04f);
                        float bb_1_11 = 25.0f * _tanh_35;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_11 * 6144 + row_s + frow_2_11)) + (0)) = (__nv_bfloat16)(aa_1_11 * bb_1_11);
                    }
                    int tok_1_12 = 24 + lane_pair * 2;
                    int frow_2_12 = row_base;
                    if (tok_1_12 < num_tokens) {
                        float gk_1_12 = (float)(__nv_bfloat16)red[12];
                        float uk_1_12 = (float)(__nv_bfloat16)red[44];
                        float _exp2_36 = approx_exp2((-gk_1_12) * 1.442695f);
                        float sig_1_12 = approx_rcp(1.0f + _exp2_36);
                        float _tanh_37 = tanhf(gk_1_12 * 0.25f);
                        float aa_1_12 = 4.0f * _tanh_37 * sig_1_12;
                        float _tanh_38 = tanhf(uk_1_12 * 0.04f);
                        float bb_1_12 = 25.0f * _tanh_38;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_12 * 6144 + row_s + frow_2_12)) + (0)) = (__nv_bfloat16)(aa_1_12 * bb_1_12);
                    }
                    int tok_1_13 = 24 + lane_pair * 2 + 1;
                    int frow_2_13 = row_base;
                    if (tok_1_13 < num_tokens) {
                        float gk_1_13 = (float)(__nv_bfloat16)red[13];
                        float uk_1_13 = (float)(__nv_bfloat16)red[45];
                        float _exp2_39 = approx_exp2((-gk_1_13) * 1.442695f);
                        float sig_1_13 = approx_rcp(1.0f + _exp2_39);
                        float _tanh_40 = tanhf(gk_1_13 * 0.25f);
                        float aa_1_13 = 4.0f * _tanh_40 * sig_1_13;
                        float _tanh_41 = tanhf(uk_1_13 * 0.04f);
                        float bb_1_13 = 25.0f * _tanh_41;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_13 * 6144 + row_s + frow_2_13)) + (0)) = (__nv_bfloat16)(aa_1_13 * bb_1_13);
                    }
                    int tok_1_14 = 24 + lane_pair * 2;
                    int frow_2_14 = row_base + 8;
                    if (tok_1_14 < num_tokens) {
                        float gk_1_14 = (float)(__nv_bfloat16)red[14];
                        float uk_1_14 = (float)(__nv_bfloat16)red[46];
                        float _exp2_42 = approx_exp2((-gk_1_14) * 1.442695f);
                        float sig_1_14 = approx_rcp(1.0f + _exp2_42);
                        float _tanh_43 = tanhf(gk_1_14 * 0.25f);
                        float aa_1_14 = 4.0f * _tanh_43 * sig_1_14;
                        float _tanh_44 = tanhf(uk_1_14 * 0.04f);
                        float bb_1_14 = 25.0f * _tanh_44;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_14 * 6144 + row_s + frow_2_14)) + (0)) = (__nv_bfloat16)(aa_1_14 * bb_1_14);
                    }
                    int tok_1_15 = 24 + lane_pair * 2 + 1;
                    int frow_2_15 = row_base + 8;
                    if (tok_1_15 < num_tokens) {
                        float gk_1_15 = (float)(__nv_bfloat16)red[15];
                        float uk_1_15 = (float)(__nv_bfloat16)red[47];
                        float _exp2_45 = approx_exp2((-gk_1_15) * 1.442695f);
                        float sig_1_15 = approx_rcp(1.0f + _exp2_45);
                        float _tanh_46 = tanhf(gk_1_15 * 0.25f);
                        float aa_1_15 = 4.0f * _tanh_46 * sig_1_15;
                        float _tanh_47 = tanhf(uk_1_15 * 0.04f);
                        float bb_1_15 = 25.0f * _tanh_47;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_15 * 6144 + row_s + frow_2_15)) + (0)) = (__nv_bfloat16)(aa_1_15 * bb_1_15);
                    }
                    int tok_1_16 = 32 + lane_pair * 2;
                    int frow_2_16 = row_base;
                    if (tok_1_16 < num_tokens) {
                        float gk_1_16 = (float)(__nv_bfloat16)red[16];
                        float uk_1_16 = (float)(__nv_bfloat16)red[48];
                        float _exp2_48 = approx_exp2((-gk_1_16) * 1.442695f);
                        float sig_1_16 = approx_rcp(1.0f + _exp2_48);
                        float _tanh_49 = tanhf(gk_1_16 * 0.25f);
                        float aa_1_16 = 4.0f * _tanh_49 * sig_1_16;
                        float _tanh_50 = tanhf(uk_1_16 * 0.04f);
                        float bb_1_16 = 25.0f * _tanh_50;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_16 * 6144 + row_s + frow_2_16)) + (0)) = (__nv_bfloat16)(aa_1_16 * bb_1_16);
                    }
                    int tok_1_17 = 32 + lane_pair * 2 + 1;
                    int frow_2_17 = row_base;
                    if (tok_1_17 < num_tokens) {
                        float gk_1_17 = (float)(__nv_bfloat16)red[17];
                        float uk_1_17 = (float)(__nv_bfloat16)red[49];
                        float _exp2_51 = approx_exp2((-gk_1_17) * 1.442695f);
                        float sig_1_17 = approx_rcp(1.0f + _exp2_51);
                        float _tanh_52 = tanhf(gk_1_17 * 0.25f);
                        float aa_1_17 = 4.0f * _tanh_52 * sig_1_17;
                        float _tanh_53 = tanhf(uk_1_17 * 0.04f);
                        float bb_1_17 = 25.0f * _tanh_53;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_17 * 6144 + row_s + frow_2_17)) + (0)) = (__nv_bfloat16)(aa_1_17 * bb_1_17);
                    }
                    int tok_1_18 = 32 + lane_pair * 2;
                    int frow_2_18 = row_base + 8;
                    if (tok_1_18 < num_tokens) {
                        float gk_1_18 = (float)(__nv_bfloat16)red[18];
                        float uk_1_18 = (float)(__nv_bfloat16)red[50];
                        float _exp2_54 = approx_exp2((-gk_1_18) * 1.442695f);
                        float sig_1_18 = approx_rcp(1.0f + _exp2_54);
                        float _tanh_55 = tanhf(gk_1_18 * 0.25f);
                        float aa_1_18 = 4.0f * _tanh_55 * sig_1_18;
                        float _tanh_56 = tanhf(uk_1_18 * 0.04f);
                        float bb_1_18 = 25.0f * _tanh_56;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_18 * 6144 + row_s + frow_2_18)) + (0)) = (__nv_bfloat16)(aa_1_18 * bb_1_18);
                    }
                    int tok_1_19 = 32 + lane_pair * 2 + 1;
                    int frow_2_19 = row_base + 8;
                    if (tok_1_19 < num_tokens) {
                        float gk_1_19 = (float)(__nv_bfloat16)red[19];
                        float uk_1_19 = (float)(__nv_bfloat16)red[51];
                        float _exp2_57 = approx_exp2((-gk_1_19) * 1.442695f);
                        float sig_1_19 = approx_rcp(1.0f + _exp2_57);
                        float _tanh_58 = tanhf(gk_1_19 * 0.25f);
                        float aa_1_19 = 4.0f * _tanh_58 * sig_1_19;
                        float _tanh_59 = tanhf(uk_1_19 * 0.04f);
                        float bb_1_19 = 25.0f * _tanh_59;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_19 * 6144 + row_s + frow_2_19)) + (0)) = (__nv_bfloat16)(aa_1_19 * bb_1_19);
                    }
                    int tok_1_20 = 40 + lane_pair * 2;
                    int frow_2_20 = row_base;
                    if (tok_1_20 < num_tokens) {
                        float gk_1_20 = (float)(__nv_bfloat16)red[20];
                        float uk_1_20 = (float)(__nv_bfloat16)red[52];
                        float _exp2_60 = approx_exp2((-gk_1_20) * 1.442695f);
                        float sig_1_20 = approx_rcp(1.0f + _exp2_60);
                        float _tanh_61 = tanhf(gk_1_20 * 0.25f);
                        float aa_1_20 = 4.0f * _tanh_61 * sig_1_20;
                        float _tanh_62 = tanhf(uk_1_20 * 0.04f);
                        float bb_1_20 = 25.0f * _tanh_62;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_20 * 6144 + row_s + frow_2_20)) + (0)) = (__nv_bfloat16)(aa_1_20 * bb_1_20);
                    }
                    int tok_1_21 = 40 + lane_pair * 2 + 1;
                    int frow_2_21 = row_base;
                    if (tok_1_21 < num_tokens) {
                        float gk_1_21 = (float)(__nv_bfloat16)red[21];
                        float uk_1_21 = (float)(__nv_bfloat16)red[53];
                        float _exp2_63 = approx_exp2((-gk_1_21) * 1.442695f);
                        float sig_1_21 = approx_rcp(1.0f + _exp2_63);
                        float _tanh_64 = tanhf(gk_1_21 * 0.25f);
                        float aa_1_21 = 4.0f * _tanh_64 * sig_1_21;
                        float _tanh_65 = tanhf(uk_1_21 * 0.04f);
                        float bb_1_21 = 25.0f * _tanh_65;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_21 * 6144 + row_s + frow_2_21)) + (0)) = (__nv_bfloat16)(aa_1_21 * bb_1_21);
                    }
                    int tok_1_22 = 40 + lane_pair * 2;
                    int frow_2_22 = row_base + 8;
                    if (tok_1_22 < num_tokens) {
                        float gk_1_22 = (float)(__nv_bfloat16)red[22];
                        float uk_1_22 = (float)(__nv_bfloat16)red[54];
                        float _exp2_66 = approx_exp2((-gk_1_22) * 1.442695f);
                        float sig_1_22 = approx_rcp(1.0f + _exp2_66);
                        float _tanh_67 = tanhf(gk_1_22 * 0.25f);
                        float aa_1_22 = 4.0f * _tanh_67 * sig_1_22;
                        float _tanh_68 = tanhf(uk_1_22 * 0.04f);
                        float bb_1_22 = 25.0f * _tanh_68;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_22 * 6144 + row_s + frow_2_22)) + (0)) = (__nv_bfloat16)(aa_1_22 * bb_1_22);
                    }
                    int tok_1_23 = 40 + lane_pair * 2 + 1;
                    int frow_2_23 = row_base + 8;
                    if (tok_1_23 < num_tokens) {
                        float gk_1_23 = (float)(__nv_bfloat16)red[23];
                        float uk_1_23 = (float)(__nv_bfloat16)red[55];
                        float _exp2_69 = approx_exp2((-gk_1_23) * 1.442695f);
                        float sig_1_23 = approx_rcp(1.0f + _exp2_69);
                        float _tanh_70 = tanhf(gk_1_23 * 0.25f);
                        float aa_1_23 = 4.0f * _tanh_70 * sig_1_23;
                        float _tanh_71 = tanhf(uk_1_23 * 0.04f);
                        float bb_1_23 = 25.0f * _tanh_71;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_23 * 6144 + row_s + frow_2_23)) + (0)) = (__nv_bfloat16)(aa_1_23 * bb_1_23);
                    }
                    int tok_1_24 = 48 + lane_pair * 2;
                    int frow_2_24 = row_base;
                    if (tok_1_24 < num_tokens) {
                        float gk_1_24 = (float)(__nv_bfloat16)red[24];
                        float uk_1_24 = (float)(__nv_bfloat16)red[56];
                        float _exp2_72 = approx_exp2((-gk_1_24) * 1.442695f);
                        float sig_1_24 = approx_rcp(1.0f + _exp2_72);
                        float _tanh_73 = tanhf(gk_1_24 * 0.25f);
                        float aa_1_24 = 4.0f * _tanh_73 * sig_1_24;
                        float _tanh_74 = tanhf(uk_1_24 * 0.04f);
                        float bb_1_24 = 25.0f * _tanh_74;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_24 * 6144 + row_s + frow_2_24)) + (0)) = (__nv_bfloat16)(aa_1_24 * bb_1_24);
                    }
                    int tok_1_25 = 48 + lane_pair * 2 + 1;
                    int frow_2_25 = row_base;
                    if (tok_1_25 < num_tokens) {
                        float gk_1_25 = (float)(__nv_bfloat16)red[25];
                        float uk_1_25 = (float)(__nv_bfloat16)red[57];
                        float _exp2_75 = approx_exp2((-gk_1_25) * 1.442695f);
                        float sig_1_25 = approx_rcp(1.0f + _exp2_75);
                        float _tanh_76 = tanhf(gk_1_25 * 0.25f);
                        float aa_1_25 = 4.0f * _tanh_76 * sig_1_25;
                        float _tanh_77 = tanhf(uk_1_25 * 0.04f);
                        float bb_1_25 = 25.0f * _tanh_77;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_25 * 6144 + row_s + frow_2_25)) + (0)) = (__nv_bfloat16)(aa_1_25 * bb_1_25);
                    }
                    int tok_1_26 = 48 + lane_pair * 2;
                    int frow_2_26 = row_base + 8;
                    if (tok_1_26 < num_tokens) {
                        float gk_1_26 = (float)(__nv_bfloat16)red[26];
                        float uk_1_26 = (float)(__nv_bfloat16)red[58];
                        float _exp2_78 = approx_exp2((-gk_1_26) * 1.442695f);
                        float sig_1_26 = approx_rcp(1.0f + _exp2_78);
                        float _tanh_79 = tanhf(gk_1_26 * 0.25f);
                        float aa_1_26 = 4.0f * _tanh_79 * sig_1_26;
                        float _tanh_80 = tanhf(uk_1_26 * 0.04f);
                        float bb_1_26 = 25.0f * _tanh_80;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_26 * 6144 + row_s + frow_2_26)) + (0)) = (__nv_bfloat16)(aa_1_26 * bb_1_26);
                    }
                    int tok_1_27 = 48 + lane_pair * 2 + 1;
                    int frow_2_27 = row_base + 8;
                    if (tok_1_27 < num_tokens) {
                        float gk_1_27 = (float)(__nv_bfloat16)red[27];
                        float uk_1_27 = (float)(__nv_bfloat16)red[59];
                        float _exp2_81 = approx_exp2((-gk_1_27) * 1.442695f);
                        float sig_1_27 = approx_rcp(1.0f + _exp2_81);
                        float _tanh_82 = tanhf(gk_1_27 * 0.25f);
                        float aa_1_27 = 4.0f * _tanh_82 * sig_1_27;
                        float _tanh_83 = tanhf(uk_1_27 * 0.04f);
                        float bb_1_27 = 25.0f * _tanh_83;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_27 * 6144 + row_s + frow_2_27)) + (0)) = (__nv_bfloat16)(aa_1_27 * bb_1_27);
                    }
                    int tok_1_28 = 56 + lane_pair * 2;
                    int frow_2_28 = row_base;
                    if (tok_1_28 < num_tokens) {
                        float gk_1_28 = (float)(__nv_bfloat16)red[28];
                        float uk_1_28 = (float)(__nv_bfloat16)red[60];
                        float _exp2_84 = approx_exp2((-gk_1_28) * 1.442695f);
                        float sig_1_28 = approx_rcp(1.0f + _exp2_84);
                        float _tanh_85 = tanhf(gk_1_28 * 0.25f);
                        float aa_1_28 = 4.0f * _tanh_85 * sig_1_28;
                        float _tanh_86 = tanhf(uk_1_28 * 0.04f);
                        float bb_1_28 = 25.0f * _tanh_86;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_28 * 6144 + row_s + frow_2_28)) + (0)) = (__nv_bfloat16)(aa_1_28 * bb_1_28);
                    }
                    int tok_1_29 = 56 + lane_pair * 2 + 1;
                    int frow_2_29 = row_base;
                    if (tok_1_29 < num_tokens) {
                        float gk_1_29 = (float)(__nv_bfloat16)red[29];
                        float uk_1_29 = (float)(__nv_bfloat16)red[61];
                        float _exp2_87 = approx_exp2((-gk_1_29) * 1.442695f);
                        float sig_1_29 = approx_rcp(1.0f + _exp2_87);
                        float _tanh_88 = tanhf(gk_1_29 * 0.25f);
                        float aa_1_29 = 4.0f * _tanh_88 * sig_1_29;
                        float _tanh_89 = tanhf(uk_1_29 * 0.04f);
                        float bb_1_29 = 25.0f * _tanh_89;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_29 * 6144 + row_s + frow_2_29)) + (0)) = (__nv_bfloat16)(aa_1_29 * bb_1_29);
                    }
                    int tok_1_30 = 56 + lane_pair * 2;
                    int frow_2_30 = row_base + 8;
                    if (tok_1_30 < num_tokens) {
                        float gk_1_30 = (float)(__nv_bfloat16)red[30];
                        float uk_1_30 = (float)(__nv_bfloat16)red[62];
                        float _exp2_90 = approx_exp2((-gk_1_30) * 1.442695f);
                        float sig_1_30 = approx_rcp(1.0f + _exp2_90);
                        float _tanh_91 = tanhf(gk_1_30 * 0.25f);
                        float aa_1_30 = 4.0f * _tanh_91 * sig_1_30;
                        float _tanh_92 = tanhf(uk_1_30 * 0.04f);
                        float bb_1_30 = 25.0f * _tanh_92;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_30 * 6144 + row_s + frow_2_30)) + (0)) = (__nv_bfloat16)(aa_1_30 * bb_1_30);
                    }
                    int tok_1_31 = 56 + lane_pair * 2 + 1;
                    int frow_2_31 = row_base + 8;
                    if (tok_1_31 < num_tokens) {
                        float gk_1_31 = (float)(__nv_bfloat16)red[31];
                        float uk_1_31 = (float)(__nv_bfloat16)red[63];
                        float _exp2_93 = approx_exp2((-gk_1_31) * 1.442695f);
                        float sig_1_31 = approx_rcp(1.0f + _exp2_93);
                        float _tanh_94 = tanhf(gk_1_31 * 0.25f);
                        float aa_1_31 = 4.0f * _tanh_94 * sig_1_31;
                        float _tanh_95 = tanhf(uk_1_31 * 0.04f);
                        float bb_1_31 = 25.0f * _tanh_95;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_31 * 6144 + row_s + frow_2_31)) + (0)) = (__nv_bfloat16)(aa_1_31 * bb_1_31);
                    }
                } else {
                    if (num_tokens > 0) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (row_r + tid_1)) + (0)) = red[0];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = (__nv_bfloat16)red[0];
                        }
                    }
                    if (num_tokens > 1) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (896 + row_r + tid_1)) + (0)) = red[1];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (3584 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[1];
                        }
                    }
                    if (num_tokens > 2) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (1792 + row_r + tid_1)) + (0)) = red[2];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (7168 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[2];
                        }
                    }
                    if (num_tokens > 3) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (2688 + row_r + tid_1)) + (0)) = red[3];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (10752 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[3];
                        }
                    }
                    if (num_tokens > 4) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (3584 + row_r + tid_1)) + (0)) = red[4];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (14336 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[4];
                        }
                    }
                    if (num_tokens > 5) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (4480 + row_r + tid_1)) + (0)) = red[5];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (17920 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[5];
                        }
                    }
                    if (num_tokens > 6) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (5376 + row_r + tid_1)) + (0)) = red[6];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (21504 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[6];
                        }
                    }
                    if (num_tokens > 7) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (6272 + row_r + tid_1)) + (0)) = red[7];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (25088 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[7];
                        }
                    }
                    if (num_tokens > 8) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (7168 + row_r + tid_1)) + (0)) = red[8];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (28672 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[8];
                        }
                    }
                    if (num_tokens > 9) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (8064 + row_r + tid_1)) + (0)) = red[9];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (32256 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[9];
                        }
                    }
                    if (num_tokens > 10) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (8960 + row_r + tid_1)) + (0)) = red[10];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (35840 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[10];
                        }
                    }
                    if (num_tokens > 11) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (9856 + row_r + tid_1)) + (0)) = red[11];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (39424 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[11];
                        }
                    }
                    if (num_tokens > 12) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (10752 + row_r + tid_1)) + (0)) = red[12];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (43008 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[12];
                        }
                    }
                    if (num_tokens > 13) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (11648 + row_r + tid_1)) + (0)) = red[13];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (46592 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[13];
                        }
                    }
                    if (num_tokens > 14) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (12544 + row_r + tid_1)) + (0)) = red[14];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (50176 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[14];
                        }
                    }
                    if (num_tokens > 15) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (13440 + row_r + tid_1)) + (0)) = red[15];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (53760 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[15];
                        }
                    }
                    if (num_tokens > 16) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (14336 + row_r + tid_1)) + (0)) = red[16];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (57344 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[16];
                        }
                    }
                    if (num_tokens > 17) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (15232 + row_r + tid_1)) + (0)) = red[17];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (60928 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[17];
                        }
                    }
                    if (num_tokens > 18) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (16128 + row_r + tid_1)) + (0)) = red[18];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (64512 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[18];
                        }
                    }
                    if (num_tokens > 19) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (17024 + row_r + tid_1)) + (0)) = red[19];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (68096 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[19];
                        }
                    }
                    if (num_tokens > 20) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (17920 + row_r + tid_1)) + (0)) = red[20];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (71680 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[20];
                        }
                    }
                    if (num_tokens > 21) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (18816 + row_r + tid_1)) + (0)) = red[21];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (75264 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[21];
                        }
                    }
                    if (num_tokens > 22) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (19712 + row_r + tid_1)) + (0)) = red[22];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (78848 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[22];
                        }
                    }
                    if (num_tokens > 23) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (20608 + row_r + tid_1)) + (0)) = red[23];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (82432 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[23];
                        }
                    }
                    if (num_tokens > 24) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (21504 + row_r + tid_1)) + (0)) = red[24];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (86016 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[24];
                        }
                    }
                    if (num_tokens > 25) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (22400 + row_r + tid_1)) + (0)) = red[25];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (89600 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[25];
                        }
                    }
                    if (num_tokens > 26) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (23296 + row_r + tid_1)) + (0)) = red[26];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (93184 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[26];
                        }
                    }
                    if (num_tokens > 27) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (24192 + row_r + tid_1)) + (0)) = red[27];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (96768 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[27];
                        }
                    }
                    if (num_tokens > 28) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (25088 + row_r + tid_1)) + (0)) = red[28];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (100352 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[28];
                        }
                    }
                    if (num_tokens > 29) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (25984 + row_r + tid_1)) + (0)) = red[29];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (103936 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[29];
                        }
                    }
                    if (num_tokens > 30) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (26880 + row_r + tid_1)) + (0)) = red[30];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (107520 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[30];
                        }
                    }
                    if (num_tokens > 31) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (27776 + row_r + tid_1)) + (0)) = red[31];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (111104 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[31];
                        }
                    }
                    if (num_tokens > 32) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (28672 + row_r + tid_1)) + (0)) = red[32];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (114688 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[32];
                        }
                    }
                    if (num_tokens > 33) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (29568 + row_r + tid_1)) + (0)) = red[33];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (118272 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[33];
                        }
                    }
                    if (num_tokens > 34) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (30464 + row_r + tid_1)) + (0)) = red[34];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (121856 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[34];
                        }
                    }
                    if (num_tokens > 35) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (31360 + row_r + tid_1)) + (0)) = red[35];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (125440 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[35];
                        }
                    }
                    if (num_tokens > 36) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (32256 + row_r + tid_1)) + (0)) = red[36];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (129024 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[36];
                        }
                    }
                    if (num_tokens > 37) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (33152 + row_r + tid_1)) + (0)) = red[37];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (132608 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[37];
                        }
                    }
                    if (num_tokens > 38) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (34048 + row_r + tid_1)) + (0)) = red[38];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (136192 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[38];
                        }
                    }
                    if (num_tokens > 39) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (34944 + row_r + tid_1)) + (0)) = red[39];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (139776 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[39];
                        }
                    }
                    if (num_tokens > 40) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (35840 + row_r + tid_1)) + (0)) = red[40];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (143360 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[40];
                        }
                    }
                    if (num_tokens > 41) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (36736 + row_r + tid_1)) + (0)) = red[41];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (146944 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[41];
                        }
                    }
                    if (num_tokens > 42) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (37632 + row_r + tid_1)) + (0)) = red[42];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (150528 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[42];
                        }
                    }
                    if (num_tokens > 43) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (38528 + row_r + tid_1)) + (0)) = red[43];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (154112 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[43];
                        }
                    }
                    if (num_tokens > 44) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (39424 + row_r + tid_1)) + (0)) = red[44];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (157696 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[44];
                        }
                    }
                    if (num_tokens > 45) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (40320 + row_r + tid_1)) + (0)) = red[45];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (161280 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[45];
                        }
                    }
                    if (num_tokens > 46) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (41216 + row_r + tid_1)) + (0)) = red[46];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (164864 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[46];
                        }
                    }
                    if (num_tokens > 47) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (42112 + row_r + tid_1)) + (0)) = red[47];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (168448 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[47];
                        }
                    }
                    if (num_tokens > 48) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (43008 + row_r + tid_1)) + (0)) = red[48];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (172032 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[48];
                        }
                    }
                    if (num_tokens > 49) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (43904 + row_r + tid_1)) + (0)) = red[49];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (175616 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[49];
                        }
                    }
                    if (num_tokens > 50) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (44800 + row_r + tid_1)) + (0)) = red[50];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (179200 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[50];
                        }
                    }
                    if (num_tokens > 51) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (45696 + row_r + tid_1)) + (0)) = red[51];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (182784 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[51];
                        }
                    }
                    if (num_tokens > 52) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (46592 + row_r + tid_1)) + (0)) = red[52];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (186368 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[52];
                        }
                    }
                    if (num_tokens > 53) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (47488 + row_r + tid_1)) + (0)) = red[53];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (189952 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[53];
                        }
                    }
                    if (num_tokens > 54) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (48384 + row_r + tid_1)) + (0)) = red[54];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (193536 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[54];
                        }
                    }
                    if (num_tokens > 55) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (49280 + row_r + tid_1)) + (0)) = red[55];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (197120 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[55];
                        }
                    }
                    if (num_tokens > 56) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (50176 + row_r + tid_1)) + (0)) = red[56];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (200704 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[56];
                        }
                    }
                    if (num_tokens > 57) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (51072 + row_r + tid_1)) + (0)) = red[57];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (204288 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[57];
                        }
                    }
                    if (num_tokens > 58) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (51968 + row_r + tid_1)) + (0)) = red[58];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (207872 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[58];
                        }
                    }
                    if (num_tokens > 59) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (52864 + row_r + tid_1)) + (0)) = red[59];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (211456 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[59];
                        }
                    }
                    if (num_tokens > 60) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (53760 + row_r + tid_1)) + (0)) = red[60];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (215040 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[60];
                        }
                    }
                    if (num_tokens > 61) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (54656 + row_r + tid_1)) + (0)) = red[61];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (218624 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[61];
                        }
                    }
                    if (num_tokens > 62) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (55552 + row_r + tid_1)) + (0)) = red[62];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (222208 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[62];
                        }
                    }
                    if (num_tokens > 63) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (56448 + row_r + tid_1)) + (0)) = red[63];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (225792 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[63];
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int g_l = bid;
            int tl_l = g_l * 16;
            int tile_l = g_l;
            int rank_l = 0;
            int d_lo_l = ((rank_l == 0) ? 0 : 0);
            int d_cnt_l = ((rank_l == 0) ? 0 : 0);
            int u_lo_l = ((rank_l == 0) ? 0 : 56);
            int u_cnt_l = ((rank_l == 0) ? 56 : 0);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int pro = 0;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_R))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_L))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_S))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_2))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_1))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_2))) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = 0;
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 7) ? 0 : ((tile_1 < 35) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = (tile_1 - 7) * 128;
                    int row_s_1 = (tile_1 - 7 - 28) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    int kb1 = k1_off + m * 2;
                    tma_3d_gmem2smem(smem_b_addr + stage * 16384, (&B_1), 0, 0, kb1, a_full_addr + (stage) * 8);
                    if (need_a == 1) {
                        int kc1 = k1_off + m * 2;
                        if (cls_1 == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 32768), "l"((&A_R)), "r"(0), "r"(row_r_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (cls_1 == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 32768), "l"((&A_L)), "r"(0), "r"(row_l_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (cls_1 == 2) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 32768), "l"((&A_S)), "r"(0), "r"(row_s_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 32768 + 16384), "l"((&A_S)), "r"(0), "r"(6144 + row_s_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 49152);
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_a_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int g_m = bid;
            int tl_m = g_m * 16;
            int tile_2 = g_m;
            int rank_m = 0;
            int u_count_m = ((rank_m == 0) ? 56 : 0);
            int d_cnt_m = ((rank_m == 0) ? 0 : 0);
            unsigned int stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 7) ? 0 : ((tile_2 < 35) ? 1 : 2));
                mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    int init_sub_1_0 = init_flag;
                    if (cls_2 == 2) {
                        int _smem_desc_lo_0 = (((smem_ag_addr) >> 4) & 0x3FFF) + (stage_1) * 2048;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (64))), "r"(((init_sub_1_0) ? 0 : 1)) : "memory");
                        int _smem_desc_lo_2 = (((smem_au_addr) >> 4) & 0x3FFF) + (stage_1) * 2048;
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc + (128))), "r"(((init_sub_1_0) ? 0 : 1)) : "memory");
                    } else {
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 2048;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc), "r"(((init_sub_1_0) ? 0 : 1)) : "memory");
                    }
                    int init_sub_1_1 = 0;
                    if (cls_2 == 2) {
                        int _smem_desc_lo_6 = (((smem_ag_addr + 8192) >> 4) & 0x3FFF) + (stage_1) * 2048;
                        int _smem_desc_lo_7 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_acc + (64))), "r"(((init_sub_1_1) ? 0 : 1)) : "memory");
                        int _smem_desc_lo_8 = (((smem_au_addr + 8192) >> 4) & 0x3FFF) + (stage_1) * 2048;
                        int _smem_desc_lo_9 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"((tmem_acc + (128))), "r"(((init_sub_1_1) ? 0 : 1)) : "memory");
                    } else {
                        int _smem_desc_lo_10 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_1) * 2048;
                        int _smem_desc_lo_11 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc), "r"(((init_sub_1_1) ? 0 : 1)) : "memory");
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
