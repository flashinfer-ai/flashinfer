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

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};






__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_gqa_det_train_f14da11bb54e8296852c(const __grid_constant__ CUtensorMap dsw, const __grid_constant__ CUtensorMap Q, int* __restrict__ tile_base, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, float* __restrict__ dk_out, int num_segments, int h_kv, int group, float softmax_scale, int n0, int h0, int tile_stride, int zero_tile)
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
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 24)
    #define acc_full_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* sa = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sa_addr = smem + 1024;
    __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int sb_addr = smem + 33792;

    // Kernel setup ops
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dsw))) : "memory"); }
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }

    // Mbarrier init (7 barriers at smem_raw[0..56))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // tma_full: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 56);
    if (warp == 4) {
        int _tmem_hold = smem + 56;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int seg = blockIdx.z;
            int rank = cta_rank & 1;
            int n_groups = group >> 1;
            int q0 = cu_seqlens_q[seg];
            int q_len = cu_seqlens_q[seg + 1] - q0;
            int k0 = cu_seqlens_k[seg];
            int k_len = cu_seqlens_k[seg + 1] - k0;
            int off = k_len - q_len;
            int m_tiles = q_len + 63 >> 6;
            int n_blocks = k_len + 127 >> 7;
            int n_pair = n0 + (blockIdx.x >> 1) * 2;
            int my_n = n_pair + rank;
            int m_lo = ((n_pair * 128 > off) ? n_pair * 128 - off : 0) >> 6;
            int per_group = ((n_pair < n_blocks) ? m_tiles - m_lo : 0);
            if (per_group < 0) {
                per_group = 0;
            }
            int mine = ((my_n < n_blocks) ? 1 : 0);
            int my_m_min = ((my_n * 128 > off) ? my_n * 128 - off : 0) >> 6;
            int h = h0 + blockIdx.y;
            const int quarter = warp % 4;
            const unsigned int lane_base = (unsigned int)(quarter * 32 << 16);
            int key = my_n * 128 + quarter * 32 + lane;
            int store_row = ((key < k_len) ? 1 : 0);
            if (mine == 0) {
                store_row = 0;
            }
            long long out_row = (long long)(k0 + key);
            long long base = (out_row * (long long)h_kv + (long long)h) * 256ll;
            mbarrier_wait(acc_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float acc_0[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3]), "=f"(acc_0[4]), "=f"(acc_0[5]), "=f"(acc_0[6]), "=f"(acc_0[7]), "=f"(acc_0[8]), "=f"(acc_0[9]), "=f"(acc_0[10]), "=f"(acc_0[11]), "=f"(acc_0[12]), "=f"(acc_0[13]), "=f"(acc_0[14]), "=f"(acc_0[15]), "=f"(acc_0[16]), "=f"(acc_0[17]), "=f"(acc_0[18]), "=f"(acc_0[19]), "=f"(acc_0[20]), "=f"(acc_0[21]), "=f"(acc_0[22]), "=f"(acc_0[23]), "=f"(acc_0[24]), "=f"(acc_0[25]), "=f"(acc_0[26]), "=f"(acc_0[27]), "=f"(acc_0[28]), "=f"(acc_0[29]), "=f"(acc_0[30]), "=f"(acc_0[31])
                    : "r"(taddr + 0 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_0 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_0)[_ls], _scale2_0);
            } else {
                #pragma unroll
                for (int i_0 = 0; i_0 < 32; i_0++) {
                    acc_0[i_0] = 0.0f;
                }
            }
            long long at_0 = base + 0ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_1 = 0; i_1 < 8; i_1++) {
                    {
                        float4 _v4 = make_float4(acc_0[i_1 * 4 + 0], acc_0[i_1 * 4 + 1], acc_0[i_1 * 4 + 2], acc_0[i_1 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_0 + (long long)(i_1 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_1[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3]), "=f"(acc_1[4]), "=f"(acc_1[5]), "=f"(acc_1[6]), "=f"(acc_1[7]), "=f"(acc_1[8]), "=f"(acc_1[9]), "=f"(acc_1[10]), "=f"(acc_1[11]), "=f"(acc_1[12]), "=f"(acc_1[13]), "=f"(acc_1[14]), "=f"(acc_1[15]), "=f"(acc_1[16]), "=f"(acc_1[17]), "=f"(acc_1[18]), "=f"(acc_1[19]), "=f"(acc_1[20]), "=f"(acc_1[21]), "=f"(acc_1[22]), "=f"(acc_1[23]), "=f"(acc_1[24]), "=f"(acc_1[25]), "=f"(acc_1[26]), "=f"(acc_1[27]), "=f"(acc_1[28]), "=f"(acc_1[29]), "=f"(acc_1[30]), "=f"(acc_1[31])
                    : "r"(taddr + 32 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_1 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_1)[_ls], _scale2_1);
            } else {
                #pragma unroll
                for (int i_2 = 0; i_2 < 32; i_2++) {
                    acc_1[i_2] = 0.0f;
                }
            }
            long long at_1 = base + 32ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_3 = 0; i_3 < 8; i_3++) {
                    {
                        float4 _v4 = make_float4(acc_1[i_3 * 4 + 0], acc_1[i_3 * 4 + 1], acc_1[i_3 * 4 + 2], acc_1[i_3 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_1 + (long long)(i_3 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_2[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_2[0]), "=f"(acc_2[1]), "=f"(acc_2[2]), "=f"(acc_2[3]), "=f"(acc_2[4]), "=f"(acc_2[5]), "=f"(acc_2[6]), "=f"(acc_2[7]), "=f"(acc_2[8]), "=f"(acc_2[9]), "=f"(acc_2[10]), "=f"(acc_2[11]), "=f"(acc_2[12]), "=f"(acc_2[13]), "=f"(acc_2[14]), "=f"(acc_2[15]), "=f"(acc_2[16]), "=f"(acc_2[17]), "=f"(acc_2[18]), "=f"(acc_2[19]), "=f"(acc_2[20]), "=f"(acc_2[21]), "=f"(acc_2[22]), "=f"(acc_2[23]), "=f"(acc_2[24]), "=f"(acc_2[25]), "=f"(acc_2[26]), "=f"(acc_2[27]), "=f"(acc_2[28]), "=f"(acc_2[29]), "=f"(acc_2[30]), "=f"(acc_2[31])
                    : "r"(taddr + 64 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_2 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_2)[_ls], _scale2_2);
            } else {
                #pragma unroll
                for (int i_4 = 0; i_4 < 32; i_4++) {
                    acc_2[i_4] = 0.0f;
                }
            }
            long long at_2 = base + 64ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_5 = 0; i_5 < 8; i_5++) {
                    {
                        float4 _v4 = make_float4(acc_2[i_5 * 4 + 0], acc_2[i_5 * 4 + 1], acc_2[i_5 * 4 + 2], acc_2[i_5 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_2 + (long long)(i_5 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_3[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_3[0]), "=f"(acc_3[1]), "=f"(acc_3[2]), "=f"(acc_3[3]), "=f"(acc_3[4]), "=f"(acc_3[5]), "=f"(acc_3[6]), "=f"(acc_3[7]), "=f"(acc_3[8]), "=f"(acc_3[9]), "=f"(acc_3[10]), "=f"(acc_3[11]), "=f"(acc_3[12]), "=f"(acc_3[13]), "=f"(acc_3[14]), "=f"(acc_3[15]), "=f"(acc_3[16]), "=f"(acc_3[17]), "=f"(acc_3[18]), "=f"(acc_3[19]), "=f"(acc_3[20]), "=f"(acc_3[21]), "=f"(acc_3[22]), "=f"(acc_3[23]), "=f"(acc_3[24]), "=f"(acc_3[25]), "=f"(acc_3[26]), "=f"(acc_3[27]), "=f"(acc_3[28]), "=f"(acc_3[29]), "=f"(acc_3[30]), "=f"(acc_3[31])
                    : "r"(taddr + 96 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_3 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_3)[_ls], _scale2_3);
            } else {
                #pragma unroll
                for (int i_6 = 0; i_6 < 32; i_6++) {
                    acc_3[i_6] = 0.0f;
                }
            }
            long long at_3 = base + 96ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_7 = 0; i_7 < 8; i_7++) {
                    {
                        float4 _v4 = make_float4(acc_3[i_7 * 4 + 0], acc_3[i_7 * 4 + 1], acc_3[i_7 * 4 + 2], acc_3[i_7 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_3 + (long long)(i_7 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_4[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_4[0]), "=f"(acc_4[1]), "=f"(acc_4[2]), "=f"(acc_4[3]), "=f"(acc_4[4]), "=f"(acc_4[5]), "=f"(acc_4[6]), "=f"(acc_4[7]), "=f"(acc_4[8]), "=f"(acc_4[9]), "=f"(acc_4[10]), "=f"(acc_4[11]), "=f"(acc_4[12]), "=f"(acc_4[13]), "=f"(acc_4[14]), "=f"(acc_4[15]), "=f"(acc_4[16]), "=f"(acc_4[17]), "=f"(acc_4[18]), "=f"(acc_4[19]), "=f"(acc_4[20]), "=f"(acc_4[21]), "=f"(acc_4[22]), "=f"(acc_4[23]), "=f"(acc_4[24]), "=f"(acc_4[25]), "=f"(acc_4[26]), "=f"(acc_4[27]), "=f"(acc_4[28]), "=f"(acc_4[29]), "=f"(acc_4[30]), "=f"(acc_4[31])
                    : "r"(taddr + 128 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_4 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_4)[_ls], _scale2_4);
            } else {
                #pragma unroll
                for (int i_8 = 0; i_8 < 32; i_8++) {
                    acc_4[i_8] = 0.0f;
                }
            }
            long long at_4 = base + 128ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_9 = 0; i_9 < 8; i_9++) {
                    {
                        float4 _v4 = make_float4(acc_4[i_9 * 4 + 0], acc_4[i_9 * 4 + 1], acc_4[i_9 * 4 + 2], acc_4[i_9 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_4 + (long long)(i_9 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_5[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_5[0]), "=f"(acc_5[1]), "=f"(acc_5[2]), "=f"(acc_5[3]), "=f"(acc_5[4]), "=f"(acc_5[5]), "=f"(acc_5[6]), "=f"(acc_5[7]), "=f"(acc_5[8]), "=f"(acc_5[9]), "=f"(acc_5[10]), "=f"(acc_5[11]), "=f"(acc_5[12]), "=f"(acc_5[13]), "=f"(acc_5[14]), "=f"(acc_5[15]), "=f"(acc_5[16]), "=f"(acc_5[17]), "=f"(acc_5[18]), "=f"(acc_5[19]), "=f"(acc_5[20]), "=f"(acc_5[21]), "=f"(acc_5[22]), "=f"(acc_5[23]), "=f"(acc_5[24]), "=f"(acc_5[25]), "=f"(acc_5[26]), "=f"(acc_5[27]), "=f"(acc_5[28]), "=f"(acc_5[29]), "=f"(acc_5[30]), "=f"(acc_5[31])
                    : "r"(taddr + 160 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_5 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_5)[_ls], _scale2_5);
            } else {
                #pragma unroll
                for (int i_10 = 0; i_10 < 32; i_10++) {
                    acc_5[i_10] = 0.0f;
                }
            }
            long long at_5 = base + 160ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_11 = 0; i_11 < 8; i_11++) {
                    {
                        float4 _v4 = make_float4(acc_5[i_11 * 4 + 0], acc_5[i_11 * 4 + 1], acc_5[i_11 * 4 + 2], acc_5[i_11 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_5 + (long long)(i_11 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_6[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_6[0]), "=f"(acc_6[1]), "=f"(acc_6[2]), "=f"(acc_6[3]), "=f"(acc_6[4]), "=f"(acc_6[5]), "=f"(acc_6[6]), "=f"(acc_6[7]), "=f"(acc_6[8]), "=f"(acc_6[9]), "=f"(acc_6[10]), "=f"(acc_6[11]), "=f"(acc_6[12]), "=f"(acc_6[13]), "=f"(acc_6[14]), "=f"(acc_6[15]), "=f"(acc_6[16]), "=f"(acc_6[17]), "=f"(acc_6[18]), "=f"(acc_6[19]), "=f"(acc_6[20]), "=f"(acc_6[21]), "=f"(acc_6[22]), "=f"(acc_6[23]), "=f"(acc_6[24]), "=f"(acc_6[25]), "=f"(acc_6[26]), "=f"(acc_6[27]), "=f"(acc_6[28]), "=f"(acc_6[29]), "=f"(acc_6[30]), "=f"(acc_6[31])
                    : "r"(taddr + 192 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_6 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_6)[_ls], _scale2_6);
            } else {
                #pragma unroll
                for (int i_12 = 0; i_12 < 32; i_12++) {
                    acc_6[i_12] = 0.0f;
                }
            }
            long long at_6 = base + 192ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_13 = 0; i_13 < 8; i_13++) {
                    {
                        float4 _v4 = make_float4(acc_6[i_13 * 4 + 0], acc_6[i_13 * 4 + 1], acc_6[i_13 * 4 + 2], acc_6[i_13 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_6 + (long long)(i_13 * 4))) + 0) = _v4;
                    }
                }
            }
            float acc_7[32];
            if (per_group > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc_7[0]), "=f"(acc_7[1]), "=f"(acc_7[2]), "=f"(acc_7[3]), "=f"(acc_7[4]), "=f"(acc_7[5]), "=f"(acc_7[6]), "=f"(acc_7[7]), "=f"(acc_7[8]), "=f"(acc_7[9]), "=f"(acc_7[10]), "=f"(acc_7[11]), "=f"(acc_7[12]), "=f"(acc_7[13]), "=f"(acc_7[14]), "=f"(acc_7[15]), "=f"(acc_7[16]), "=f"(acc_7[17]), "=f"(acc_7[18]), "=f"(acc_7[19]), "=f"(acc_7[20]), "=f"(acc_7[21]), "=f"(acc_7[22]), "=f"(acc_7[23]), "=f"(acc_7[24]), "=f"(acc_7[25]), "=f"(acc_7[26]), "=f"(acc_7[27]), "=f"(acc_7[28]), "=f"(acc_7[29]), "=f"(acc_7[30]), "=f"(acc_7[31])
                    : "r"(taddr + 224 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_7 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_7)[_ls], _scale2_7);
            } else {
                #pragma unroll
                for (int i_14 = 0; i_14 < 32; i_14++) {
                    acc_7[i_14] = 0.0f;
                }
            }
            long long at_7 = base + 224ll;
            if (store_row != 0) {
                #pragma unroll
                for (int i_15 = 0; i_15 < 8; i_15++) {
                    {
                        float4 _v4 = make_float4(acc_7[i_15 * 4 + 0], acc_7[i_15 * 4 + 1], acc_7[i_15 * 4 + 2], acc_7[i_15 * 4 + 3]);
                        *reinterpret_cast<float4*>((dk_out + (at_7 + (long long)(i_15 * 4))) + 0) = _v4;
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int seg_1 = blockIdx.z;
            int rank_1 = cta_rank & 1;
            int n_groups_1 = group >> 1;
            int q0_1 = cu_seqlens_q[seg_1];
            int q_len_1 = cu_seqlens_q[seg_1 + 1] - q0_1;
            int k0_1 = cu_seqlens_k[seg_1];
            int k_len_1 = cu_seqlens_k[seg_1 + 1] - k0_1;
            int off_1 = k_len_1 - q_len_1;
            int m_tiles_1 = q_len_1 + 63 >> 6;
            int n_blocks_1 = k_len_1 + 127 >> 7;
            int n_pair_1 = n0 + (blockIdx.x >> 1) * 2;
            int my_n_1 = n_pair_1 + rank_1;
            int m_lo_1 = ((n_pair_1 * 128 > off_1) ? n_pair_1 * 128 - off_1 : 0) >> 6;
            int per_group_1 = ((n_pair_1 < n_blocks_1) ? m_tiles_1 - m_lo_1 : 0);
            if (per_group_1 < 0) {
                per_group_1 = 0;
            }
            int mine_1 = ((my_n_1 < n_blocks_1) ? 1 : 0);
            int my_m_min_1 = ((my_n_1 * 128 > off_1) ? my_n_1 * 128 - off_1 : 0) >> 6;
            int steps = n_groups_1 * per_group_1;
            unsigned int mc = 0;
            unsigned int _phase_tma_full = 0;
            if ((cta_rank & 1) == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int s = 0; s < steps; s++) {
                        int first = ((s == 0) ? 1 : 0);
                        mbarrier_wait_cluster_hint(tma_full_addr + (mc) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = (((sa_addr) >> 4) & 0x7FFF) + (mc) * 4096;
                        int _smem_desc_lo_1 = ((((sb_addr) >> 4) & 0x7FFF) | 0x4000000) + (mc) * 4096;
#else
                        int _smem_desc_lo_0 = (((sa_addr) >> 4) & 0x3FFF) + (mc) * 4096;
                        int _smem_desc_lo_1 = ((((sb_addr) >> 4) & 0x3FFF) | 0x4000000) + (mc) * 4096;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272696464;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc), "r"(((first) ? 0 : 1)) : "memory");
                        tcgen05_commit_cg2_multicast(mma_done_addr + (mc) * 8, (uint16_t)(3));
                        mc += 1;
                        if (mc == 3) { mc = 0; _phase_tma_full ^= 1; }
                    }
                    tcgen05_commit_cg2_multicast(acc_full_addr, (uint16_t)(3));
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            int seg_2 = blockIdx.z;
            int rank_2 = cta_rank & 1;
            int n_groups_2 = group >> 1;
            int q0_2 = cu_seqlens_q[seg_2];
            int q_len_2 = cu_seqlens_q[seg_2 + 1] - q0_2;
            int k0_2 = cu_seqlens_k[seg_2];
            int k_len_2 = cu_seqlens_k[seg_2 + 1] - k0_2;
            int off_2 = k_len_2 - q_len_2;
            int m_tiles_2 = q_len_2 + 63 >> 6;
            int n_blocks_2 = k_len_2 + 127 >> 7;
            int n_pair_2 = n0 + (blockIdx.x >> 1) * 2;
            int my_n_2 = n_pair_2 + rank_2;
            int m_lo_2 = ((n_pair_2 * 128 > off_2) ? n_pair_2 * 128 - off_2 : 0) >> 6;
            int per_group_2 = ((n_pair_2 < n_blocks_2) ? m_tiles_2 - m_lo_2 : 0);
            if (per_group_2 < 0) {
                per_group_2 = 0;
            }
            int mine_2 = ((my_n_2 < n_blocks_2) ? 1 : 0);
            int my_m_min_2 = ((my_n_2 * 128 > off_2) ? my_n_2 * 128 - off_2 : 0) >> 6;
            int hl = blockIdx.y;
            int h_1 = h0 + hl;
            unsigned int lc = 0;
            int block_tile = 0;
            if (mine_2 != 0) {
                block_tile = tile_base[my_n_2 * num_segments + seg_2] - tile_base[n0 * num_segments];
            }
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int g = 0; g < n_groups_2; g++) {
                    int head0 = h_1 * group + g * 2;
                    int group_tile = (hl * n_groups_2 + g) * tile_stride + block_tile;
                    #pragma unroll 1
                    for (int mi = 0; mi < per_group_2; mi++) {
                        int m = m_lo_2 + mi;
                        int tile = zero_tile;
                        if (mine_2 != 0) {
                            if (m >= my_m_min_2) {
                                tile = group_tile + (m - my_m_min_2);
                            }
                        }
                        mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (lc) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                        tma_2d_gmem2smem_cta2(sa_addr + lc * 65536, (&dsw), 0, tile * 256, ((tma_full_addr + (lc) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sb_addr + lc * 65536, (&Q), rank_2 * 128, head0, q0_2 + m * 64, ((tma_full_addr + (lc) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sb_addr + lc * 65536 + 8192, (&Q), rank_2 * 128, head0, q0_2 + m * 64 + 32, ((tma_full_addr + (lc) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sb_addr + lc * 65536 + 16384, (&Q), rank_2 * 128 + 64, head0, q0_2 + m * 64, ((tma_full_addr + (lc) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sb_addr + lc * 65536 + 24576, (&Q), rank_2 * 128 + 64, head0, q0_2 + m * 64 + 32, ((tma_full_addr + (lc) * 8) & 0xFEFFFFFF));
                        lc += 1;
                        if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
                    }
                }
                mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                lc += 1;
                if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
                mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                lc += 1;
                if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
                mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                lc += 1;
                if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
