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
#define TMEM_NCOLS 512
#define TMEM_X_ACC_OFFSET 0
#define TMEM_X_SPLIT_OFFSET 128
#define TMEM_B_HI_OFFSET 384
#define TMEM_B_LO_OFFSET 448
#define NUM_RING_STAGES 2
#define NUM_FRING_STAGES 1
#define NUM_ONE_STAGES 1
#define SMEM_W_PANEL_OFF 1024
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_K_PANEL_OFF 33792
#define SMEM_K_PANEL_STAGE_BYTES 8192
#define SMEM_K_PANEL_STRIDE 8192
#define SMEM_W_LAND_OFF 1024
#define SMEM_W_LAND_STAGE_BYTES 16384
#define SMEM_W_LAND_STRIDE 16384
#define SMEM_K_LAND_OFF 33792
#define SMEM_K_LAND_STAGE_BYTES 16384
#define SMEM_K_LAND_STRIDE 16384
#define SMEM_WF_OFF 66560
#define SMEM_WF_STAGE_BYTES 32768
#define SMEM_WF_STRIDE 32768
#define SMEM_KF_OFF 99328
#define SMEM_KF_STAGE_BYTES 32768
#define SMEM_KF_STRIDE 32768
#define SMEM_WF_LO_OFF 132096
#define SMEM_WF_LO_STAGE_BYTES 32768
#define SMEM_WF_LO_STRIDE 32768
#define SMEM_KF_LO_OFF 164864
#define SMEM_KF_LO_STAGE_BYTES 32768
#define SMEM_KF_LO_STRIDE 32768
#define SMEM_RSB_OFF 197632
#define SMEM_RSB_STAGE_BYTES 512
#define SMEM_RSB_STRIDE 512
#define SMEM_TOTAL 198144
#define THREADS 192
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_092cee18b83fb8f48a43(const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap w_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ rowscale, float* __restrict__ gamma, int num_heads, int num_qk_heads, int* __restrict__ chunk_bos, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define f32_full_addr (mbar_base + 32)
    #define f32_empty_addr (mbar_base + 40)
    #define xs_ready_addr (mbar_base + 48)
    #define vp_ready_addr (mbar_base + 56)
    #define bv_ready_addr (mbar_base + 64)
    #define h_ready_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* k_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_panel_addr = smem + 33792;
    __nv_bfloat16* w_land = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_land_addr = smem + 1024;
    __nv_bfloat16* k_land = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_land_addr = smem + 33792;
    float* wf = reinterpret_cast<float*>(smem_raw + 66560);
    const int wf_addr = smem + 66560;
    float* kf = reinterpret_cast<float*>(smem_raw + 99328);
    const int kf_addr = smem + 99328;
    float* wf_lo = reinterpret_cast<float*>(smem_raw + 132096);
    const int wf_lo_addr = smem + 132096;
    float* kf_lo = reinterpret_cast<float*>(smem_raw + 164864);
    const int kf_lo_addr = smem + 164864;
    float* rsb = reinterpret_cast<float*>(smem_raw + 197632);
    const int rsb_addr = smem + 197632;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 16, 128);
            mbarrier_init(smem + 24, 128);
            // --- pipeline 'fring' ---
            // f32_full: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // f32_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'one' ---
            // xs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_x_acc = taddr;
    const int tmem_x_split = taddr + 128;
    const int tmem_b_hi = taddr + 384;
    const int tmem_b_lo = taddr + 448;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seg = walk_seg[blockIdx.x / num_heads];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int lane_v = warp_in_wg * 32 + lane;
            int row_base = warp_in_wg * 32 << 16;
            unsigned int ls = 0;
            int w_row = warp_in_wg * 16 + (lane >> 1);
            int w_col = (lane & 1) * 64;
            int t_col = (warp_in_wg & 1) * 32 + lane;
            int k_row0 = (warp_in_wg >> 1) * 64;
            unsigned int mask_hi = 4294901760u;
            unsigned int mask_tf32 = 4294959104u;
            unsigned int half_tf32 = 4096;
            float init_vals[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                float one_here = 0.0f;
                if (init_mode == 0) {
                    if (j == lane_v) {
                        one_here = 1.0f;
                    }
                }
                init_vals[j] = one_here;
            }
            tmem_st_x32_f32(taddr + (unsigned int)row_base, init_vals);
            #pragma unroll
            for (int j_0 = 0; j_0 < 32; j_0++) {
                float one_here_0 = 0.0f;
                if (init_mode == 0) {
                    if (32 + j_0 == lane_v) {
                        one_here_0 = 1.0f;
                    }
                }
                init_vals[j_0] = one_here_0;
            }
            tmem_st_x32_f32(taddr + (unsigned int)row_base + 32, init_vals);
            #pragma unroll
            for (int j_2 = 0; j_2 < 32; j_2++) {
                float one_here_2 = 0.0f;
                if (init_mode == 0) {
                    if (64 + j_2 == lane_v) {
                        one_here_2 = 1.0f;
                    }
                }
                init_vals[j_2] = one_here_2;
            }
            tmem_st_x32_f32(taddr + (unsigned int)row_base + 64, init_vals);
            #pragma unroll
            for (int j_3 = 0; j_3 < 32; j_3++) {
                float one_here_3 = 0.0f;
                if (init_mode == 0) {
                    if (96 + j_3 == lane_v) {
                        one_here_3 = 1.0f;
                    }
                }
                init_vals[j_3] = one_here_3;
            }
            tmem_st_x32_f32(taddr + (unsigned int)row_base + 96, init_vals);
            float zeros[32];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            zeros[8] = 0.0f;
            zeros[9] = 0.0f;
            zeros[10] = 0.0f;
            zeros[11] = 0.0f;
            zeros[12] = 0.0f;
            zeros[13] = 0.0f;
            zeros[14] = 0.0f;
            zeros[15] = 0.0f;
            zeros[16] = 0.0f;
            zeros[17] = 0.0f;
            zeros[18] = 0.0f;
            zeros[19] = 0.0f;
            zeros[20] = 0.0f;
            zeros[21] = 0.0f;
            zeros[22] = 0.0f;
            zeros[23] = 0.0f;
            zeros[24] = 0.0f;
            zeros[25] = 0.0f;
            zeros[26] = 0.0f;
            zeros[27] = 0.0f;
            zeros[28] = 0.0f;
            zeros[29] = 0.0f;
            zeros[30] = 0.0f;
            zeros[31] = 0.0f;
            tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + 128, zeros);
            tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + 160, zeros);
            tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + 192, zeros);
            tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + 224, zeros);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int _phase_f32_empty = 1;
            unsigned int _phase_tiles_full = 0;
            if (nc > 0) {
                mbarrier_wait(f32_empty_addr, _phase_f32_empty);
                mbarrier_wait(tiles_full_addr + (ls) * 8, _phase_tiles_full);
                unsigned int words[4];
                float wide[8];
                unsigned int wide_hi[8];
                unsigned int wide_lo[8];
                int col_7_0 = w_col;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_0 / 64 * 8192 + w_row * 128 + col_7_0 % 64 * 2 ^ (col_7_0 / 64 * 8192 + w_row * 128 + col_7_0 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    wide_hi[j_1] = __as_u32(wide[j_1]) + half_tf32 & mask_tf32;
                    wide_lo[j_1] = __as_u32(wide[j_1] - __uint_as_float(wide_hi[j_1])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_0 / 32 * 8192 + w_row * 128 + col_7_0 % 32 * 4 ^ (col_7_0 / 32 * 8192 + w_row * 128 + col_7_0 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_0 / 32 * 8192 + w_row * 128 + col_7_0 % 32 * 4 ^ (col_7_0 / 32 * 8192 + w_row * 128 + col_7_0 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_0 + 4) / 32 * 8192 + w_row * 128 + (col_7_0 + 4) % 32 * 4 ^ ((col_7_0 + 4) / 32 * 8192 + w_row * 128 + (col_7_0 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_0 + 4) / 32 * 8192 + w_row * 128 + (col_7_0 + 4) % 32 * 4 ^ ((col_7_0 + 4) / 32 * 8192 + w_row * 128 + (col_7_0 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_1 = w_col + 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_1 / 64 * 8192 + w_row * 128 + col_7_1 % 64 * 2 ^ (col_7_1 / 64 * 8192 + w_row * 128 + col_7_1 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    wide_hi[j_4] = __as_u32(wide[j_4]) + half_tf32 & mask_tf32;
                    wide_lo[j_4] = __as_u32(wide[j_4] - __uint_as_float(wide_hi[j_4])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_1 / 32 * 8192 + w_row * 128 + col_7_1 % 32 * 4 ^ (col_7_1 / 32 * 8192 + w_row * 128 + col_7_1 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_1 / 32 * 8192 + w_row * 128 + col_7_1 % 32 * 4 ^ (col_7_1 / 32 * 8192 + w_row * 128 + col_7_1 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_1 + 4) / 32 * 8192 + w_row * 128 + (col_7_1 + 4) % 32 * 4 ^ ((col_7_1 + 4) / 32 * 8192 + w_row * 128 + (col_7_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_1 + 4) / 32 * 8192 + w_row * 128 + (col_7_1 + 4) % 32 * 4 ^ ((col_7_1 + 4) / 32 * 8192 + w_row * 128 + (col_7_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_2 = w_col + 16;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_2 / 64 * 8192 + w_row * 128 + col_7_2 % 64 * 2 ^ (col_7_2 / 64 * 8192 + w_row * 128 + col_7_2 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    wide_hi[j_5] = __as_u32(wide[j_5]) + half_tf32 & mask_tf32;
                    wide_lo[j_5] = __as_u32(wide[j_5] - __uint_as_float(wide_hi[j_5])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_2 / 32 * 8192 + w_row * 128 + col_7_2 % 32 * 4 ^ (col_7_2 / 32 * 8192 + w_row * 128 + col_7_2 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_2 / 32 * 8192 + w_row * 128 + col_7_2 % 32 * 4 ^ (col_7_2 / 32 * 8192 + w_row * 128 + col_7_2 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_2 + 4) / 32 * 8192 + w_row * 128 + (col_7_2 + 4) % 32 * 4 ^ ((col_7_2 + 4) / 32 * 8192 + w_row * 128 + (col_7_2 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_2 + 4) / 32 * 8192 + w_row * 128 + (col_7_2 + 4) % 32 * 4 ^ ((col_7_2 + 4) / 32 * 8192 + w_row * 128 + (col_7_2 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_3 = w_col + 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_3 / 64 * 8192 + w_row * 128 + col_7_3 % 64 * 2 ^ (col_7_3 / 64 * 8192 + w_row * 128 + col_7_3 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_6 = 0; j_6 < 8; j_6++) {
                    wide_hi[j_6] = __as_u32(wide[j_6]) + half_tf32 & mask_tf32;
                    wide_lo[j_6] = __as_u32(wide[j_6] - __uint_as_float(wide_hi[j_6])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_3 / 32 * 8192 + w_row * 128 + col_7_3 % 32 * 4 ^ (col_7_3 / 32 * 8192 + w_row * 128 + col_7_3 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_3 / 32 * 8192 + w_row * 128 + col_7_3 % 32 * 4 ^ (col_7_3 / 32 * 8192 + w_row * 128 + col_7_3 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_3 + 4) / 32 * 8192 + w_row * 128 + (col_7_3 + 4) % 32 * 4 ^ ((col_7_3 + 4) / 32 * 8192 + w_row * 128 + (col_7_3 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_3 + 4) / 32 * 8192 + w_row * 128 + (col_7_3 + 4) % 32 * 4 ^ ((col_7_3 + 4) / 32 * 8192 + w_row * 128 + (col_7_3 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_4 = w_col + 32;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_4 / 64 * 8192 + w_row * 128 + col_7_4 % 64 * 2 ^ (col_7_4 / 64 * 8192 + w_row * 128 + col_7_4 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    wide_hi[j_7] = __as_u32(wide[j_7]) + half_tf32 & mask_tf32;
                    wide_lo[j_7] = __as_u32(wide[j_7] - __uint_as_float(wide_hi[j_7])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_4 / 32 * 8192 + w_row * 128 + col_7_4 % 32 * 4 ^ (col_7_4 / 32 * 8192 + w_row * 128 + col_7_4 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_4 / 32 * 8192 + w_row * 128 + col_7_4 % 32 * 4 ^ (col_7_4 / 32 * 8192 + w_row * 128 + col_7_4 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_4 + 4) / 32 * 8192 + w_row * 128 + (col_7_4 + 4) % 32 * 4 ^ ((col_7_4 + 4) / 32 * 8192 + w_row * 128 + (col_7_4 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_4 + 4) / 32 * 8192 + w_row * 128 + (col_7_4 + 4) % 32 * 4 ^ ((col_7_4 + 4) / 32 * 8192 + w_row * 128 + (col_7_4 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_5 = w_col + 40;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_5 / 64 * 8192 + w_row * 128 + col_7_5 % 64 * 2 ^ (col_7_5 / 64 * 8192 + w_row * 128 + col_7_5 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    wide_hi[j_8] = __as_u32(wide[j_8]) + half_tf32 & mask_tf32;
                    wide_lo[j_8] = __as_u32(wide[j_8] - __uint_as_float(wide_hi[j_8])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_5 / 32 * 8192 + w_row * 128 + col_7_5 % 32 * 4 ^ (col_7_5 / 32 * 8192 + w_row * 128 + col_7_5 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_5 / 32 * 8192 + w_row * 128 + col_7_5 % 32 * 4 ^ (col_7_5 / 32 * 8192 + w_row * 128 + col_7_5 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_5 + 4) / 32 * 8192 + w_row * 128 + (col_7_5 + 4) % 32 * 4 ^ ((col_7_5 + 4) / 32 * 8192 + w_row * 128 + (col_7_5 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_5 + 4) / 32 * 8192 + w_row * 128 + (col_7_5 + 4) % 32 * 4 ^ ((col_7_5 + 4) / 32 * 8192 + w_row * 128 + (col_7_5 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_6 = w_col + 48;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_6 / 64 * 8192 + w_row * 128 + col_7_6 % 64 * 2 ^ (col_7_6 / 64 * 8192 + w_row * 128 + col_7_6 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_9 = 0; j_9 < 8; j_9++) {
                    wide_hi[j_9] = __as_u32(wide[j_9]) + half_tf32 & mask_tf32;
                    wide_lo[j_9] = __as_u32(wide[j_9] - __uint_as_float(wide_hi[j_9])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_6 / 32 * 8192 + w_row * 128 + col_7_6 % 32 * 4 ^ (col_7_6 / 32 * 8192 + w_row * 128 + col_7_6 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_6 / 32 * 8192 + w_row * 128 + col_7_6 % 32 * 4 ^ (col_7_6 / 32 * 8192 + w_row * 128 + col_7_6 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_6 + 4) / 32 * 8192 + w_row * 128 + (col_7_6 + 4) % 32 * 4 ^ ((col_7_6 + 4) / 32 * 8192 + w_row * 128 + (col_7_6 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_6 + 4) / 32 * 8192 + w_row * 128 + (col_7_6 + 4) % 32 * 4 ^ ((col_7_6 + 4) / 32 * 8192 + w_row * 128 + (col_7_6 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int col_7_7 = w_col + 56;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_7 / 64 * 8192 + w_row * 128 + col_7_7 % 64 * 2 ^ (col_7_7 / 64 * 8192 + w_row * 128 + col_7_7 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    wide_hi[j_10] = __as_u32(wide[j_10]) + half_tf32 & mask_tf32;
                    wide_lo[j_10] = __as_u32(wide[j_10] - __uint_as_float(wide_hi[j_10])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_7 / 32 * 8192 + w_row * 128 + col_7_7 % 32 * 4 ^ (col_7_7 / 32 * 8192 + w_row * 128 + col_7_7 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]), "r"(wide_hi[1]), "r"(wide_hi[2]), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_7 / 32 * 8192 + w_row * 128 + col_7_7 % 32 * 4 ^ (col_7_7 / 32 * 8192 + w_row * 128 + col_7_7 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]), "r"(wide_lo[1]), "r"(wide_lo[2]), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_7 + 4) / 32 * 8192 + w_row * 128 + (col_7_7 + 4) % 32 * 4 ^ ((col_7_7 + 4) / 32 * 8192 + w_row * 128 + (col_7_7 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]), "r"(wide_hi[5]), "r"(wide_hi[6]), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_7 + 4) / 32 * 8192 + w_row * 128 + (col_7_7 + 4) % 32 * 4 ^ ((col_7_7 + 4) / 32 * 8192 + w_row * 128 + (col_7_7 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]), "r"(wide_lo[5]), "r"(wide_lo[6]), "r"(wide_lo[7]) : "memory");
                int krow_1_0 = k_row0;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_0 / 64 * 8192 + t_col * 128 + krow_1_0 % 64 * 2 ^ (krow_1_0 / 64 * 8192 + t_col * 128 + krow_1_0 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_11 = 0; j_11 < 8; j_11++) {
                    wide_hi[j_11] = __as_u32(wide[j_11]) + half_tf32 & mask_tf32;
                    wide_lo[j_11] = __as_u32(wide[j_11] - __uint_as_float(wide_hi[j_11])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_0 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_0 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_0 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_0 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_1 = k_row0 + 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_1 / 64 * 8192 + t_col * 128 + krow_1_1 % 64 * 2 ^ (krow_1_1 / 64 * 8192 + t_col * 128 + krow_1_1 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_12 = 0; j_12 < 8; j_12++) {
                    wide_hi[j_12] = __as_u32(wide[j_12]) + half_tf32 & mask_tf32;
                    wide_lo[j_12] = __as_u32(wide[j_12] - __uint_as_float(wide_hi[j_12])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_2 = k_row0 + 16;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_2 / 64 * 8192 + t_col * 128 + krow_1_2 % 64 * 2 ^ (krow_1_2 / 64 * 8192 + t_col * 128 + krow_1_2 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_13 = 0; j_13 < 8; j_13++) {
                    wide_hi[j_13] = __as_u32(wide[j_13]) + half_tf32 & mask_tf32;
                    wide_lo[j_13] = __as_u32(wide[j_13] - __uint_as_float(wide_hi[j_13])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_2 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_2 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_2 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_2 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_3 = k_row0 + 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_3 / 64 * 8192 + t_col * 128 + krow_1_3 % 64 * 2 ^ (krow_1_3 / 64 * 8192 + t_col * 128 + krow_1_3 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_14 = 0; j_14 < 8; j_14++) {
                    wide_hi[j_14] = __as_u32(wide[j_14]) + half_tf32 & mask_tf32;
                    wide_lo[j_14] = __as_u32(wide[j_14] - __uint_as_float(wide_hi[j_14])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_3 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_3 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_3 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_3 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_4 = k_row0 + 32;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_4 / 64 * 8192 + t_col * 128 + krow_1_4 % 64 * 2 ^ (krow_1_4 / 64 * 8192 + t_col * 128 + krow_1_4 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_15 = 0; j_15 < 8; j_15++) {
                    wide_hi[j_15] = __as_u32(wide[j_15]) + half_tf32 & mask_tf32;
                    wide_lo[j_15] = __as_u32(wide[j_15] - __uint_as_float(wide_hi[j_15])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_4 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_4 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_4 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_4 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_5 = k_row0 + 40;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_5 / 64 * 8192 + t_col * 128 + krow_1_5 % 64 * 2 ^ (krow_1_5 / 64 * 8192 + t_col * 128 + krow_1_5 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_16 = 0; j_16 < 8; j_16++) {
                    wide_hi[j_16] = __as_u32(wide[j_16]) + half_tf32 & mask_tf32;
                    wide_lo[j_16] = __as_u32(wide[j_16] - __uint_as_float(wide_hi[j_16])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_5 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_5 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_5 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_5 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_6 = k_row0 + 48;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_6 / 64 * 8192 + t_col * 128 + krow_1_6 % 64 * 2 ^ (krow_1_6 / 64 * 8192 + t_col * 128 + krow_1_6 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_17 = 0; j_17 < 8; j_17++) {
                    wide_hi[j_17] = __as_u32(wide[j_17]) + half_tf32 & mask_tf32;
                    wide_lo[j_17] = __as_u32(wide[j_17] - __uint_as_float(wide_hi[j_17])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_6 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_6 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_6 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_6 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                int krow_1_7 = k_row0 + 56;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_7 / 64 * 8192 + t_col * 128 + krow_1_7 % 64 * 2 ^ (krow_1_7 / 64 * 8192 + t_col * 128 + krow_1_7 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                #pragma unroll
                for (int j_18 = 0; j_18 < 8; j_18++) {
                    wide_hi[j_18] = __as_u32(wide[j_18]) + half_tf32 & mask_tf32;
                    wide_lo[j_18] = __as_u32(wide[j_18] - __uint_as_float(wide_hi[j_18])) + half_tf32 & mask_tf32;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_7 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_7 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_7 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_7 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi[7]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(f32_full_addr);
                mbarrier_arrive(tiles_empty_addr + (ls) * 8);
                ls += 1;
                if (ls == 2) { ls = 0; _phase_tiles_full ^= 1; }
                _phase_f32_empty ^= 1;
            }
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int par = c % 2;
                long long g_index = (long long)(c0 + c) * (long long)num_heads + (long long)head;
                float decay = gamma[g_index];
                if (lane_v < 64) {
                    long long rs_index = (long long)(row0 + c * 64 + lane_v) * (long long)num_heads + (long long)head;
                    float my_rs = rowscale[rs_index];
                    {
                        uint32_t _addr_0 = static_cast<uint32_t>(rsb_addr + (unsigned int)(par * 256 + lane_v * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(my_rs) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_3[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(taddr + (unsigned int)row_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                    : "r"(taddr + (unsigned int)row_base + 32));
                float _tmem_load_13_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_13_0[0]), "=f"(_tmem_load_13_0[1]), "=f"(_tmem_load_13_0[2]), "=f"(_tmem_load_13_0[3]), "=f"(_tmem_load_13_0[4]), "=f"(_tmem_load_13_0[5]), "=f"(_tmem_load_13_0[6]), "=f"(_tmem_load_13_0[7]), "=f"(_tmem_load_13_0[8]), "=f"(_tmem_load_13_0[9]), "=f"(_tmem_load_13_0[10]), "=f"(_tmem_load_13_0[11]), "=f"(_tmem_load_13_0[12]), "=f"(_tmem_load_13_0[13]), "=f"(_tmem_load_13_0[14]), "=f"(_tmem_load_13_0[15]), "=f"(_tmem_load_13_0[16]), "=f"(_tmem_load_13_0[17]), "=f"(_tmem_load_13_0[18]), "=f"(_tmem_load_13_0[19]), "=f"(_tmem_load_13_0[20]), "=f"(_tmem_load_13_0[21]), "=f"(_tmem_load_13_0[22]), "=f"(_tmem_load_13_0[23]), "=f"(_tmem_load_13_0[24]), "=f"(_tmem_load_13_0[25]), "=f"(_tmem_load_13_0[26]), "=f"(_tmem_load_13_0[27]), "=f"(_tmem_load_13_0[28]), "=f"(_tmem_load_13_0[29]), "=f"(_tmem_load_13_0[30]), "=f"(_tmem_load_13_0[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_13_0[32]), "=f"(_tmem_load_13_0[33]), "=f"(_tmem_load_13_0[34]), "=f"(_tmem_load_13_0[35]), "=f"(_tmem_load_13_0[36]), "=f"(_tmem_load_13_0[37]), "=f"(_tmem_load_13_0[38]), "=f"(_tmem_load_13_0[39]), "=f"(_tmem_load_13_0[40]), "=f"(_tmem_load_13_0[41]), "=f"(_tmem_load_13_0[42]), "=f"(_tmem_load_13_0[43]), "=f"(_tmem_load_13_0[44]), "=f"(_tmem_load_13_0[45]), "=f"(_tmem_load_13_0[46]), "=f"(_tmem_load_13_0[47]), "=f"(_tmem_load_13_0[48]), "=f"(_tmem_load_13_0[49]), "=f"(_tmem_load_13_0[50]), "=f"(_tmem_load_13_0[51]), "=f"(_tmem_load_13_0[52]), "=f"(_tmem_load_13_0[53]), "=f"(_tmem_load_13_0[54]), "=f"(_tmem_load_13_0[55]), "=f"(_tmem_load_13_0[56]), "=f"(_tmem_load_13_0[57]), "=f"(_tmem_load_13_0[58]), "=f"(_tmem_load_13_0[59]), "=f"(_tmem_load_13_0[60]), "=f"(_tmem_load_13_0[61]), "=f"(_tmem_load_13_0[62]), "=f"(_tmem_load_13_0[63])
                    : "r"(taddr + 128 + (unsigned int)row_base + 128 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_0 = 0; j_1_0 < 64; j_1_0++) {
                    _tmem_load_3[j_1_0] = _tmem_load_3[j_1_0] + _tmem_load_13_0[j_1_0];
                }
                unsigned int hi_1_0[32];
                unsigned int lo_2_0[32];
                #pragma unroll
                for (int j_19 = 0; j_19 < 32; j_19++) {
                    hi_1_0[j_19] = __as_u32(_tmem_load_3[j_19]) + half_tf32 & mask_tf32;
                    lo_2_0[j_19] = __as_u32(_tmem_load_3[j_19] - __uint_as_float(hi_1_0[j_19])) + half_tf32 & mask_tf32;
                }
                float xd_1_0[32];
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 32; j_1_1++) {
                    xd_1_0[j_1_1] = _tmem_load_3[j_1_1] * decay;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base), "r"(hi_1_0[0]), "r"(hi_1_0[1]), "r"(hi_1_0[2]), "r"(hi_1_0[3]), "r"(hi_1_0[4]), "r"(hi_1_0[5]), "r"(hi_1_0[6]), "r"(hi_1_0[7]), "r"(hi_1_0[8]), "r"(hi_1_0[9]), "r"(hi_1_0[10]), "r"(hi_1_0[11]), "r"(hi_1_0[12]), "r"(hi_1_0[13]), "r"(hi_1_0[14]), "r"(hi_1_0[15]), "r"(hi_1_0[16]), "r"(hi_1_0[17]), "r"(hi_1_0[18]), "r"(hi_1_0[19]), "r"(hi_1_0[20]), "r"(hi_1_0[21]), "r"(hi_1_0[22]), "r"(hi_1_0[23]), "r"(hi_1_0[24]), "r"(hi_1_0[25]), "r"(hi_1_0[26]), "r"(hi_1_0[27]), "r"(hi_1_0[28]), "r"(hi_1_0[29]), "r"(hi_1_0[30]), "r"(hi_1_0[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 128), "r"(lo_2_0[0]), "r"(lo_2_0[1]), "r"(lo_2_0[2]), "r"(lo_2_0[3]), "r"(lo_2_0[4]), "r"(lo_2_0[5]), "r"(lo_2_0[6]), "r"(lo_2_0[7]), "r"(lo_2_0[8]), "r"(lo_2_0[9]), "r"(lo_2_0[10]), "r"(lo_2_0[11]), "r"(lo_2_0[12]), "r"(lo_2_0[13]), "r"(lo_2_0[14]), "r"(lo_2_0[15]), "r"(lo_2_0[16]), "r"(lo_2_0[17]), "r"(lo_2_0[18]), "r"(lo_2_0[19]), "r"(lo_2_0[20]), "r"(lo_2_0[21]), "r"(lo_2_0[22]), "r"(lo_2_0[23]), "r"(lo_2_0[24]), "r"(lo_2_0[25]), "r"(lo_2_0[26]), "r"(lo_2_0[27]), "r"(lo_2_0[28]), "r"(lo_2_0[29]), "r"(lo_2_0[30]), "r"(lo_2_0[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base, xd_1_0);
                unsigned int hi_2_0[32];
                unsigned int lo_3_0[32];
                #pragma unroll
                for (int j_20 = 0; j_20 < 32; j_20++) {
                    hi_2_0[j_20] = __as_u32(_tmem_load_3[32 + j_20]) + half_tf32 & mask_tf32;
                    lo_3_0[j_20] = __as_u32(_tmem_load_3[32 + j_20] - __uint_as_float(hi_2_0[j_20])) + half_tf32 & mask_tf32;
                }
                float xd_3_0[32];
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 32; j_1_2++) {
                    xd_3_0[j_1_2] = _tmem_load_3[32 + j_1_2] * decay;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 32), "r"(hi_2_0[0]), "r"(hi_2_0[1]), "r"(hi_2_0[2]), "r"(hi_2_0[3]), "r"(hi_2_0[4]), "r"(hi_2_0[5]), "r"(hi_2_0[6]), "r"(hi_2_0[7]), "r"(hi_2_0[8]), "r"(hi_2_0[9]), "r"(hi_2_0[10]), "r"(hi_2_0[11]), "r"(hi_2_0[12]), "r"(hi_2_0[13]), "r"(hi_2_0[14]), "r"(hi_2_0[15]), "r"(hi_2_0[16]), "r"(hi_2_0[17]), "r"(hi_2_0[18]), "r"(hi_2_0[19]), "r"(hi_2_0[20]), "r"(hi_2_0[21]), "r"(hi_2_0[22]), "r"(hi_2_0[23]), "r"(hi_2_0[24]), "r"(hi_2_0[25]), "r"(hi_2_0[26]), "r"(hi_2_0[27]), "r"(hi_2_0[28]), "r"(hi_2_0[29]), "r"(hi_2_0[30]), "r"(hi_2_0[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 160), "r"(lo_3_0[0]), "r"(lo_3_0[1]), "r"(lo_3_0[2]), "r"(lo_3_0[3]), "r"(lo_3_0[4]), "r"(lo_3_0[5]), "r"(lo_3_0[6]), "r"(lo_3_0[7]), "r"(lo_3_0[8]), "r"(lo_3_0[9]), "r"(lo_3_0[10]), "r"(lo_3_0[11]), "r"(lo_3_0[12]), "r"(lo_3_0[13]), "r"(lo_3_0[14]), "r"(lo_3_0[15]), "r"(lo_3_0[16]), "r"(lo_3_0[17]), "r"(lo_3_0[18]), "r"(lo_3_0[19]), "r"(lo_3_0[20]), "r"(lo_3_0[21]), "r"(lo_3_0[22]), "r"(lo_3_0[23]), "r"(lo_3_0[24]), "r"(lo_3_0[25]), "r"(lo_3_0[26]), "r"(lo_3_0[27]), "r"(lo_3_0[28]), "r"(lo_3_0[29]), "r"(lo_3_0[30]), "r"(lo_3_0[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base + 32, xd_3_0);
                float _tmem_load_14[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_14[0]), "=f"(_tmem_load_14[1]), "=f"(_tmem_load_14[2]), "=f"(_tmem_load_14[3]), "=f"(_tmem_load_14[4]), "=f"(_tmem_load_14[5]), "=f"(_tmem_load_14[6]), "=f"(_tmem_load_14[7]), "=f"(_tmem_load_14[8]), "=f"(_tmem_load_14[9]), "=f"(_tmem_load_14[10]), "=f"(_tmem_load_14[11]), "=f"(_tmem_load_14[12]), "=f"(_tmem_load_14[13]), "=f"(_tmem_load_14[14]), "=f"(_tmem_load_14[15]), "=f"(_tmem_load_14[16]), "=f"(_tmem_load_14[17]), "=f"(_tmem_load_14[18]), "=f"(_tmem_load_14[19]), "=f"(_tmem_load_14[20]), "=f"(_tmem_load_14[21]), "=f"(_tmem_load_14[22]), "=f"(_tmem_load_14[23]), "=f"(_tmem_load_14[24]), "=f"(_tmem_load_14[25]), "=f"(_tmem_load_14[26]), "=f"(_tmem_load_14[27]), "=f"(_tmem_load_14[28]), "=f"(_tmem_load_14[29]), "=f"(_tmem_load_14[30]), "=f"(_tmem_load_14[31])
                    : "r"(taddr + (unsigned int)row_base + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_14[32]), "=f"(_tmem_load_14[33]), "=f"(_tmem_load_14[34]), "=f"(_tmem_load_14[35]), "=f"(_tmem_load_14[36]), "=f"(_tmem_load_14[37]), "=f"(_tmem_load_14[38]), "=f"(_tmem_load_14[39]), "=f"(_tmem_load_14[40]), "=f"(_tmem_load_14[41]), "=f"(_tmem_load_14[42]), "=f"(_tmem_load_14[43]), "=f"(_tmem_load_14[44]), "=f"(_tmem_load_14[45]), "=f"(_tmem_load_14[46]), "=f"(_tmem_load_14[47]), "=f"(_tmem_load_14[48]), "=f"(_tmem_load_14[49]), "=f"(_tmem_load_14[50]), "=f"(_tmem_load_14[51]), "=f"(_tmem_load_14[52]), "=f"(_tmem_load_14[53]), "=f"(_tmem_load_14[54]), "=f"(_tmem_load_14[55]), "=f"(_tmem_load_14[56]), "=f"(_tmem_load_14[57]), "=f"(_tmem_load_14[58]), "=f"(_tmem_load_14[59]), "=f"(_tmem_load_14[60]), "=f"(_tmem_load_14[61]), "=f"(_tmem_load_14[62]), "=f"(_tmem_load_14[63])
                    : "r"(taddr + (unsigned int)row_base + 64 + 32));
                float _tmem_load_13_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_13_1[0]), "=f"(_tmem_load_13_1[1]), "=f"(_tmem_load_13_1[2]), "=f"(_tmem_load_13_1[3]), "=f"(_tmem_load_13_1[4]), "=f"(_tmem_load_13_1[5]), "=f"(_tmem_load_13_1[6]), "=f"(_tmem_load_13_1[7]), "=f"(_tmem_load_13_1[8]), "=f"(_tmem_load_13_1[9]), "=f"(_tmem_load_13_1[10]), "=f"(_tmem_load_13_1[11]), "=f"(_tmem_load_13_1[12]), "=f"(_tmem_load_13_1[13]), "=f"(_tmem_load_13_1[14]), "=f"(_tmem_load_13_1[15]), "=f"(_tmem_load_13_1[16]), "=f"(_tmem_load_13_1[17]), "=f"(_tmem_load_13_1[18]), "=f"(_tmem_load_13_1[19]), "=f"(_tmem_load_13_1[20]), "=f"(_tmem_load_13_1[21]), "=f"(_tmem_load_13_1[22]), "=f"(_tmem_load_13_1[23]), "=f"(_tmem_load_13_1[24]), "=f"(_tmem_load_13_1[25]), "=f"(_tmem_load_13_1[26]), "=f"(_tmem_load_13_1[27]), "=f"(_tmem_load_13_1[28]), "=f"(_tmem_load_13_1[29]), "=f"(_tmem_load_13_1[30]), "=f"(_tmem_load_13_1[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 192));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_13_1[32]), "=f"(_tmem_load_13_1[33]), "=f"(_tmem_load_13_1[34]), "=f"(_tmem_load_13_1[35]), "=f"(_tmem_load_13_1[36]), "=f"(_tmem_load_13_1[37]), "=f"(_tmem_load_13_1[38]), "=f"(_tmem_load_13_1[39]), "=f"(_tmem_load_13_1[40]), "=f"(_tmem_load_13_1[41]), "=f"(_tmem_load_13_1[42]), "=f"(_tmem_load_13_1[43]), "=f"(_tmem_load_13_1[44]), "=f"(_tmem_load_13_1[45]), "=f"(_tmem_load_13_1[46]), "=f"(_tmem_load_13_1[47]), "=f"(_tmem_load_13_1[48]), "=f"(_tmem_load_13_1[49]), "=f"(_tmem_load_13_1[50]), "=f"(_tmem_load_13_1[51]), "=f"(_tmem_load_13_1[52]), "=f"(_tmem_load_13_1[53]), "=f"(_tmem_load_13_1[54]), "=f"(_tmem_load_13_1[55]), "=f"(_tmem_load_13_1[56]), "=f"(_tmem_load_13_1[57]), "=f"(_tmem_load_13_1[58]), "=f"(_tmem_load_13_1[59]), "=f"(_tmem_load_13_1[60]), "=f"(_tmem_load_13_1[61]), "=f"(_tmem_load_13_1[62]), "=f"(_tmem_load_13_1[63])
                    : "r"(taddr + 128 + (unsigned int)row_base + 192 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_3 = 0; j_1_3 < 64; j_1_3++) {
                    _tmem_load_14[j_1_3] = _tmem_load_14[j_1_3] + _tmem_load_13_1[j_1_3];
                }
                unsigned int hi_1_1[32];
                unsigned int lo_2_1[32];
                #pragma unroll
                for (int j_21 = 0; j_21 < 32; j_21++) {
                    hi_1_1[j_21] = __as_u32(_tmem_load_14[j_21]) + half_tf32 & mask_tf32;
                    lo_2_1[j_21] = __as_u32(_tmem_load_14[j_21] - __uint_as_float(hi_1_1[j_21])) + half_tf32 & mask_tf32;
                }
                float xd_1_1[32];
                #pragma unroll
                for (int j_1_4 = 0; j_1_4 < 32; j_1_4++) {
                    xd_1_1[j_1_4] = _tmem_load_14[j_1_4] * decay;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 64), "r"(hi_1_1[0]), "r"(hi_1_1[1]), "r"(hi_1_1[2]), "r"(hi_1_1[3]), "r"(hi_1_1[4]), "r"(hi_1_1[5]), "r"(hi_1_1[6]), "r"(hi_1_1[7]), "r"(hi_1_1[8]), "r"(hi_1_1[9]), "r"(hi_1_1[10]), "r"(hi_1_1[11]), "r"(hi_1_1[12]), "r"(hi_1_1[13]), "r"(hi_1_1[14]), "r"(hi_1_1[15]), "r"(hi_1_1[16]), "r"(hi_1_1[17]), "r"(hi_1_1[18]), "r"(hi_1_1[19]), "r"(hi_1_1[20]), "r"(hi_1_1[21]), "r"(hi_1_1[22]), "r"(hi_1_1[23]), "r"(hi_1_1[24]), "r"(hi_1_1[25]), "r"(hi_1_1[26]), "r"(hi_1_1[27]), "r"(hi_1_1[28]), "r"(hi_1_1[29]), "r"(hi_1_1[30]), "r"(hi_1_1[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 192), "r"(lo_2_1[0]), "r"(lo_2_1[1]), "r"(lo_2_1[2]), "r"(lo_2_1[3]), "r"(lo_2_1[4]), "r"(lo_2_1[5]), "r"(lo_2_1[6]), "r"(lo_2_1[7]), "r"(lo_2_1[8]), "r"(lo_2_1[9]), "r"(lo_2_1[10]), "r"(lo_2_1[11]), "r"(lo_2_1[12]), "r"(lo_2_1[13]), "r"(lo_2_1[14]), "r"(lo_2_1[15]), "r"(lo_2_1[16]), "r"(lo_2_1[17]), "r"(lo_2_1[18]), "r"(lo_2_1[19]), "r"(lo_2_1[20]), "r"(lo_2_1[21]), "r"(lo_2_1[22]), "r"(lo_2_1[23]), "r"(lo_2_1[24]), "r"(lo_2_1[25]), "r"(lo_2_1[26]), "r"(lo_2_1[27]), "r"(lo_2_1[28]), "r"(lo_2_1[29]), "r"(lo_2_1[30]), "r"(lo_2_1[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base + 64, xd_1_1);
                unsigned int hi_2_1[32];
                unsigned int lo_3_1[32];
                #pragma unroll
                for (int j_22 = 0; j_22 < 32; j_22++) {
                    hi_2_1[j_22] = __as_u32(_tmem_load_14[32 + j_22]) + half_tf32 & mask_tf32;
                    lo_3_1[j_22] = __as_u32(_tmem_load_14[32 + j_22] - __uint_as_float(hi_2_1[j_22])) + half_tf32 & mask_tf32;
                }
                float xd_3_1[32];
                #pragma unroll
                for (int j_1_5 = 0; j_1_5 < 32; j_1_5++) {
                    xd_3_1[j_1_5] = _tmem_load_14[32 + j_1_5] * decay;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 96), "r"(hi_2_1[0]), "r"(hi_2_1[1]), "r"(hi_2_1[2]), "r"(hi_2_1[3]), "r"(hi_2_1[4]), "r"(hi_2_1[5]), "r"(hi_2_1[6]), "r"(hi_2_1[7]), "r"(hi_2_1[8]), "r"(hi_2_1[9]), "r"(hi_2_1[10]), "r"(hi_2_1[11]), "r"(hi_2_1[12]), "r"(hi_2_1[13]), "r"(hi_2_1[14]), "r"(hi_2_1[15]), "r"(hi_2_1[16]), "r"(hi_2_1[17]), "r"(hi_2_1[18]), "r"(hi_2_1[19]), "r"(hi_2_1[20]), "r"(hi_2_1[21]), "r"(hi_2_1[22]), "r"(hi_2_1[23]), "r"(hi_2_1[24]), "r"(hi_2_1[25]), "r"(hi_2_1[26]), "r"(hi_2_1[27]), "r"(hi_2_1[28]), "r"(hi_2_1[29]), "r"(hi_2_1[30]), "r"(hi_2_1[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 224), "r"(lo_3_1[0]), "r"(lo_3_1[1]), "r"(lo_3_1[2]), "r"(lo_3_1[3]), "r"(lo_3_1[4]), "r"(lo_3_1[5]), "r"(lo_3_1[6]), "r"(lo_3_1[7]), "r"(lo_3_1[8]), "r"(lo_3_1[9]), "r"(lo_3_1[10]), "r"(lo_3_1[11]), "r"(lo_3_1[12]), "r"(lo_3_1[13]), "r"(lo_3_1[14]), "r"(lo_3_1[15]), "r"(lo_3_1[16]), "r"(lo_3_1[17]), "r"(lo_3_1[18]), "r"(lo_3_1[19]), "r"(lo_3_1[20]), "r"(lo_3_1[21]), "r"(lo_3_1[22]), "r"(lo_3_1[23]), "r"(lo_3_1[24]), "r"(lo_3_1[25]), "r"(lo_3_1[26]), "r"(lo_3_1[27]), "r"(lo_3_1[28]), "r"(lo_3_1[29]), "r"(lo_3_1[30]), "r"(lo_3_1[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base + 96, xd_3_1);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(xs_ready_addr);
                long long u_base = ((long long)(row0 + c * 64) * (long long)num_heads + (long long)head) * 128 + (long long)lane_v;
                int u_stride = num_heads * 128;
                float uv[64];
                uv[0] = 0.0f;
                uv[1] = 0.0f;
                uv[2] = 0.0f;
                uv[3] = 0.0f;
                uv[4] = 0.0f;
                uv[5] = 0.0f;
                uv[6] = 0.0f;
                uv[7] = 0.0f;
                uv[8] = 0.0f;
                uv[9] = 0.0f;
                uv[10] = 0.0f;
                uv[11] = 0.0f;
                uv[12] = 0.0f;
                uv[13] = 0.0f;
                uv[14] = 0.0f;
                uv[15] = 0.0f;
                uv[16] = 0.0f;
                uv[17] = 0.0f;
                uv[18] = 0.0f;
                uv[19] = 0.0f;
                uv[20] = 0.0f;
                uv[21] = 0.0f;
                uv[22] = 0.0f;
                uv[23] = 0.0f;
                uv[24] = 0.0f;
                uv[25] = 0.0f;
                uv[26] = 0.0f;
                uv[27] = 0.0f;
                uv[28] = 0.0f;
                uv[29] = 0.0f;
                uv[30] = 0.0f;
                uv[31] = 0.0f;
                uv[32] = 0.0f;
                uv[33] = 0.0f;
                uv[34] = 0.0f;
                uv[35] = 0.0f;
                uv[36] = 0.0f;
                uv[37] = 0.0f;
                uv[38] = 0.0f;
                uv[39] = 0.0f;
                uv[40] = 0.0f;
                uv[41] = 0.0f;
                uv[42] = 0.0f;
                uv[43] = 0.0f;
                uv[44] = 0.0f;
                uv[45] = 0.0f;
                uv[46] = 0.0f;
                uv[47] = 0.0f;
                uv[48] = 0.0f;
                uv[49] = 0.0f;
                uv[50] = 0.0f;
                uv[51] = 0.0f;
                uv[52] = 0.0f;
                uv[53] = 0.0f;
                uv[54] = 0.0f;
                uv[55] = 0.0f;
                uv[56] = 0.0f;
                uv[57] = 0.0f;
                uv[58] = 0.0f;
                uv[59] = 0.0f;
                uv[60] = 0.0f;
                uv[61] = 0.0f;
                uv[62] = 0.0f;
                uv[63] = 0.0f;
                if (use_u == 1) {
                    #pragma unroll
                    for (int t = 0; t < 64; t++) {
                        {
                            __nv_bfloat16 _bf16_1 = *reinterpret_cast<const __nv_bfloat16*>(u + u_base + (long long)(t * u_stride));
                            uv[t] = __bfloat162float(_bf16_1);
                        }
                    }
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_4[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                    : "r"(taddr + 384 + (unsigned int)row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_4[32]), "=f"(_tmem_load_4[33]), "=f"(_tmem_load_4[34]), "=f"(_tmem_load_4[35]), "=f"(_tmem_load_4[36]), "=f"(_tmem_load_4[37]), "=f"(_tmem_load_4[38]), "=f"(_tmem_load_4[39]), "=f"(_tmem_load_4[40]), "=f"(_tmem_load_4[41]), "=f"(_tmem_load_4[42]), "=f"(_tmem_load_4[43]), "=f"(_tmem_load_4[44]), "=f"(_tmem_load_4[45]), "=f"(_tmem_load_4[46]), "=f"(_tmem_load_4[47]), "=f"(_tmem_load_4[48]), "=f"(_tmem_load_4[49]), "=f"(_tmem_load_4[50]), "=f"(_tmem_load_4[51]), "=f"(_tmem_load_4[52]), "=f"(_tmem_load_4[53]), "=f"(_tmem_load_4[54]), "=f"(_tmem_load_4[55]), "=f"(_tmem_load_4[56]), "=f"(_tmem_load_4[57]), "=f"(_tmem_load_4[58]), "=f"(_tmem_load_4[59]), "=f"(_tmem_load_4[60]), "=f"(_tmem_load_4[61]), "=f"(_tmem_load_4[62]), "=f"(_tmem_load_4[63])
                    : "r"(taddr + 384 + (unsigned int)row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float bvv[64];
                #pragma unroll
                for (int j_23 = 0; j_23 < 64; j_23++) {
                    bvv[j_23] = uv[j_23] - _tmem_load_4[j_23];
                }
                float rs[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[0])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(0) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[4])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(4) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[8])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(8) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[12])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(12) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[16])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(16) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[20])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(20) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[24])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(24) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[28])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(28) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 112)));
                float bs[32];
                #pragma unroll
                for (int j_24 = 0; j_24 < 32; j_24++) {
                    bs[j_24] = bvv[j_24] * rs[j_24];
                }
                unsigned int bhi[32];
                unsigned int blo[32];
                #pragma unroll
                for (int j_25 = 0; j_25 < 32; j_25++) {
                    bhi[j_25] = __as_u32(bs[j_25]) + half_tf32 & mask_tf32;
                    blo[j_25] = __as_u32(bs[j_25] - __uint_as_float(bhi[j_25])) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 384 + (unsigned int)row_base), "r"(bhi[0]), "r"(bhi[1]), "r"(bhi[2]), "r"(bhi[3]), "r"(bhi[4]), "r"(bhi[5]), "r"(bhi[6]), "r"(bhi[7]), "r"(bhi[8]), "r"(bhi[9]), "r"(bhi[10]), "r"(bhi[11]), "r"(bhi[12]), "r"(bhi[13]), "r"(bhi[14]), "r"(bhi[15]), "r"(bhi[16]), "r"(bhi[17]), "r"(bhi[18]), "r"(bhi[19]), "r"(bhi[20]), "r"(bhi[21]), "r"(bhi[22]), "r"(bhi[23]), "r"(bhi[24]), "r"(bhi[25]), "r"(bhi[26]), "r"(bhi[27]), "r"(bhi[28]), "r"(bhi[29]), "r"(bhi[30]), "r"(bhi[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 448 + (unsigned int)row_base), "r"(blo[0]), "r"(blo[1]), "r"(blo[2]), "r"(blo[3]), "r"(blo[4]), "r"(blo[5]), "r"(blo[6]), "r"(blo[7]), "r"(blo[8]), "r"(blo[9]), "r"(blo[10]), "r"(blo[11]), "r"(blo[12]), "r"(blo[13]), "r"(blo[14]), "r"(blo[15]), "r"(blo[16]), "r"(blo[17]), "r"(blo[18]), "r"(blo[19]), "r"(blo[20]), "r"(blo[21]), "r"(blo[22]), "r"(blo[23]), "r"(blo[24]), "r"(blo[25]), "r"(blo[26]), "r"(blo[27]), "r"(blo[28]), "r"(blo[29]), "r"(blo[30]), "r"(blo[31]));
                float rs_9[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(0) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(4) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(8) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(12) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[16])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(16) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[20])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(20) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[24])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(24) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_9[28])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_9[(28) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + 128 + 112)));
                float bs_10[32];
                #pragma unroll
                for (int j_26 = 0; j_26 < 32; j_26++) {
                    bs_10[j_26] = bvv[32 + j_26] * rs_9[j_26];
                }
                unsigned int bhi_11[32];
                unsigned int blo_12[32];
                #pragma unroll
                for (int j_27 = 0; j_27 < 32; j_27++) {
                    bhi_11[j_27] = __as_u32(bs_10[j_27]) + half_tf32 & mask_tf32;
                    blo_12[j_27] = __as_u32(bs_10[j_27] - __uint_as_float(bhi_11[j_27])) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 384 + (unsigned int)row_base + 32), "r"(bhi_11[0]), "r"(bhi_11[1]), "r"(bhi_11[2]), "r"(bhi_11[3]), "r"(bhi_11[4]), "r"(bhi_11[5]), "r"(bhi_11[6]), "r"(bhi_11[7]), "r"(bhi_11[8]), "r"(bhi_11[9]), "r"(bhi_11[10]), "r"(bhi_11[11]), "r"(bhi_11[12]), "r"(bhi_11[13]), "r"(bhi_11[14]), "r"(bhi_11[15]), "r"(bhi_11[16]), "r"(bhi_11[17]), "r"(bhi_11[18]), "r"(bhi_11[19]), "r"(bhi_11[20]), "r"(bhi_11[21]), "r"(bhi_11[22]), "r"(bhi_11[23]), "r"(bhi_11[24]), "r"(bhi_11[25]), "r"(bhi_11[26]), "r"(bhi_11[27]), "r"(bhi_11[28]), "r"(bhi_11[29]), "r"(bhi_11[30]), "r"(bhi_11[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 448 + (unsigned int)row_base + 32), "r"(blo_12[0]), "r"(blo_12[1]), "r"(blo_12[2]), "r"(blo_12[3]), "r"(blo_12[4]), "r"(blo_12[5]), "r"(blo_12[6]), "r"(blo_12[7]), "r"(blo_12[8]), "r"(blo_12[9]), "r"(blo_12[10]), "r"(blo_12[11]), "r"(blo_12[12]), "r"(blo_12[13]), "r"(blo_12[14]), "r"(blo_12[15]), "r"(blo_12[16]), "r"(blo_12[17]), "r"(blo_12[18]), "r"(blo_12[19]), "r"(blo_12[20]), "r"(blo_12[21]), "r"(blo_12[22]), "r"(blo_12[23]), "r"(blo_12[24]), "r"(blo_12[25]), "r"(blo_12[26]), "r"(blo_12[27]), "r"(blo_12[28]), "r"(blo_12[29]), "r"(blo_12[30]), "r"(blo_12[31]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    mbarrier_wait(f32_empty_addr, _phase_f32_empty);
                    mbarrier_wait(tiles_full_addr + (ls) * 8, _phase_tiles_full);
                    unsigned int words_1[4];
                    float wide_1[8];
                    unsigned int wide_hi_1[8];
                    unsigned int wide_lo_1[8];
                    int col_7_8 = w_col;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_8 / 64 * 8192 + w_row * 128 + col_7_8 % 64 * 2 ^ (col_7_8 / 64 * 8192 + w_row * 128 + col_7_8 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_28 = 0; j_28 < 8; j_28++) {
                        wide_hi_1[j_28] = __as_u32(wide_1[j_28]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_28] = __as_u32(wide_1[j_28] - __uint_as_float(wide_hi_1[j_28])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_8 / 32 * 8192 + w_row * 128 + col_7_8 % 32 * 4 ^ (col_7_8 / 32 * 8192 + w_row * 128 + col_7_8 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_8 / 32 * 8192 + w_row * 128 + col_7_8 % 32 * 4 ^ (col_7_8 / 32 * 8192 + w_row * 128 + col_7_8 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_8 + 4) / 32 * 8192 + w_row * 128 + (col_7_8 + 4) % 32 * 4 ^ ((col_7_8 + 4) / 32 * 8192 + w_row * 128 + (col_7_8 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_8 + 4) / 32 * 8192 + w_row * 128 + (col_7_8 + 4) % 32 * 4 ^ ((col_7_8 + 4) / 32 * 8192 + w_row * 128 + (col_7_8 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_9 = w_col + 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_9 / 64 * 8192 + w_row * 128 + col_7_9 % 64 * 2 ^ (col_7_9 / 64 * 8192 + w_row * 128 + col_7_9 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_29 = 0; j_29 < 8; j_29++) {
                        wide_hi_1[j_29] = __as_u32(wide_1[j_29]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_29] = __as_u32(wide_1[j_29] - __uint_as_float(wide_hi_1[j_29])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_9 / 32 * 8192 + w_row * 128 + col_7_9 % 32 * 4 ^ (col_7_9 / 32 * 8192 + w_row * 128 + col_7_9 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_9 / 32 * 8192 + w_row * 128 + col_7_9 % 32 * 4 ^ (col_7_9 / 32 * 8192 + w_row * 128 + col_7_9 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_9 + 4) / 32 * 8192 + w_row * 128 + (col_7_9 + 4) % 32 * 4 ^ ((col_7_9 + 4) / 32 * 8192 + w_row * 128 + (col_7_9 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_9 + 4) / 32 * 8192 + w_row * 128 + (col_7_9 + 4) % 32 * 4 ^ ((col_7_9 + 4) / 32 * 8192 + w_row * 128 + (col_7_9 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_10 = w_col + 16;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_10 / 64 * 8192 + w_row * 128 + col_7_10 % 64 * 2 ^ (col_7_10 / 64 * 8192 + w_row * 128 + col_7_10 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_30 = 0; j_30 < 8; j_30++) {
                        wide_hi_1[j_30] = __as_u32(wide_1[j_30]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_30] = __as_u32(wide_1[j_30] - __uint_as_float(wide_hi_1[j_30])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_10 / 32 * 8192 + w_row * 128 + col_7_10 % 32 * 4 ^ (col_7_10 / 32 * 8192 + w_row * 128 + col_7_10 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_10 / 32 * 8192 + w_row * 128 + col_7_10 % 32 * 4 ^ (col_7_10 / 32 * 8192 + w_row * 128 + col_7_10 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_10 + 4) / 32 * 8192 + w_row * 128 + (col_7_10 + 4) % 32 * 4 ^ ((col_7_10 + 4) / 32 * 8192 + w_row * 128 + (col_7_10 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_10 + 4) / 32 * 8192 + w_row * 128 + (col_7_10 + 4) % 32 * 4 ^ ((col_7_10 + 4) / 32 * 8192 + w_row * 128 + (col_7_10 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_11 = w_col + 24;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_11 / 64 * 8192 + w_row * 128 + col_7_11 % 64 * 2 ^ (col_7_11 / 64 * 8192 + w_row * 128 + col_7_11 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_31 = 0; j_31 < 8; j_31++) {
                        wide_hi_1[j_31] = __as_u32(wide_1[j_31]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_31] = __as_u32(wide_1[j_31] - __uint_as_float(wide_hi_1[j_31])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_11 / 32 * 8192 + w_row * 128 + col_7_11 % 32 * 4 ^ (col_7_11 / 32 * 8192 + w_row * 128 + col_7_11 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_11 / 32 * 8192 + w_row * 128 + col_7_11 % 32 * 4 ^ (col_7_11 / 32 * 8192 + w_row * 128 + col_7_11 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_11 + 4) / 32 * 8192 + w_row * 128 + (col_7_11 + 4) % 32 * 4 ^ ((col_7_11 + 4) / 32 * 8192 + w_row * 128 + (col_7_11 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_11 + 4) / 32 * 8192 + w_row * 128 + (col_7_11 + 4) % 32 * 4 ^ ((col_7_11 + 4) / 32 * 8192 + w_row * 128 + (col_7_11 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_12 = w_col + 32;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_12 / 64 * 8192 + w_row * 128 + col_7_12 % 64 * 2 ^ (col_7_12 / 64 * 8192 + w_row * 128 + col_7_12 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_32 = 0; j_32 < 8; j_32++) {
                        wide_hi_1[j_32] = __as_u32(wide_1[j_32]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_32] = __as_u32(wide_1[j_32] - __uint_as_float(wide_hi_1[j_32])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_12 / 32 * 8192 + w_row * 128 + col_7_12 % 32 * 4 ^ (col_7_12 / 32 * 8192 + w_row * 128 + col_7_12 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_12 / 32 * 8192 + w_row * 128 + col_7_12 % 32 * 4 ^ (col_7_12 / 32 * 8192 + w_row * 128 + col_7_12 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_12 + 4) / 32 * 8192 + w_row * 128 + (col_7_12 + 4) % 32 * 4 ^ ((col_7_12 + 4) / 32 * 8192 + w_row * 128 + (col_7_12 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_12 + 4) / 32 * 8192 + w_row * 128 + (col_7_12 + 4) % 32 * 4 ^ ((col_7_12 + 4) / 32 * 8192 + w_row * 128 + (col_7_12 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_13 = w_col + 40;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_13 / 64 * 8192 + w_row * 128 + col_7_13 % 64 * 2 ^ (col_7_13 / 64 * 8192 + w_row * 128 + col_7_13 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_33 = 0; j_33 < 8; j_33++) {
                        wide_hi_1[j_33] = __as_u32(wide_1[j_33]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_33] = __as_u32(wide_1[j_33] - __uint_as_float(wide_hi_1[j_33])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_13 / 32 * 8192 + w_row * 128 + col_7_13 % 32 * 4 ^ (col_7_13 / 32 * 8192 + w_row * 128 + col_7_13 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_13 / 32 * 8192 + w_row * 128 + col_7_13 % 32 * 4 ^ (col_7_13 / 32 * 8192 + w_row * 128 + col_7_13 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_13 + 4) / 32 * 8192 + w_row * 128 + (col_7_13 + 4) % 32 * 4 ^ ((col_7_13 + 4) / 32 * 8192 + w_row * 128 + (col_7_13 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_13 + 4) / 32 * 8192 + w_row * 128 + (col_7_13 + 4) % 32 * 4 ^ ((col_7_13 + 4) / 32 * 8192 + w_row * 128 + (col_7_13 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_14 = w_col + 48;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_14 / 64 * 8192 + w_row * 128 + col_7_14 % 64 * 2 ^ (col_7_14 / 64 * 8192 + w_row * 128 + col_7_14 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_34 = 0; j_34 < 8; j_34++) {
                        wide_hi_1[j_34] = __as_u32(wide_1[j_34]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_34] = __as_u32(wide_1[j_34] - __uint_as_float(wide_hi_1[j_34])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_14 / 32 * 8192 + w_row * 128 + col_7_14 % 32 * 4 ^ (col_7_14 / 32 * 8192 + w_row * 128 + col_7_14 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_14 / 32 * 8192 + w_row * 128 + col_7_14 % 32 * 4 ^ (col_7_14 / 32 * 8192 + w_row * 128 + col_7_14 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_14 + 4) / 32 * 8192 + w_row * 128 + (col_7_14 + 4) % 32 * 4 ^ ((col_7_14 + 4) / 32 * 8192 + w_row * 128 + (col_7_14 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_14 + 4) / 32 * 8192 + w_row * 128 + (col_7_14 + 4) % 32 * 4 ^ ((col_7_14 + 4) / 32 * 8192 + w_row * 128 + (col_7_14 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int col_7_15 = w_col + 56;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7_15 / 64 * 8192 + w_row * 128 + col_7_15 % 64 * 2 ^ (col_7_15 / 64 * 8192 + w_row * 128 + col_7_15 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_35 = 0; j_35 < 8; j_35++) {
                        wide_hi_1[j_35] = __as_u32(wide_1[j_35]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_35] = __as_u32(wide_1[j_35] - __uint_as_float(wide_hi_1[j_35])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)(col_7_15 / 32 * 8192 + w_row * 128 + col_7_15 % 32 * 4 ^ (col_7_15 / 32 * 8192 + w_row * 128 + col_7_15 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]), "r"(wide_hi_1[1]), "r"(wide_hi_1[2]), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)(col_7_15 / 32 * 8192 + w_row * 128 + col_7_15 % 32 * 4 ^ (col_7_15 / 32 * 8192 + w_row * 128 + col_7_15 % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]), "r"(wide_lo_1[1]), "r"(wide_lo_1[2]), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + 0 * 32768 + (unsigned int)((col_7_15 + 4) / 32 * 8192 + w_row * 128 + (col_7_15 + 4) % 32 * 4 ^ ((col_7_15 + 4) / 32 * 8192 + w_row * 128 + (col_7_15 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]), "r"(wide_hi_1[5]), "r"(wide_hi_1[6]), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((wf_lo_addr + (unsigned int)((col_7_15 + 4) / 32 * 8192 + w_row * 128 + (col_7_15 + 4) % 32 * 4 ^ ((col_7_15 + 4) / 32 * 8192 + w_row * 128 + (col_7_15 + 4) % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]), "r"(wide_lo_1[5]), "r"(wide_lo_1[6]), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_8 = k_row0;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_8 / 64 * 8192 + t_col * 128 + krow_1_8 % 64 * 2 ^ (krow_1_8 / 64 * 8192 + t_col * 128 + krow_1_8 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_36 = 0; j_36 < 8; j_36++) {
                        wide_hi_1[j_36] = __as_u32(wide_1[j_36]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_36] = __as_u32(wide_1[j_36] - __uint_as_float(wide_hi_1[j_36])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_8 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_8 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_8 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_8 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_8 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_8 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_9 = k_row0 + 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_9 / 64 * 8192 + t_col * 128 + krow_1_9 % 64 * 2 ^ (krow_1_9 / 64 * 8192 + t_col * 128 + krow_1_9 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_37 = 0; j_37 < 8; j_37++) {
                        wide_hi_1[j_37] = __as_u32(wide_1[j_37]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_37] = __as_u32(wide_1[j_37] - __uint_as_float(wide_hi_1[j_37])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_9 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_9 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_9 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_9 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_9 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_9 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_10 = k_row0 + 16;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_10 / 64 * 8192 + t_col * 128 + krow_1_10 % 64 * 2 ^ (krow_1_10 / 64 * 8192 + t_col * 128 + krow_1_10 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_38 = 0; j_38 < 8; j_38++) {
                        wide_hi_1[j_38] = __as_u32(wide_1[j_38]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_38] = __as_u32(wide_1[j_38] - __uint_as_float(wide_hi_1[j_38])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_10 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_10 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_10 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_10 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_10 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_10 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_11 = k_row0 + 24;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_11 / 64 * 8192 + t_col * 128 + krow_1_11 % 64 * 2 ^ (krow_1_11 / 64 * 8192 + t_col * 128 + krow_1_11 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_39 = 0; j_39 < 8; j_39++) {
                        wide_hi_1[j_39] = __as_u32(wide_1[j_39]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_39] = __as_u32(wide_1[j_39] - __uint_as_float(wide_hi_1[j_39])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_11 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_11 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_11 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_11 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_11 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_11 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_12 = k_row0 + 32;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_12 / 64 * 8192 + t_col * 128 + krow_1_12 % 64 * 2 ^ (krow_1_12 / 64 * 8192 + t_col * 128 + krow_1_12 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_40 = 0; j_40 < 8; j_40++) {
                        wide_hi_1[j_40] = __as_u32(wide_1[j_40]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_40] = __as_u32(wide_1[j_40] - __uint_as_float(wide_hi_1[j_40])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_12 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_12 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_12 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_12 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_12 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_12 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_13 = k_row0 + 40;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_13 / 64 * 8192 + t_col * 128 + krow_1_13 % 64 * 2 ^ (krow_1_13 / 64 * 8192 + t_col * 128 + krow_1_13 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_41 = 0; j_41 < 8; j_41++) {
                        wide_hi_1[j_41] = __as_u32(wide_1[j_41]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_41] = __as_u32(wide_1[j_41] - __uint_as_float(wide_hi_1[j_41])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_13 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_13 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_13 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_13 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_13 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_13 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_14 = k_row0 + 48;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_14 / 64 * 8192 + t_col * 128 + krow_1_14 % 64 * 2 ^ (krow_1_14 / 64 * 8192 + t_col * 128 + krow_1_14 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_42 = 0; j_42 < 8; j_42++) {
                        wide_hi_1[j_42] = __as_u32(wide_1[j_42]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_42] = __as_u32(wide_1[j_42] - __uint_as_float(wide_hi_1[j_42])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_14 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_14 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_14 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_14 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_14 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_14 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    int krow_1_15 = k_row0 + 56;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_15 / 64 * 8192 + t_col * 128 + krow_1_15 % 64 * 2 ^ (krow_1_15 / 64 * 8192 + t_col * 128 + krow_1_15 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    #pragma unroll
                    for (int j_43 = 0; j_43 < 8; j_43++) {
                        wide_hi_1[j_43] = __as_u32(wide_1[j_43]) + half_tf32 & mask_tf32;
                        wide_lo_1[j_43] = __as_u32(wide_1[j_43] - __uint_as_float(wide_hi_1[j_43])) + half_tf32 & mask_tf32;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_15 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_15 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + krow_1_15 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_15 * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[0]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[1]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[2]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[3]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[4]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[5]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[6]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_addr + 0 * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_hi_1[7]) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((kf_lo_addr + (unsigned int)(t_col / 32 * 16384 + (krow_1_15 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_15 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4))), "r"(wide_lo_1[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(f32_full_addr);
                    mbarrier_arrive(tiles_empty_addr + (ls) * 8);
                    ls += 1;
                    if (ls == 2) { ls = 0; _phase_tiles_full ^= 1; }
                    _phase_f32_empty ^= 1;
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            int slot = out_index[seg];
            if (slot >= 0) {
                long long tile = (long long)((head / out_hg * out_n + slot) * out_hg + head % out_hg) * (long long)tile_stride * 16384;
                float _tmem_load_5[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                    : "r"(taddr + (unsigned int)row_base));
                float _tmem_load_6[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_44 = 0; j_44 < 32; j_44++) {
                    _tmem_load_5[j_44] = _tmem_load_5[j_44] + _tmem_load_6[j_44];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_45 = 0; j_45 < 32; j_45++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)(j_45 * 128 + lane_v))) = _tmem_load_5[j_45];
                    }
                } else {
                    long long row_at = tile + (long long)(lane_v * 128);
                    {
                        float4 _v4 = make_float4(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1], _tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[4 + 0], _tmem_load_5[4 + 1], _tmem_load_5[4 + 2], _tmem_load_5[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[8 + 0], _tmem_load_5[8 + 1], _tmem_load_5[8 + 2], _tmem_load_5[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[12 + 0], _tmem_load_5[12 + 1], _tmem_load_5[12 + 2], _tmem_load_5[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[16 + 0], _tmem_load_5[16 + 1], _tmem_load_5[16 + 2], _tmem_load_5[16 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[20 + 0], _tmem_load_5[20 + 1], _tmem_load_5[20 + 2], _tmem_load_5[20 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[24 + 0], _tmem_load_5[24 + 1], _tmem_load_5[24 + 2], _tmem_load_5[24 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_5[28 + 0], _tmem_load_5[28 + 1], _tmem_load_5[28 + 2], _tmem_load_5[28 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 28) = _v4;
                    }
                }
                float _tmem_load_7[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                    : "r"(taddr + (unsigned int)row_base + 32));
                float _tmem_load_8[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 160));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_46 = 0; j_46 < 32; j_46++) {
                    _tmem_load_7[j_46] = _tmem_load_7[j_46] + _tmem_load_8[j_46];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_47 = 0; j_47 < 32; j_47++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((32 + j_47) * 128 + lane_v))) = _tmem_load_7[j_47];
                    }
                } else {
                    long long row_at_1 = tile + (long long)(lane_v * 128 + 32);
                    {
                        float4 _v4 = make_float4(_tmem_load_7[0 + 0], _tmem_load_7[0 + 1], _tmem_load_7[0 + 2], _tmem_load_7[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[4 + 0], _tmem_load_7[4 + 1], _tmem_load_7[4 + 2], _tmem_load_7[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[8 + 0], _tmem_load_7[8 + 1], _tmem_load_7[8 + 2], _tmem_load_7[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[12 + 0], _tmem_load_7[12 + 1], _tmem_load_7[12 + 2], _tmem_load_7[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[16 + 0], _tmem_load_7[16 + 1], _tmem_load_7[16 + 2], _tmem_load_7[16 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[20 + 0], _tmem_load_7[20 + 1], _tmem_load_7[20 + 2], _tmem_load_7[20 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[24 + 0], _tmem_load_7[24 + 1], _tmem_load_7[24 + 2], _tmem_load_7[24 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[28 + 0], _tmem_load_7[28 + 1], _tmem_load_7[28 + 2], _tmem_load_7[28 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 28) = _v4;
                    }
                }
                float _tmem_load_9[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                    : "r"(taddr + (unsigned int)row_base + 64));
                float _tmem_load_10[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_10[0]), "=f"(_tmem_load_10[1]), "=f"(_tmem_load_10[2]), "=f"(_tmem_load_10[3]), "=f"(_tmem_load_10[4]), "=f"(_tmem_load_10[5]), "=f"(_tmem_load_10[6]), "=f"(_tmem_load_10[7]), "=f"(_tmem_load_10[8]), "=f"(_tmem_load_10[9]), "=f"(_tmem_load_10[10]), "=f"(_tmem_load_10[11]), "=f"(_tmem_load_10[12]), "=f"(_tmem_load_10[13]), "=f"(_tmem_load_10[14]), "=f"(_tmem_load_10[15]), "=f"(_tmem_load_10[16]), "=f"(_tmem_load_10[17]), "=f"(_tmem_load_10[18]), "=f"(_tmem_load_10[19]), "=f"(_tmem_load_10[20]), "=f"(_tmem_load_10[21]), "=f"(_tmem_load_10[22]), "=f"(_tmem_load_10[23]), "=f"(_tmem_load_10[24]), "=f"(_tmem_load_10[25]), "=f"(_tmem_load_10[26]), "=f"(_tmem_load_10[27]), "=f"(_tmem_load_10[28]), "=f"(_tmem_load_10[29]), "=f"(_tmem_load_10[30]), "=f"(_tmem_load_10[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 192));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_48 = 0; j_48 < 32; j_48++) {
                    _tmem_load_9[j_48] = _tmem_load_9[j_48] + _tmem_load_10[j_48];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_49 = 0; j_49 < 32; j_49++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((64 + j_49) * 128 + lane_v))) = _tmem_load_9[j_49];
                    }
                } else {
                    long long row_at_2 = tile + (long long)(lane_v * 128 + 64);
                    {
                        float4 _v4 = make_float4(_tmem_load_9[0 + 0], _tmem_load_9[0 + 1], _tmem_load_9[0 + 2], _tmem_load_9[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[4 + 0], _tmem_load_9[4 + 1], _tmem_load_9[4 + 2], _tmem_load_9[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[8 + 0], _tmem_load_9[8 + 1], _tmem_load_9[8 + 2], _tmem_load_9[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[12 + 0], _tmem_load_9[12 + 1], _tmem_load_9[12 + 2], _tmem_load_9[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[16 + 0], _tmem_load_9[16 + 1], _tmem_load_9[16 + 2], _tmem_load_9[16 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[20 + 0], _tmem_load_9[20 + 1], _tmem_load_9[20 + 2], _tmem_load_9[20 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[24 + 0], _tmem_load_9[24 + 1], _tmem_load_9[24 + 2], _tmem_load_9[24 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[28 + 0], _tmem_load_9[28 + 1], _tmem_load_9[28 + 2], _tmem_load_9[28 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 28) = _v4;
                    }
                }
                float _tmem_load_11[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_11[0]), "=f"(_tmem_load_11[1]), "=f"(_tmem_load_11[2]), "=f"(_tmem_load_11[3]), "=f"(_tmem_load_11[4]), "=f"(_tmem_load_11[5]), "=f"(_tmem_load_11[6]), "=f"(_tmem_load_11[7]), "=f"(_tmem_load_11[8]), "=f"(_tmem_load_11[9]), "=f"(_tmem_load_11[10]), "=f"(_tmem_load_11[11]), "=f"(_tmem_load_11[12]), "=f"(_tmem_load_11[13]), "=f"(_tmem_load_11[14]), "=f"(_tmem_load_11[15]), "=f"(_tmem_load_11[16]), "=f"(_tmem_load_11[17]), "=f"(_tmem_load_11[18]), "=f"(_tmem_load_11[19]), "=f"(_tmem_load_11[20]), "=f"(_tmem_load_11[21]), "=f"(_tmem_load_11[22]), "=f"(_tmem_load_11[23]), "=f"(_tmem_load_11[24]), "=f"(_tmem_load_11[25]), "=f"(_tmem_load_11[26]), "=f"(_tmem_load_11[27]), "=f"(_tmem_load_11[28]), "=f"(_tmem_load_11[29]), "=f"(_tmem_load_11[30]), "=f"(_tmem_load_11[31])
                    : "r"(taddr + (unsigned int)row_base + 96));
                float _tmem_load_12[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_12[0]), "=f"(_tmem_load_12[1]), "=f"(_tmem_load_12[2]), "=f"(_tmem_load_12[3]), "=f"(_tmem_load_12[4]), "=f"(_tmem_load_12[5]), "=f"(_tmem_load_12[6]), "=f"(_tmem_load_12[7]), "=f"(_tmem_load_12[8]), "=f"(_tmem_load_12[9]), "=f"(_tmem_load_12[10]), "=f"(_tmem_load_12[11]), "=f"(_tmem_load_12[12]), "=f"(_tmem_load_12[13]), "=f"(_tmem_load_12[14]), "=f"(_tmem_load_12[15]), "=f"(_tmem_load_12[16]), "=f"(_tmem_load_12[17]), "=f"(_tmem_load_12[18]), "=f"(_tmem_load_12[19]), "=f"(_tmem_load_12[20]), "=f"(_tmem_load_12[21]), "=f"(_tmem_load_12[22]), "=f"(_tmem_load_12[23]), "=f"(_tmem_load_12[24]), "=f"(_tmem_load_12[25]), "=f"(_tmem_load_12[26]), "=f"(_tmem_load_12[27]), "=f"(_tmem_load_12[28]), "=f"(_tmem_load_12[29]), "=f"(_tmem_load_12[30]), "=f"(_tmem_load_12[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 224));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_50 = 0; j_50 < 32; j_50++) {
                    _tmem_load_11[j_50] = _tmem_load_11[j_50] + _tmem_load_12[j_50];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_51 = 0; j_51 < 32; j_51++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((96 + j_51) * 128 + lane_v))) = _tmem_load_11[j_51];
                    }
                } else {
                    long long row_at_3 = tile + (long long)(lane_v * 128 + 96);
                    {
                        float4 _v4 = make_float4(_tmem_load_11[0 + 0], _tmem_load_11[0 + 1], _tmem_load_11[0 + 2], _tmem_load_11[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[4 + 0], _tmem_load_11[4 + 1], _tmem_load_11[4 + 2], _tmem_load_11[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[8 + 0], _tmem_load_11[8 + 1], _tmem_load_11[8 + 2], _tmem_load_11[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[12 + 0], _tmem_load_11[12 + 1], _tmem_load_11[12 + 2], _tmem_load_11[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[16 + 0], _tmem_load_11[16 + 1], _tmem_load_11[16 + 2], _tmem_load_11[16 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[20 + 0], _tmem_load_11[20 + 1], _tmem_load_11[20 + 2], _tmem_load_11[20 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[24 + 0], _tmem_load_11[24 + 1], _tmem_load_11[24 + 2], _tmem_load_11[24 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[28 + 0], _tmem_load_11[28 + 1], _tmem_load_11[28 + 2], _tmem_load_11[28 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 28) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int head_1 = blockIdx.x % num_heads;
            int seg_1 = walk_seg[blockIdx.x / num_heads];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int k_head = head_1 / (num_heads / num_qk_heads);
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow = row0_1 + c_1 * 64;
                    int ktok = chunk_bos[c0_1 + c_1];
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)half) * 8192, (&w_tma), half * 64, head_1, trow, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(k_panel_addr + (stage * 2 + (unsigned int)half) * 8192, (&k_tma), half * 64, k_head, ktok, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int head_2 = blockIdx.x % num_heads;
            int seg_2 = walk_seg[blockIdx.x / num_heads];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int _phase_f32_full = 0;
            unsigned int _phase_xs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(f32_full_addr, _phase_f32_full);
                    mbarrier_wait(xs_ready_addr, _phase_xs_ready_0);
                    _phase_xs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((wf_addr) >> 4) & 0x7FFF) + (0) * 2048;
#else
                    int _smem_desc_lo_0 = (((wf_addr) >> 4) & 0x3FFF) + (0) * 2048;
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
                    "mov.b32 id, 135268624;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_b_hi), "r"(_smem_desc_lo_0), "r"(tmem_x_split + 128), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_1 = (((wf_lo_addr) >> 4) & 0x7FFF) + (0) * 2048;
#else
                    int _smem_desc_lo_1 = (((wf_lo_addr) >> 4) & 0x3FFF) + (0) * 2048;
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
                    "mov.b32 id, 135268624;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_b_hi), "r"(_smem_desc_lo_1), "r"(tmem_x_split), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((wf_addr) >> 4) & 0x7FFF) + (0) * 2048;
#else
                    int _smem_desc_lo_2 = (((wf_addr) >> 4) & 0x3FFF) + (0) * 2048;
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
                    "mov.b32 id, 135268624;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_b_hi), "r"(_smem_desc_lo_2), "r"(tmem_x_split), "r"(1) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((kf_addr) >> 4) & 0x7FFF) + (0) * 2048;
#else
                    int _smem_desc_lo_3 = (((kf_addr) >> 4) & 0x3FFF) + (0) * 2048;
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
                    "mov.b32 id, 136317200;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_x_split + (128))), "r"(_smem_desc_lo_3), "r"(tmem_b_lo), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((kf_lo_addr) >> 4) & 0x7FFF) + (0) * 2048;
#else
                    int _smem_desc_lo_4 = (((kf_lo_addr) >> 4) & 0x3FFF) + (0) * 2048;
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
                    "mov.b32 id, 136317200;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_x_split + (128))), "r"(_smem_desc_lo_4), "r"(tmem_b_hi), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_5 = (((kf_addr) >> 4) & 0x7FFF) + (0) * 2048;
#else
                    int _smem_desc_lo_5 = (((kf_addr) >> 4) & 0x3FFF) + (0) * 2048;
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
                    "mov.b32 id, 136317200;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_x_split + (128))), "r"(_smem_desc_lo_5), "r"(tmem_b_hi), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(f32_empty_addr);
                    _phase_f32_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
