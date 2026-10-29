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
#define NUM_FRING_STAGES 2
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
#define SMEM_KF_OFF 132096
#define SMEM_KF_STAGE_BYTES 32768
#define SMEM_KF_STRIDE 32768
#define SMEM_RSB_OFF 197632
#define SMEM_RSB_STAGE_BYTES 512
#define SMEM_RSB_STRIDE 512
#define SMEM_TOTAL 198144
#define THREADS 320
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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_7c591421af4c01babb73(const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap w_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ rowscale, float* __restrict__ gamma, int num_heads, int num_qk_heads, int* __restrict__ chunk_bos, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
    #define f32_empty_addr (mbar_base + 48)
    #define xs_ready_addr (mbar_base + 64)
    #define vp_ready_addr (mbar_base + 72)
    #define bv_ready_addr (mbar_base + 80)
    #define h_ready_addr (mbar_base + 88)

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
    float* kf = reinterpret_cast<float*>(smem_raw + 132096);
    const int kf_addr = smem + 132096;
    float* rsb = reinterpret_cast<float*>(smem_raw + 197632);
    const int rsb_addr = smem + 197632;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 16, 256);
            mbarrier_init(smem + 24, 256);
            // --- pipeline 'fring' ---
            // f32_full: 2 barriers, init_count=256
            mbarrier_init(smem + 32, 256);
            mbarrier_init(smem + 40, 256);
            // f32_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'one' ---
            // xs_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
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
    if (warp <= 7) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seg = walk_seg[blockIdx.x / num_heads];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int lane_v = warp_in_wg * 32 + lane;
            int row_base = warp_in_wg * 32 << 16;
            unsigned int warp_group_idx = ((unsigned int)tid - 0) / 128;
            int ch = warp_group_idx;
            int kc0 = ch * 64;
            int tc0 = ch * 32;
            unsigned int ls = 0;
            unsigned int fs = 0;
            int w_row = warp_in_wg * 16 + (lane & 7) + (lane >> 4 << 3);
            int w_col = (lane >> 3 & 1) * 64;
            int t_col = (warp_in_wg & 1) * 32 + lane;
            int k_row0 = (warp_in_wg >> 1) * 64;
            unsigned int mask_hi = 4294901760u;
            unsigned int mask_tf32 = 4294959104u;
            unsigned int half_tf32 = 4096;
            float init_vals[32];
            int icol_1_0 = kc0;
            #pragma unroll
            for (int j_1_0 = 0; j_1_0 < 32; j_1_0++) {
                float one_here_1_0 = 0.0f;
                if (init_mode == 0) {
                    if (icol_1_0 + j_1_0 == lane_v) {
                        one_here_1_0 = 1.0f;
                    }
                }
                init_vals[j_1_0] = one_here_1_0;
            }
            tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)icol_1_0, init_vals);
            int icol_1_1 = kc0 + 32;
            #pragma unroll
            for (int j_1_1 = 0; j_1_1 < 32; j_1_1++) {
                float one_here_1_1 = 0.0f;
                if (init_mode == 0) {
                    if (icol_1_1 + j_1_1 == lane_v) {
                        one_here_1_1 = 1.0f;
                    }
                }
                init_vals[j_1_1] = one_here_1_1;
            }
            tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)icol_1_1, init_vals);
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
            int zcol_2_0 = kc0;
            tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + zcol_2_0), zeros);
            int zcol_2_1 = kc0 + 32;
            tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + zcol_2_1), zeros);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float decay = 0.0f;
            float rs_next = 0.0f;
            unsigned int _phase_f32_empty = 1;
            unsigned int _phase_tiles_full = 0;
            if (nc > 0) {
                long long g_first = (long long)c0 * (long long)num_heads + (long long)head;
                decay = gamma[g_first];
                if (ch == 0) {
                    if (lane_v < 64) {
                        long long rs_first = (long long)(row0 + lane_v) * (long long)num_heads + (long long)head;
                        rs_next = rowscale[rs_first];
                    }
                }
                mbarrier_wait(f32_empty_addr + (fs) * 8, _phase_f32_empty);
                mbarrier_wait(tiles_full_addr + (ls) * 8, _phase_tiles_full);
                unsigned int words[4];
                float wide[8];
                int col_3_0 = w_col + ch * 4 * 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3_0 / 64 * 8192 + w_row * 128 + col_3_0 % 64 * 2 ^ (col_3_0 / 64 * 8192 + w_row * 128 + col_3_0 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3_0 / 32 * 8192 + w_row * 128 + col_3_0 % 32 * 4 ^ (col_3_0 / 32 * 8192 + w_row * 128 + col_3_0 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3_0 + 4) / 32 * 8192 + w_row * 128 + (col_3_0 + 4) % 32 * 4 ^ ((col_3_0 + 4) / 32 * 8192 + w_row * 128 + (col_3_0 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_3_1 = w_col + (ch * 4 + 1) * 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3_1 / 64 * 8192 + w_row * 128 + col_3_1 % 64 * 2 ^ (col_3_1 / 64 * 8192 + w_row * 128 + col_3_1 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3_1 / 32 * 8192 + w_row * 128 + col_3_1 % 32 * 4 ^ (col_3_1 / 32 * 8192 + w_row * 128 + col_3_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3_1 + 4) / 32 * 8192 + w_row * 128 + (col_3_1 + 4) % 32 * 4 ^ ((col_3_1 + 4) / 32 * 8192 + w_row * 128 + (col_3_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_3_2 = w_col + (ch * 4 + 2) * 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3_2 / 64 * 8192 + w_row * 128 + col_3_2 % 64 * 2 ^ (col_3_2 / 64 * 8192 + w_row * 128 + col_3_2 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3_2 / 32 * 8192 + w_row * 128 + col_3_2 % 32 * 4 ^ (col_3_2 / 32 * 8192 + w_row * 128 + col_3_2 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3_2 + 4) / 32 * 8192 + w_row * 128 + (col_3_2 + 4) % 32 * 4 ^ ((col_3_2 + 4) / 32 * 8192 + w_row * 128 + (col_3_2 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_3_3 = w_col + (ch * 4 + 3) * 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3_3 / 64 * 8192 + w_row * 128 + col_3_3 % 64 * 2 ^ (col_3_3 / 64 * 8192 + w_row * 128 + col_3_3 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3_3 / 32 * 8192 + w_row * 128 + col_3_3 % 32 * 4 ^ (col_3_3 / 32 * 8192 + w_row * 128 + col_3_3 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3_3 + 4) / 32 * 8192 + w_row * 128 + (col_3_3 + 4) % 32 * 4 ^ ((col_3_3 + 4) / 32 * 8192 + w_row * 128 + (col_3_3 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int krow_1_0 = k_row0 + ch * 4 * 8;
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
                {
                    uint32_t _addr_0 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_0 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_0 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_1 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_2 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_3 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_4 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_5 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_6 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_7 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_0 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_0 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(wide[7]) : "memory");
                }
                int krow_1_1 = k_row0 + (ch * 4 + 1) * 8;
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
                {
                    uint32_t _addr_8 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_9 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_10 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_11 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_12 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_13 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_14 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_15 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(wide[7]) : "memory");
                }
                int krow_1_2 = k_row0 + (ch * 4 + 2) * 8;
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
                {
                    uint32_t _addr_16 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_2 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_2 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_17 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_18 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_19 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_20 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_21 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_22 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_23 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_2 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_2 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(wide[7]) : "memory");
                }
                int krow_1_3 = k_row0 + (ch * 4 + 3) * 8;
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
                {
                    uint32_t _addr_24 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_3 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_3 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_25 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_26 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_27 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_28 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_29 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_30 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_31 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_3 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_3 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(wide[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(f32_full_addr + (fs) * 8);
                mbarrier_arrive(tiles_empty_addr + (ls) * 8);
                ls += 1;
                if (ls == 2) { ls = 0; _phase_tiles_full ^= 1; }
                fs += 1;
                if (fs == 2) { fs = 0; _phase_f32_empty ^= 1; }
                if (ch == 0) {
                    if (lane_v < 64) {
                        {
                            uint32_t _addr_32 = static_cast<uint32_t>(rsb_addr + (unsigned int)(lane_v * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(rs_next) : "memory");
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
            }
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int par = c % 2;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int hcol = kc0;
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)hcol));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)hcol + 32));
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + hcol)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + hcol) + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j = 0; j < 64; j++) {
                    _tmem_load_0[j] = _tmem_load_0[j] + _tmem_load_1[j];
                }
                int col = hcol;
                unsigned int hi[32];
                unsigned int lo[32];
                #pragma unroll
                for (int j_1 = 0; j_1 < 32; j_1++) {
                    hi[j_1] = __as_u32(_tmem_load_0[j_1]) + half_tf32 & mask_tf32;
                    lo[j_1] = __as_u32(_tmem_load_0[j_1] - __uint_as_float(hi[j_1])) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)col), "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]), "r"(hi[4]), "r"(hi[5]), "r"(hi[6]), "r"(hi[7]), "r"(hi[8]), "r"(hi[9]), "r"(hi[10]), "r"(hi[11]), "r"(hi[12]), "r"(hi[13]), "r"(hi[14]), "r"(hi[15]), "r"(hi[16]), "r"(hi[17]), "r"(hi[18]), "r"(hi[19]), "r"(hi[20]), "r"(hi[21]), "r"(hi[22]), "r"(hi[23]), "r"(hi[24]), "r"(hi[25]), "r"(hi[26]), "r"(hi[27]), "r"(hi[28]), "r"(hi[29]), "r"(hi[30]), "r"(hi[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + col)), "r"(lo[0]), "r"(lo[1]), "r"(lo[2]), "r"(lo[3]), "r"(lo[4]), "r"(lo[5]), "r"(lo[6]), "r"(lo[7]), "r"(lo[8]), "r"(lo[9]), "r"(lo[10]), "r"(lo[11]), "r"(lo[12]), "r"(lo[13]), "r"(lo[14]), "r"(lo[15]), "r"(lo[16]), "r"(lo[17]), "r"(lo[18]), "r"(lo[19]), "r"(lo[20]), "r"(lo[21]), "r"(lo[22]), "r"(lo[23]), "r"(lo[24]), "r"(lo[25]), "r"(lo[26]), "r"(lo[27]), "r"(lo[28]), "r"(lo[29]), "r"(lo[30]), "r"(lo[31]));
                int col_0 = hcol + 32;
                unsigned int hi_1[32];
                unsigned int lo_2[32];
                #pragma unroll
                for (int j_2 = 0; j_2 < 32; j_2++) {
                    hi_1[j_2] = __as_u32(_tmem_load_0[32 + j_2]) + half_tf32 & mask_tf32;
                    lo_2[j_2] = __as_u32(_tmem_load_0[32 + j_2] - __uint_as_float(hi_1[j_2])) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)col_0), "r"(hi_1[0]), "r"(hi_1[1]), "r"(hi_1[2]), "r"(hi_1[3]), "r"(hi_1[4]), "r"(hi_1[5]), "r"(hi_1[6]), "r"(hi_1[7]), "r"(hi_1[8]), "r"(hi_1[9]), "r"(hi_1[10]), "r"(hi_1[11]), "r"(hi_1[12]), "r"(hi_1[13]), "r"(hi_1[14]), "r"(hi_1[15]), "r"(hi_1[16]), "r"(hi_1[17]), "r"(hi_1[18]), "r"(hi_1[19]), "r"(hi_1[20]), "r"(hi_1[21]), "r"(hi_1[22]), "r"(hi_1[23]), "r"(hi_1[24]), "r"(hi_1[25]), "r"(hi_1[26]), "r"(hi_1[27]), "r"(hi_1[28]), "r"(hi_1[29]), "r"(hi_1[30]), "r"(hi_1[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + col_0)), "r"(lo_2[0]), "r"(lo_2[1]), "r"(lo_2[2]), "r"(lo_2[3]), "r"(lo_2[4]), "r"(lo_2[5]), "r"(lo_2[6]), "r"(lo_2[7]), "r"(lo_2[8]), "r"(lo_2[9]), "r"(lo_2[10]), "r"(lo_2[11]), "r"(lo_2[12]), "r"(lo_2[13]), "r"(lo_2[14]), "r"(lo_2[15]), "r"(lo_2[16]), "r"(lo_2[17]), "r"(lo_2[18]), "r"(lo_2[19]), "r"(lo_2[20]), "r"(lo_2[21]), "r"(lo_2[22]), "r"(lo_2[23]), "r"(lo_2[24]), "r"(lo_2[25]), "r"(lo_2[26]), "r"(lo_2[27]), "r"(lo_2[28]), "r"(lo_2[29]), "r"(lo_2[30]), "r"(lo_2[31]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(xs_ready_addr);
                float xd[32];
                #pragma unroll
                for (int j_3 = 0; j_3 < 32; j_3++) {
                    xd[j_3] = _tmem_load_0[j_3] * decay;
                }
                tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)col, xd);
                float xd_3[32];
                #pragma unroll
                for (int j_4 = 0; j_4 < 32; j_4++) {
                    xd_3[j_4] = _tmem_load_0[32 + j_4] * decay;
                }
                tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)col_0, xd_3);
                if (c + 1 < nc) {
                    long long g_next = (long long)(c0 + c + 1) * (long long)num_heads + (long long)head;
                    decay = gamma[g_next];
                    if (ch == 0) {
                        if (lane_v < 64) {
                            int t_next = row0 + (c + 1) * 64 + lane_v;
                            long long rs_next_index = (long long)t_next * (long long)num_heads + (long long)head;
                            rs_next = rowscale[rs_next_index];
                        }
                    }
                }
                long long u_base = ((long long)(row0 + c * 64) * (long long)num_heads + (long long)head) * 128 + (long long)lane_v;
                int u_stride = num_heads * 128;
                float uv[32];
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
                if (use_u == 1) {
                    #pragma unroll
                    for (int t = 0; t < 32; t++) {
                        {
                            __nv_bfloat16 _bf16_33 = *reinterpret_cast<const __nv_bfloat16*>(u + u_base + (long long)((tc0 + t) * u_stride));
                            uv[t] = __bfloat162float(_bf16_33);
                        }
                    }
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)tc0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float bvv[32];
                #pragma unroll
                for (int j_5 = 0; j_5 < 32; j_5++) {
                    bvv[j_5] = uv[j_5] - _tmem_load_2[j_5];
                }
                int bcol = tc0;
                float rs[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[0])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(0) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[4])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(4) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[8])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(8) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[12])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(12) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[16])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(16) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[20])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(20) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[24])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(24) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs[28])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs[(28) + 3]))
                    : "r"(rsb_addr + (unsigned int)(par * 256 + bcol * 4 + 112)));
                float bs[32];
                #pragma unroll
                for (int j_6 = 0; j_6 < 32; j_6++) {
                    bs[j_6] = bvv[j_6] * rs[j_6];
                }
                unsigned int bhi[32];
                unsigned int blo[32];
                #pragma unroll
                for (int j_7 = 0; j_7 < 32; j_7++) {
                    bhi[j_7] = __as_u32(bs[j_7]) + half_tf32 & mask_tf32;
                    blo[j_7] = __as_u32(bs[j_7] - __uint_as_float(bhi[j_7])) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)bcol), "r"(bhi[0]), "r"(bhi[1]), "r"(bhi[2]), "r"(bhi[3]), "r"(bhi[4]), "r"(bhi[5]), "r"(bhi[6]), "r"(bhi[7]), "r"(bhi[8]), "r"(bhi[9]), "r"(bhi[10]), "r"(bhi[11]), "r"(bhi[12]), "r"(bhi[13]), "r"(bhi[14]), "r"(bhi[15]), "r"(bhi[16]), "r"(bhi[17]), "r"(bhi[18]), "r"(bhi[19]), "r"(bhi[20]), "r"(bhi[21]), "r"(bhi[22]), "r"(bhi[23]), "r"(bhi[24]), "r"(bhi[25]), "r"(bhi[26]), "r"(bhi[27]), "r"(bhi[28]), "r"(bhi[29]), "r"(bhi[30]), "r"(bhi[31]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 448 + (unsigned int)row_base + (unsigned int)bcol), "r"(blo[0]), "r"(blo[1]), "r"(blo[2]), "r"(blo[3]), "r"(blo[4]), "r"(blo[5]), "r"(blo[6]), "r"(blo[7]), "r"(blo[8]), "r"(blo[9]), "r"(blo[10]), "r"(blo[11]), "r"(blo[12]), "r"(blo[13]), "r"(blo[14]), "r"(blo[15]), "r"(blo[16]), "r"(blo[17]), "r"(blo[18]), "r"(blo[19]), "r"(blo[20]), "r"(blo[21]), "r"(blo[22]), "r"(blo[23]), "r"(blo[24]), "r"(blo[25]), "r"(blo[26]), "r"(blo[27]), "r"(blo[28]), "r"(blo[29]), "r"(blo[30]), "r"(blo[31]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    mbarrier_wait(f32_empty_addr + (fs) * 8, _phase_f32_empty);
                    mbarrier_wait(tiles_full_addr + (ls) * 8, _phase_tiles_full);
                    unsigned int words_1[4];
                    float wide_1[8];
                    int col_1 = w_col + ch * 4 * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_1 / 64 * 8192 + w_row * 128 + col_1 % 64 * 2 ^ (col_1 / 64 * 8192 + w_row * 128 + col_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_1 / 32 * 8192 + w_row * 128 + col_1 % 32 * 4 ^ (col_1 / 32 * 8192 + w_row * 128 + col_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_1 + 4) / 32 * 8192 + w_row * 128 + (col_1 + 4) % 32 * 4 ^ ((col_1 + 4) / 32 * 8192 + w_row * 128 + (col_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_2 = w_col + (ch * 4 + 1) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_2 / 64 * 8192 + w_row * 128 + col_2 % 64 * 2 ^ (col_2 / 64 * 8192 + w_row * 128 + col_2 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_2 / 32 * 8192 + w_row * 128 + col_2 % 32 * 4 ^ (col_2 / 32 * 8192 + w_row * 128 + col_2 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_2 + 4) / 32 * 8192 + w_row * 128 + (col_2 + 4) % 32 * 4 ^ ((col_2 + 4) / 32 * 8192 + w_row * 128 + (col_2 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_3 = w_col + (ch * 4 + 2) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3 / 64 * 8192 + w_row * 128 + col_3 % 64 * 2 ^ (col_3 / 64 * 8192 + w_row * 128 + col_3 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3 / 32 * 8192 + w_row * 128 + col_3 % 32 * 4 ^ (col_3 / 32 * 8192 + w_row * 128 + col_3 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3 + 4) / 32 * 8192 + w_row * 128 + (col_3 + 4) % 32 * 4 ^ ((col_3 + 4) / 32 * 8192 + w_row * 128 + (col_3 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_4 = w_col + (ch * 4 + 3) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_4 / 64 * 8192 + w_row * 128 + col_4 % 64 * 2 ^ (col_4 / 64 * 8192 + w_row * 128 + col_4 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_4 / 32 * 8192 + w_row * 128 + col_4 % 32 * 4 ^ (col_4 / 32 * 8192 + w_row * 128 + col_4 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_4 + 4) / 32 * 8192 + w_row * 128 + (col_4 + 4) % 32 * 4 ^ ((col_4 + 4) / 32 * 8192 + w_row * 128 + (col_4 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int krow_1_4 = k_row0 + ch * 4 * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_4 / 64 * 8192 + t_col * 128 + krow_1_4 % 64 * 2 ^ (krow_1_4 / 64 * 8192 + t_col * 128 + krow_1_4 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_34 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_4 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_4 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_35 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_36 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_37 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_38 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_39 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_40 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_41 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_4 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_4 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(wide_1[7]) : "memory");
                    }
                    int krow_1_5 = k_row0 + (ch * 4 + 1) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_5 / 64 * 8192 + t_col * 128 + krow_1_5 % 64 * 2 ^ (krow_1_5 / 64 * 8192 + t_col * 128 + krow_1_5 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_42 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_5 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_5 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_43 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_44 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_45 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_46 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_47 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_48 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_49 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_5 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_5 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(wide_1[7]) : "memory");
                    }
                    int krow_1_6 = k_row0 + (ch * 4 + 2) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_6 / 64 * 8192 + t_col * 128 + krow_1_6 % 64 * 2 ^ (krow_1_6 / 64 * 8192 + t_col * 128 + krow_1_6 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_50 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_6 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_6 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_51 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_52 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_53 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_54 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_55 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_56 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_57 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_6 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_6 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(wide_1[7]) : "memory");
                    }
                    int krow_1_7 = k_row0 + (ch * 4 + 3) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((k_land_addr + ls * 16384 + (unsigned int)(krow_1_7 / 64 * 8192 + t_col * 128 + krow_1_7 % 64 * 2 ^ (krow_1_7 / 64 * 8192 + t_col * 128 + krow_1_7 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_58 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1_7 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1_7 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_59 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_60 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_61 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_62 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_63 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_64 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_65 = static_cast<uint32_t>((kf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1_7 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1_7 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(wide_1[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(f32_full_addr + (fs) * 8);
                    mbarrier_arrive(tiles_empty_addr + (ls) * 8);
                    ls += 1;
                    if (ls == 2) { ls = 0; _phase_tiles_full ^= 1; }
                    fs += 1;
                    if (fs == 2) { fs = 0; _phase_f32_empty ^= 1; }
                    if (ch == 0) {
                        if (lane_v < 64) {
                            {
                                uint32_t _addr_66 = static_cast<uint32_t>(rsb_addr + (unsigned int)((1 - par) * 256 + lane_v * 4));
                                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(rs_next) : "memory");
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            int slot = out_index[seg];
            if (slot >= 0) {
                long long tile = (long long)((head / out_hg * out_n + slot) * out_hg + head % out_hg) * (long long)tile_stride * 16384;
                int fcol_1_0 = kc0;
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)fcol_1_0));
                float _tmem_load_7_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_7_0[0]), "=f"(_tmem_load_7_0[1]), "=f"(_tmem_load_7_0[2]), "=f"(_tmem_load_7_0[3]), "=f"(_tmem_load_7_0[4]), "=f"(_tmem_load_7_0[5]), "=f"(_tmem_load_7_0[6]), "=f"(_tmem_load_7_0[7]), "=f"(_tmem_load_7_0[8]), "=f"(_tmem_load_7_0[9]), "=f"(_tmem_load_7_0[10]), "=f"(_tmem_load_7_0[11]), "=f"(_tmem_load_7_0[12]), "=f"(_tmem_load_7_0[13]), "=f"(_tmem_load_7_0[14]), "=f"(_tmem_load_7_0[15]), "=f"(_tmem_load_7_0[16]), "=f"(_tmem_load_7_0[17]), "=f"(_tmem_load_7_0[18]), "=f"(_tmem_load_7_0[19]), "=f"(_tmem_load_7_0[20]), "=f"(_tmem_load_7_0[21]), "=f"(_tmem_load_7_0[22]), "=f"(_tmem_load_7_0[23]), "=f"(_tmem_load_7_0[24]), "=f"(_tmem_load_7_0[25]), "=f"(_tmem_load_7_0[26]), "=f"(_tmem_load_7_0[27]), "=f"(_tmem_load_7_0[28]), "=f"(_tmem_load_7_0[29]), "=f"(_tmem_load_7_0[30]), "=f"(_tmem_load_7_0[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + fcol_1_0)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 32; j_1_2++) {
                    _tmem_load_3[j_1_2] = _tmem_load_3[j_1_2] + _tmem_load_7_0[j_1_2];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_1_3 = 0; j_1_3 < 32; j_1_3++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((fcol_1_0 + j_1_3) * 128 + lane_v))) = _tmem_load_3[j_1_3];
                    }
                } else {
                    long long row_at_1_0 = tile + (long long)(lane_v * 128 + fcol_1_0);
                    {
                        float4 _v4 = make_float4(_tmem_load_3[0 + 0], _tmem_load_3[0 + 1], _tmem_load_3[0 + 2], _tmem_load_3[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[4 + 0], _tmem_load_3[4 + 1], _tmem_load_3[4 + 2], _tmem_load_3[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[8 + 0], _tmem_load_3[8 + 1], _tmem_load_3[8 + 2], _tmem_load_3[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[12 + 0], _tmem_load_3[12 + 1], _tmem_load_3[12 + 2], _tmem_load_3[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[16 + 0], _tmem_load_3[16 + 1], _tmem_load_3[16 + 2], _tmem_load_3[16 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[20 + 0], _tmem_load_3[20 + 1], _tmem_load_3[20 + 2], _tmem_load_3[20 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[24 + 0], _tmem_load_3[24 + 1], _tmem_load_3[24 + 2], _tmem_load_3[24 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_3[28 + 0], _tmem_load_3[28 + 1], _tmem_load_3[28 + 2], _tmem_load_3[28 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 28) = _v4;
                    }
                }
                int fcol_1_1 = kc0 + 32;
                float _tmem_load_14[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_14[0]), "=f"(_tmem_load_14[1]), "=f"(_tmem_load_14[2]), "=f"(_tmem_load_14[3]), "=f"(_tmem_load_14[4]), "=f"(_tmem_load_14[5]), "=f"(_tmem_load_14[6]), "=f"(_tmem_load_14[7]), "=f"(_tmem_load_14[8]), "=f"(_tmem_load_14[9]), "=f"(_tmem_load_14[10]), "=f"(_tmem_load_14[11]), "=f"(_tmem_load_14[12]), "=f"(_tmem_load_14[13]), "=f"(_tmem_load_14[14]), "=f"(_tmem_load_14[15]), "=f"(_tmem_load_14[16]), "=f"(_tmem_load_14[17]), "=f"(_tmem_load_14[18]), "=f"(_tmem_load_14[19]), "=f"(_tmem_load_14[20]), "=f"(_tmem_load_14[21]), "=f"(_tmem_load_14[22]), "=f"(_tmem_load_14[23]), "=f"(_tmem_load_14[24]), "=f"(_tmem_load_14[25]), "=f"(_tmem_load_14[26]), "=f"(_tmem_load_14[27]), "=f"(_tmem_load_14[28]), "=f"(_tmem_load_14[29]), "=f"(_tmem_load_14[30]), "=f"(_tmem_load_14[31])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)fcol_1_1));
                float _tmem_load_7_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_7_1[0]), "=f"(_tmem_load_7_1[1]), "=f"(_tmem_load_7_1[2]), "=f"(_tmem_load_7_1[3]), "=f"(_tmem_load_7_1[4]), "=f"(_tmem_load_7_1[5]), "=f"(_tmem_load_7_1[6]), "=f"(_tmem_load_7_1[7]), "=f"(_tmem_load_7_1[8]), "=f"(_tmem_load_7_1[9]), "=f"(_tmem_load_7_1[10]), "=f"(_tmem_load_7_1[11]), "=f"(_tmem_load_7_1[12]), "=f"(_tmem_load_7_1[13]), "=f"(_tmem_load_7_1[14]), "=f"(_tmem_load_7_1[15]), "=f"(_tmem_load_7_1[16]), "=f"(_tmem_load_7_1[17]), "=f"(_tmem_load_7_1[18]), "=f"(_tmem_load_7_1[19]), "=f"(_tmem_load_7_1[20]), "=f"(_tmem_load_7_1[21]), "=f"(_tmem_load_7_1[22]), "=f"(_tmem_load_7_1[23]), "=f"(_tmem_load_7_1[24]), "=f"(_tmem_load_7_1[25]), "=f"(_tmem_load_7_1[26]), "=f"(_tmem_load_7_1[27]), "=f"(_tmem_load_7_1[28]), "=f"(_tmem_load_7_1[29]), "=f"(_tmem_load_7_1[30]), "=f"(_tmem_load_7_1[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(128 + fcol_1_1)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_4 = 0; j_1_4 < 32; j_1_4++) {
                    _tmem_load_14[j_1_4] = _tmem_load_14[j_1_4] + _tmem_load_7_1[j_1_4];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_1_5 = 0; j_1_5 < 32; j_1_5++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((fcol_1_1 + j_1_5) * 128 + lane_v))) = _tmem_load_14[j_1_5];
                    }
                } else {
                    long long row_at_1_1 = tile + (long long)(lane_v * 128 + fcol_1_1);
                    {
                        float4 _v4 = make_float4(_tmem_load_14[0 + 0], _tmem_load_14[0 + 1], _tmem_load_14[0 + 2], _tmem_load_14[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[4 + 0], _tmem_load_14[4 + 1], _tmem_load_14[4 + 2], _tmem_load_14[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[8 + 0], _tmem_load_14[8 + 1], _tmem_load_14[8 + 2], _tmem_load_14[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[12 + 0], _tmem_load_14[12 + 1], _tmem_load_14[12 + 2], _tmem_load_14[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[16 + 0], _tmem_load_14[16 + 1], _tmem_load_14[16 + 2], _tmem_load_14[16 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[20 + 0], _tmem_load_14[20 + 1], _tmem_load_14[20 + 2], _tmem_load_14[20 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[24 + 0], _tmem_load_14[24 + 1], _tmem_load_14[24 + 2], _tmem_load_14[24 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[28 + 0], _tmem_load_14[28 + 1], _tmem_load_14[28 + 2], _tmem_load_14[28 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 28) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
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
    if (warp == 9) {
        { // mma_main
            int head_2 = blockIdx.x % num_heads;
            int seg_2 = walk_seg[blockIdx.x / num_heads];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int fs_1 = 0;
            unsigned int _phase_f32_full = 0;
            unsigned int _phase_xs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(f32_full_addr + (fs_1) * 8, _phase_f32_full);
                    mbarrier_wait(xs_ready_addr, _phase_xs_ready_0);
                    _phase_xs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((wf_addr) >> 4) & 0x7FFF) + (fs_1) * 2048;
#else
                    int _smem_desc_lo_0 = (((wf_addr) >> 4) & 0x3FFF) + (fs_1) * 2048;
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
                    int _smem_desc_lo_1 = (((wf_addr) >> 4) & 0x7FFF) + (fs_1) * 2048;
#else
                    int _smem_desc_lo_1 = (((wf_addr) >> 4) & 0x3FFF) + (fs_1) * 2048;
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
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((kf_addr) >> 4) & 0x7FFF) + (fs_1) * 2048;
#else
                    int _smem_desc_lo_2 = (((kf_addr) >> 4) & 0x3FFF) + (fs_1) * 2048;
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
                    :: "r"((tmem_x_split + (128))), "r"(_smem_desc_lo_2), "r"(tmem_b_lo), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((kf_addr) >> 4) & 0x7FFF) + (fs_1) * 2048;
#else
                    int _smem_desc_lo_3 = (((kf_addr) >> 4) & 0x3FFF) + (fs_1) * 2048;
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
                    :: "r"((tmem_x_split + (128))), "r"(_smem_desc_lo_3), "r"(tmem_b_hi), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(f32_empty_addr + (fs_1) * 8);
                    fs_1 += 1;
                    if (fs_1 == 2) { fs_1 = 0; _phase_f32_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
