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
#define SMEM_KG_PANEL_OFF 33792
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_W_LAND_OFF 1024
#define SMEM_W_LAND_STAGE_BYTES 16384
#define SMEM_W_LAND_STRIDE 16384
#define SMEM_KG_LAND_OFF 33792
#define SMEM_KG_LAND_STAGE_BYTES 16384
#define SMEM_KG_LAND_STRIDE 16384
#define SMEM_WF_OFF 66560
#define SMEM_WF_STAGE_BYTES 32768
#define SMEM_WF_STRIDE 32768
#define SMEM_KGF_OFF 132096
#define SMEM_KGF_STAGE_BYTES 32768
#define SMEM_KGF_STRIDE 32768
#define SMEM_DEC_OFF 197632
#define SMEM_DEC_STAGE_BYTES 1024
#define SMEM_DEC_STRIDE 1024
#define SMEM_TOTAL 198656
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
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
kernel_cake_kda_train_64ba2583f28f7d4c0102(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_panel_addr = smem + 33792;
    __nv_bfloat16* w_land = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_land_addr = smem + 1024;
    __nv_bfloat16* kg_land = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_land_addr = smem + 33792;
    float* wf = reinterpret_cast<float*>(smem_raw + 66560);
    const int wf_addr = smem + 66560;
    float* kgf = reinterpret_cast<float*>(smem_raw + 132096);
    const int kgf_addr = smem + 132096;
    float* dec = reinterpret_cast<float*>(smem_raw + 197632);
    const int dec_addr = smem + 197632;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

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
            // f32_full: 2 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            mbarrier_init(smem + 40, 128);
            // f32_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'one' ---
            // xs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
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
            unsigned int fs = 0;
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
            for (int j_1 = 0; j_1 < 32; j_1++) {
                float one_here_1 = 0.0f;
                if (init_mode == 0) {
                    if (32 + j_1 == lane_v) {
                        one_here_1 = 1.0f;
                    }
                }
                init_vals[j_1] = one_here_1;
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
                mbarrier_wait(f32_empty_addr + (fs) * 8, _phase_f32_empty);
                mbarrier_wait(tiles_full_addr + (ls) * 8, _phase_tiles_full);
                unsigned int words[4];
                float wide[8];
                int col = w_col;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col / 64 * 8192 + w_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + w_row * 128 + col % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col / 32 * 8192 + w_row * 128 + col % 32 * 4 ^ (col / 32 * 8192 + w_row * 128 + col % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col + 4) / 32 * 8192 + w_row * 128 + (col + 4) % 32 * 4 ^ ((col + 4) / 32 * 8192 + w_row * 128 + (col + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_0 = w_col + 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_0 / 64 * 8192 + w_row * 128 + col_0 % 64 * 2 ^ (col_0 / 64 * 8192 + w_row * 128 + col_0 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_0 / 32 * 8192 + w_row * 128 + col_0 % 32 * 4 ^ (col_0 / 32 * 8192 + w_row * 128 + col_0 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_0 + 4) / 32 * 8192 + w_row * 128 + (col_0 + 4) % 32 * 4 ^ ((col_0 + 4) / 32 * 8192 + w_row * 128 + (col_0 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_1 = w_col + 16;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_1 / 64 * 8192 + w_row * 128 + col_1 % 64 * 2 ^ (col_1 / 64 * 8192 + w_row * 128 + col_1 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_1 / 32 * 8192 + w_row * 128 + col_1 % 32 * 4 ^ (col_1 / 32 * 8192 + w_row * 128 + col_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_1 + 4) / 32 * 8192 + w_row * 128 + (col_1 + 4) % 32 * 4 ^ ((col_1 + 4) / 32 * 8192 + w_row * 128 + (col_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_2 = w_col + 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_2 / 64 * 8192 + w_row * 128 + col_2 % 64 * 2 ^ (col_2 / 64 * 8192 + w_row * 128 + col_2 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_2 / 32 * 8192 + w_row * 128 + col_2 % 32 * 4 ^ (col_2 / 32 * 8192 + w_row * 128 + col_2 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_2 + 4) / 32 * 8192 + w_row * 128 + (col_2 + 4) % 32 * 4 ^ ((col_2 + 4) / 32 * 8192 + w_row * 128 + (col_2 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_3 = w_col + 32;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3 / 64 * 8192 + w_row * 128 + col_3 % 64 * 2 ^ (col_3 / 64 * 8192 + w_row * 128 + col_3 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3 / 32 * 8192 + w_row * 128 + col_3 % 32 * 4 ^ (col_3 / 32 * 8192 + w_row * 128 + col_3 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3 + 4) / 32 * 8192 + w_row * 128 + (col_3 + 4) % 32 * 4 ^ ((col_3 + 4) / 32 * 8192 + w_row * 128 + (col_3 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_4 = w_col + 40;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_4 / 64 * 8192 + w_row * 128 + col_4 % 64 * 2 ^ (col_4 / 64 * 8192 + w_row * 128 + col_4 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_4 / 32 * 8192 + w_row * 128 + col_4 % 32 * 4 ^ (col_4 / 32 * 8192 + w_row * 128 + col_4 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_4 + 4) / 32 * 8192 + w_row * 128 + (col_4 + 4) % 32 * 4 ^ ((col_4 + 4) / 32 * 8192 + w_row * 128 + (col_4 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_5 = w_col + 48;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_5 / 64 * 8192 + w_row * 128 + col_5 % 64 * 2 ^ (col_5 / 64 * 8192 + w_row * 128 + col_5 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_5 / 32 * 8192 + w_row * 128 + col_5 % 32 * 4 ^ (col_5 / 32 * 8192 + w_row * 128 + col_5 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_5 + 4) / 32 * 8192 + w_row * 128 + (col_5 + 4) % 32 * 4 ^ ((col_5 + 4) / 32 * 8192 + w_row * 128 + (col_5 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int col_6 = w_col + 56;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_6 / 64 * 8192 + w_row * 128 + col_6 % 64 * 2 ^ (col_6 / 64 * 8192 + w_row * 128 + col_6 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_6 / 32 * 8192 + w_row * 128 + col_6 % 32 * 4 ^ (col_6 / 32 * 8192 + w_row * 128 + col_6 % 32 * 4 >> 7 & 7) << 4))), "f"(wide[0]), "f"(wide[1]), "f"(wide[2]), "f"(wide[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_6 + 4) / 32 * 8192 + w_row * 128 + (col_6 + 4) % 32 * 4 ^ ((col_6 + 4) / 32 * 8192 + w_row * 128 + (col_6 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide[4]), "f"(wide[5]), "f"(wide[6]), "f"(wide[7]) : "memory");
                int krow = k_row0;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow / 64 * 8192 + t_col * 128 + krow % 64 * 2 ^ (krow / 64 * 8192 + t_col * 128 + krow % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_0 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_1 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_2 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_3 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_4 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_5 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_6 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_7 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(wide[7]) : "memory");
                }
                int krow_7 = k_row0 + 8;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_7 / 64 * 8192 + t_col * 128 + krow_7 % 64 * 2 ^ (krow_7 / 64 * 8192 + t_col * 128 + krow_7 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_8 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_7 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_7 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_9 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_10 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_11 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_12 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_13 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_14 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_15 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(wide[7]) : "memory");
                }
                int krow_8 = k_row0 + 16;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_8 / 64 * 8192 + t_col * 128 + krow_8 % 64 * 2 ^ (krow_8 / 64 * 8192 + t_col * 128 + krow_8 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_16 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_8 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_8 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_17 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_18 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_19 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_20 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_21 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_22 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_23 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(wide[7]) : "memory");
                }
                int krow_9 = k_row0 + 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_9 / 64 * 8192 + t_col * 128 + krow_9 % 64 * 2 ^ (krow_9 / 64 * 8192 + t_col * 128 + krow_9 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_24 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_9 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_9 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_25 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_26 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_27 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_28 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_29 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_30 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_31 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(wide[7]) : "memory");
                }
                int krow_10 = k_row0 + 32;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_10 / 64 * 8192 + t_col * 128 + krow_10 % 64 * 2 ^ (krow_10 / 64 * 8192 + t_col * 128 + krow_10 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_32 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_10 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_10 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_33 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_34 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_35 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_36 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_37 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_38 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_39 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(wide[7]) : "memory");
                }
                int krow_11 = k_row0 + 40;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_11 / 64 * 8192 + t_col * 128 + krow_11 % 64 * 2 ^ (krow_11 / 64 * 8192 + t_col * 128 + krow_11 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_40 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_11 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_11 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_41 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_42 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_43 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_44 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_45 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_46 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_47 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(wide[7]) : "memory");
                }
                int krow_12 = k_row0 + 48;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_12 / 64 * 8192 + t_col * 128 + krow_12 % 64 * 2 ^ (krow_12 / 64 * 8192 + t_col * 128 + krow_12 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_48 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_12 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_12 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_49 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_50 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_51 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_52 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_53 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_54 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_55 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(wide[7]) : "memory");
                }
                int krow_13 = k_row0 + 56;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_13 / 64 * 8192 + t_col * 128 + krow_13 % 64 * 2 ^ (krow_13 / 64 * 8192 + t_col * 128 + krow_13 % 64 * 2 >> 7 & 7) << 4))));
                wide[0] = __uint_as_float(words[0] << 16);
                wide[1] = __uint_as_float(words[0] & mask_hi);
                wide[2] = __uint_as_float(words[1] << 16);
                wide[3] = __uint_as_float(words[1] & mask_hi);
                wide[4] = __uint_as_float(words[2] << 16);
                wide[5] = __uint_as_float(words[2] & mask_hi);
                wide[6] = __uint_as_float(words[3] << 16);
                wide[7] = __uint_as_float(words[3] & mask_hi);
                {
                    uint32_t _addr_56 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_13 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_13 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(wide[0]) : "memory");
                }
                {
                    uint32_t _addr_57 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(wide[1]) : "memory");
                }
                {
                    uint32_t _addr_58 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(wide[2]) : "memory");
                }
                {
                    uint32_t _addr_59 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(wide[3]) : "memory");
                }
                {
                    uint32_t _addr_60 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(wide[4]) : "memory");
                }
                {
                    uint32_t _addr_61 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(wide[5]) : "memory");
                }
                {
                    uint32_t _addr_62 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(wide[6]) : "memory");
                }
                {
                    uint32_t _addr_63 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(wide[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(f32_full_addr + (fs) * 8);
                mbarrier_arrive(tiles_empty_addr + (ls) * 8);
                ls += 1;
                if (ls == 2) { ls = 0; _phase_tiles_full ^= 1; }
                fs += 1;
                if (fs == 2) { fs = 0; _phase_f32_empty ^= 1; }
            }
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int par = c % 2;
                long long gn_base = ((long long)(row0 + c * 64 + 64 - 1) * (long long)num_heads + (long long)head) * 128;
                float my_decay = approx_exp2(gk[gn_base + (long long)lane_v]);
                {
                    uint32_t _addr_64 = static_cast<uint32_t>(dec_addr + (unsigned int)(par * 512 + lane_v * 4));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(my_decay) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)row_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)row_base + 32));
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(taddr + 128 + (unsigned int)row_base + 128 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_4 = 0; j_4 < 64; j_4++) {
                    _tmem_load_0[j_4] = _tmem_load_0[j_4] + _tmem_load_1[j_4];
                }
                float decs[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[0])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(0) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[4])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(4) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[8])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(8) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[12])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(12) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[16])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(16) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[20])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(20) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[24])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(24) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs[28])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs[(28) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 112)));
                unsigned int hi[32];
                unsigned int lo[32];
                #pragma unroll
                for (int j_5 = 0; j_5 < 32; j_5++) {
                    hi[j_5] = __as_u32(_tmem_load_0[j_5]) + half_tf32 & mask_tf32;
                }
                float xd[32];
                #pragma unroll
                for (int j_6 = 0; j_6 < 32; j_6++) {
                    xd[j_6] = _tmem_load_0[j_6] * decs[j_6];
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base), "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]), "r"(hi[4]), "r"(hi[5]), "r"(hi[6]), "r"(hi[7]), "r"(hi[8]), "r"(hi[9]), "r"(hi[10]), "r"(hi[11]), "r"(hi[12]), "r"(hi[13]), "r"(hi[14]), "r"(hi[15]), "r"(hi[16]), "r"(hi[17]), "r"(hi[18]), "r"(hi[19]), "r"(hi[20]), "r"(hi[21]), "r"(hi[22]), "r"(hi[23]), "r"(hi[24]), "r"(hi[25]), "r"(hi[26]), "r"(hi[27]), "r"(hi[28]), "r"(hi[29]), "r"(hi[30]), "r"(hi[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base, xd);
                float decs_0[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(0) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(4) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(8) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(12) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(16) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(20) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(24) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_0[(28) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 128 + 112)));
                unsigned int hi_1[32];
                unsigned int lo_2[32];
                #pragma unroll
                for (int j_7 = 0; j_7 < 32; j_7++) {
                    hi_1[j_7] = __as_u32(_tmem_load_0[32 + j_7]) + half_tf32 & mask_tf32;
                }
                float xd_3[32];
                #pragma unroll
                for (int j_8 = 0; j_8 < 32; j_8++) {
                    xd_3[j_8] = _tmem_load_0[32 + j_8] * decs_0[j_8];
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 32), "r"(hi_1[0]), "r"(hi_1[1]), "r"(hi_1[2]), "r"(hi_1[3]), "r"(hi_1[4]), "r"(hi_1[5]), "r"(hi_1[6]), "r"(hi_1[7]), "r"(hi_1[8]), "r"(hi_1[9]), "r"(hi_1[10]), "r"(hi_1[11]), "r"(hi_1[12]), "r"(hi_1[13]), "r"(hi_1[14]), "r"(hi_1[15]), "r"(hi_1[16]), "r"(hi_1[17]), "r"(hi_1[18]), "r"(hi_1[19]), "r"(hi_1[20]), "r"(hi_1[21]), "r"(hi_1[22]), "r"(hi_1[23]), "r"(hi_1[24]), "r"(hi_1[25]), "r"(hi_1[26]), "r"(hi_1[27]), "r"(hi_1[28]), "r"(hi_1[29]), "r"(hi_1[30]), "r"(hi_1[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base + 32, xd_3);
                float _tmem_load_2[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(taddr + (unsigned int)row_base + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(taddr + (unsigned int)row_base + 64 + 32));
                float _tmem_load_3[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(taddr + 128 + (unsigned int)row_base + 192));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                    : "r"(taddr + 128 + (unsigned int)row_base + 192 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_9 = 0; j_9 < 64; j_9++) {
                    _tmem_load_2[j_9] = _tmem_load_2[j_9] + _tmem_load_3[j_9];
                }
                float decs_4[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(0) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(4) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(8) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(12) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(16) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(20) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(24) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_4[(28) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 256 + 112)));
                unsigned int hi_5[32];
                unsigned int lo_6[32];
                #pragma unroll
                for (int j_10 = 0; j_10 < 32; j_10++) {
                    hi_5[j_10] = __as_u32(_tmem_load_2[j_10]) + half_tf32 & mask_tf32;
                }
                float xd_7[32];
                #pragma unroll
                for (int j_11 = 0; j_11 < 32; j_11++) {
                    xd_7[j_11] = _tmem_load_2[j_11] * decs_4[j_11];
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 64), "r"(hi_5[0]), "r"(hi_5[1]), "r"(hi_5[2]), "r"(hi_5[3]), "r"(hi_5[4]), "r"(hi_5[5]), "r"(hi_5[6]), "r"(hi_5[7]), "r"(hi_5[8]), "r"(hi_5[9]), "r"(hi_5[10]), "r"(hi_5[11]), "r"(hi_5[12]), "r"(hi_5[13]), "r"(hi_5[14]), "r"(hi_5[15]), "r"(hi_5[16]), "r"(hi_5[17]), "r"(hi_5[18]), "r"(hi_5[19]), "r"(hi_5[20]), "r"(hi_5[21]), "r"(hi_5[22]), "r"(hi_5[23]), "r"(hi_5[24]), "r"(hi_5[25]), "r"(hi_5[26]), "r"(hi_5[27]), "r"(hi_5[28]), "r"(hi_5[29]), "r"(hi_5[30]), "r"(hi_5[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base + 64, xd_7);
                float decs_8[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(0) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(4) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(8) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(12) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[16])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(16) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[20])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(20) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[24])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(24) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&decs_8[28])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&decs_8[(28) + 3]))
                    : "r"(dec_addr + (unsigned int)(par * 512 + 384 + 112)));
                unsigned int hi_9[32];
                unsigned int lo_10[32];
                #pragma unroll
                for (int j_12 = 0; j_12 < 32; j_12++) {
                    hi_9[j_12] = __as_u32(_tmem_load_2[32 + j_12]) + half_tf32 & mask_tf32;
                }
                float xd_11[32];
                #pragma unroll
                for (int j_13 = 0; j_13 < 32; j_13++) {
                    xd_11[j_13] = _tmem_load_2[32 + j_13] * decs_8[j_13];
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + (unsigned int)row_base + 96), "r"(hi_9[0]), "r"(hi_9[1]), "r"(hi_9[2]), "r"(hi_9[3]), "r"(hi_9[4]), "r"(hi_9[5]), "r"(hi_9[6]), "r"(hi_9[7]), "r"(hi_9[8]), "r"(hi_9[9]), "r"(hi_9[10]), "r"(hi_9[11]), "r"(hi_9[12]), "r"(hi_9[13]), "r"(hi_9[14]), "r"(hi_9[15]), "r"(hi_9[16]), "r"(hi_9[17]), "r"(hi_9[18]), "r"(hi_9[19]), "r"(hi_9[20]), "r"(hi_9[21]), "r"(hi_9[22]), "r"(hi_9[23]), "r"(hi_9[24]), "r"(hi_9[25]), "r"(hi_9[26]), "r"(hi_9[27]), "r"(hi_9[28]), "r"(hi_9[29]), "r"(hi_9[30]), "r"(hi_9[31]));
                tmem_st_x32_f32(taddr + (unsigned int)row_base + 96, xd_11);
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
                            __nv_bfloat16 _bf16_65 = *reinterpret_cast<const __nv_bfloat16*>(u + u_base + (long long)(t * u_stride));
                            uv[t] = __bfloat162float(_bf16_65);
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
                float bv[64];
                #pragma unroll
                for (int j_14 = 0; j_14 < 64; j_14++) {
                    bv[j_14] = uv[j_14] - _tmem_load_4[j_14];
                }
                unsigned int bhi[32];
                unsigned int blo[32];
                #pragma unroll
                for (int j_15 = 0; j_15 < 32; j_15++) {
                    bhi[j_15] = __as_u32(bv[j_15]) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 384 + (unsigned int)row_base), "r"(bhi[0]), "r"(bhi[1]), "r"(bhi[2]), "r"(bhi[3]), "r"(bhi[4]), "r"(bhi[5]), "r"(bhi[6]), "r"(bhi[7]), "r"(bhi[8]), "r"(bhi[9]), "r"(bhi[10]), "r"(bhi[11]), "r"(bhi[12]), "r"(bhi[13]), "r"(bhi[14]), "r"(bhi[15]), "r"(bhi[16]), "r"(bhi[17]), "r"(bhi[18]), "r"(bhi[19]), "r"(bhi[20]), "r"(bhi[21]), "r"(bhi[22]), "r"(bhi[23]), "r"(bhi[24]), "r"(bhi[25]), "r"(bhi[26]), "r"(bhi[27]), "r"(bhi[28]), "r"(bhi[29]), "r"(bhi[30]), "r"(bhi[31]));
                unsigned int bhi_12[32];
                unsigned int blo_13[32];
                #pragma unroll
                for (int j_16 = 0; j_16 < 32; j_16++) {
                    bhi_12[j_16] = __as_u32(bv[32 + j_16]) + half_tf32 & mask_tf32;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 384 + (unsigned int)row_base + 32), "r"(bhi_12[0]), "r"(bhi_12[1]), "r"(bhi_12[2]), "r"(bhi_12[3]), "r"(bhi_12[4]), "r"(bhi_12[5]), "r"(bhi_12[6]), "r"(bhi_12[7]), "r"(bhi_12[8]), "r"(bhi_12[9]), "r"(bhi_12[10]), "r"(bhi_12[11]), "r"(bhi_12[12]), "r"(bhi_12[13]), "r"(bhi_12[14]), "r"(bhi_12[15]), "r"(bhi_12[16]), "r"(bhi_12[17]), "r"(bhi_12[18]), "r"(bhi_12[19]), "r"(bhi_12[20]), "r"(bhi_12[21]), "r"(bhi_12[22]), "r"(bhi_12[23]), "r"(bhi_12[24]), "r"(bhi_12[25]), "r"(bhi_12[26]), "r"(bhi_12[27]), "r"(bhi_12[28]), "r"(bhi_12[29]), "r"(bhi_12[30]), "r"(bhi_12[31]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    mbarrier_wait(f32_empty_addr + (fs) * 8, _phase_f32_empty);
                    mbarrier_wait(tiles_full_addr + (ls) * 8, _phase_tiles_full);
                    unsigned int words_1[4];
                    float wide_1[8];
                    int col_7 = w_col;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_7 / 64 * 8192 + w_row * 128 + col_7 % 64 * 2 ^ (col_7 / 64 * 8192 + w_row * 128 + col_7 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_7 / 32 * 8192 + w_row * 128 + col_7 % 32 * 4 ^ (col_7 / 32 * 8192 + w_row * 128 + col_7 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_7 + 4) / 32 * 8192 + w_row * 128 + (col_7 + 4) % 32 * 4 ^ ((col_7 + 4) / 32 * 8192 + w_row * 128 + (col_7 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_0_1 = w_col + 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_0_1 / 64 * 8192 + w_row * 128 + col_0_1 % 64 * 2 ^ (col_0_1 / 64 * 8192 + w_row * 128 + col_0_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_0_1 / 32 * 8192 + w_row * 128 + col_0_1 % 32 * 4 ^ (col_0_1 / 32 * 8192 + w_row * 128 + col_0_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_0_1 + 4) / 32 * 8192 + w_row * 128 + (col_0_1 + 4) % 32 * 4 ^ ((col_0_1 + 4) / 32 * 8192 + w_row * 128 + (col_0_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_1_1 = w_col + 16;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_1_1 / 64 * 8192 + w_row * 128 + col_1_1 % 64 * 2 ^ (col_1_1 / 64 * 8192 + w_row * 128 + col_1_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_1_1 / 32 * 8192 + w_row * 128 + col_1_1 % 32 * 4 ^ (col_1_1 / 32 * 8192 + w_row * 128 + col_1_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_1_1 + 4) / 32 * 8192 + w_row * 128 + (col_1_1 + 4) % 32 * 4 ^ ((col_1_1 + 4) / 32 * 8192 + w_row * 128 + (col_1_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_2_1 = w_col + 24;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_2_1 / 64 * 8192 + w_row * 128 + col_2_1 % 64 * 2 ^ (col_2_1 / 64 * 8192 + w_row * 128 + col_2_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_2_1 / 32 * 8192 + w_row * 128 + col_2_1 % 32 * 4 ^ (col_2_1 / 32 * 8192 + w_row * 128 + col_2_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_2_1 + 4) / 32 * 8192 + w_row * 128 + (col_2_1 + 4) % 32 * 4 ^ ((col_2_1 + 4) / 32 * 8192 + w_row * 128 + (col_2_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_3_1 = w_col + 32;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_3_1 / 64 * 8192 + w_row * 128 + col_3_1 % 64 * 2 ^ (col_3_1 / 64 * 8192 + w_row * 128 + col_3_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_3_1 / 32 * 8192 + w_row * 128 + col_3_1 % 32 * 4 ^ (col_3_1 / 32 * 8192 + w_row * 128 + col_3_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_3_1 + 4) / 32 * 8192 + w_row * 128 + (col_3_1 + 4) % 32 * 4 ^ ((col_3_1 + 4) / 32 * 8192 + w_row * 128 + (col_3_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_4_1 = w_col + 40;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_4_1 / 64 * 8192 + w_row * 128 + col_4_1 % 64 * 2 ^ (col_4_1 / 64 * 8192 + w_row * 128 + col_4_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_4_1 / 32 * 8192 + w_row * 128 + col_4_1 % 32 * 4 ^ (col_4_1 / 32 * 8192 + w_row * 128 + col_4_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_4_1 + 4) / 32 * 8192 + w_row * 128 + (col_4_1 + 4) % 32 * 4 ^ ((col_4_1 + 4) / 32 * 8192 + w_row * 128 + (col_4_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_5_1 = w_col + 48;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_5_1 / 64 * 8192 + w_row * 128 + col_5_1 % 64 * 2 ^ (col_5_1 / 64 * 8192 + w_row * 128 + col_5_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_5_1 / 32 * 8192 + w_row * 128 + col_5_1 % 32 * 4 ^ (col_5_1 / 32 * 8192 + w_row * 128 + col_5_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_5_1 + 4) / 32 * 8192 + w_row * 128 + (col_5_1 + 4) % 32 * 4 ^ ((col_5_1 + 4) / 32 * 8192 + w_row * 128 + (col_5_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int col_6_1 = w_col + 56;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((w_land_addr + ls * 16384 + (unsigned int)(col_6_1 / 64 * 8192 + w_row * 128 + col_6_1 % 64 * 2 ^ (col_6_1 / 64 * 8192 + w_row * 128 + col_6_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)(col_6_1 / 32 * 8192 + w_row * 128 + col_6_1 % 32 * 4 ^ (col_6_1 / 32 * 8192 + w_row * 128 + col_6_1 % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[0]), "f"(wide_1[1]), "f"(wide_1[2]), "f"(wide_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((wf_addr + fs * 32768 + (unsigned int)((col_6_1 + 4) / 32 * 8192 + w_row * 128 + (col_6_1 + 4) % 32 * 4 ^ ((col_6_1 + 4) / 32 * 8192 + w_row * 128 + (col_6_1 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(wide_1[4]), "f"(wide_1[5]), "f"(wide_1[6]), "f"(wide_1[7]) : "memory");
                    int krow_1 = k_row0;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_1 / 64 * 8192 + t_col * 128 + krow_1 % 64 * 2 ^ (krow_1 / 64 * 8192 + t_col * 128 + krow_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_66 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_67 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_68 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_69 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_70 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_71 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_71), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_72 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_73 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(wide_1[7]) : "memory");
                    }
                    int krow_7_1 = k_row0 + 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_7_1 / 64 * 8192 + t_col * 128 + krow_7_1 % 64 * 2 ^ (krow_7_1 / 64 * 8192 + t_col * 128 + krow_7_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_74 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_7_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_7_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_75 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_76 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_77 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_78 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_79 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_80 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_81 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_7_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_7_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(wide_1[7]) : "memory");
                    }
                    int krow_8_1 = k_row0 + 16;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_8_1 / 64 * 8192 + t_col * 128 + krow_8_1 % 64 * 2 ^ (krow_8_1 / 64 * 8192 + t_col * 128 + krow_8_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_82 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_8_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_8_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_83 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_84 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_85 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_86 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_86), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_87 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_88 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_89 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_8_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_8_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(wide_1[7]) : "memory");
                    }
                    int krow_9_1 = k_row0 + 24;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_9_1 / 64 * 8192 + t_col * 128 + krow_9_1 % 64 * 2 ^ (krow_9_1 / 64 * 8192 + t_col * 128 + krow_9_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_90 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_9_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_9_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_91 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_92 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_93 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_94 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_95 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_96 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_97 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_9_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_9_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(wide_1[7]) : "memory");
                    }
                    int krow_10_1 = k_row0 + 32;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_10_1 / 64 * 8192 + t_col * 128 + krow_10_1 % 64 * 2 ^ (krow_10_1 / 64 * 8192 + t_col * 128 + krow_10_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_98 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_10_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_10_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_99 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_99), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_100 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_100), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_101 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_102 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_103 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_104 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_105 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_10_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_10_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(wide_1[7]) : "memory");
                    }
                    int krow_11_1 = k_row0 + 40;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_11_1 / 64 * 8192 + t_col * 128 + krow_11_1 % 64 * 2 ^ (krow_11_1 / 64 * 8192 + t_col * 128 + krow_11_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_106 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_11_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_11_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_107 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_108 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_108), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_109 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_109), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_110 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_110), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_111 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_111), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_112 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_112), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_113 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_11_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_11_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_113), "f"(wide_1[7]) : "memory");
                    }
                    int krow_12_1 = k_row0 + 48;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_12_1 / 64 * 8192 + t_col * 128 + krow_12_1 % 64 * 2 ^ (krow_12_1 / 64 * 8192 + t_col * 128 + krow_12_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_114 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_12_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_12_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_114), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_115 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_115), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_116 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_117 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_117), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_118 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_118), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_119 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_119), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_120 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_120), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_121 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_12_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_12_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_121), "f"(wide_1[7]) : "memory");
                    }
                    int krow_13_1 = k_row0 + 56;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"((kg_land_addr + ls * 16384 + (unsigned int)(krow_13_1 / 64 * 8192 + t_col * 128 + krow_13_1 % 64 * 2 ^ (krow_13_1 / 64 * 8192 + t_col * 128 + krow_13_1 % 64 * 2 >> 7 & 7) << 4))));
                    wide_1[0] = __uint_as_float(words_1[0] << 16);
                    wide_1[1] = __uint_as_float(words_1[0] & mask_hi);
                    wide_1[2] = __uint_as_float(words_1[1] << 16);
                    wide_1[3] = __uint_as_float(words_1[1] & mask_hi);
                    wide_1[4] = __uint_as_float(words_1[2] << 16);
                    wide_1[5] = __uint_as_float(words_1[2] & mask_hi);
                    wide_1[6] = __uint_as_float(words_1[3] << 16);
                    wide_1[7] = __uint_as_float(words_1[3] & mask_hi);
                    {
                        uint32_t _addr_122 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + krow_13_1 * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + krow_13_1 * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_122), "f"(wide_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_123 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 1) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 1) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_123), "f"(wide_1[1]) : "memory");
                    }
                    {
                        uint32_t _addr_124 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 2) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 2) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_124), "f"(wide_1[2]) : "memory");
                    }
                    {
                        uint32_t _addr_125 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 3) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 3) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_125), "f"(wide_1[3]) : "memory");
                    }
                    {
                        uint32_t _addr_126 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 4) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 4) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_126), "f"(wide_1[4]) : "memory");
                    }
                    {
                        uint32_t _addr_127 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 5) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 5) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_127), "f"(wide_1[5]) : "memory");
                    }
                    {
                        uint32_t _addr_128 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 6) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 6) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_128), "f"(wide_1[6]) : "memory");
                    }
                    {
                        uint32_t _addr_129 = static_cast<uint32_t>((kgf_addr + fs * 32768 + (unsigned int)(t_col / 32 * 16384 + (krow_13_1 + 7) * 128 + t_col % 32 * 4 ^ (t_col / 32 * 16384 + (krow_13_1 + 7) * 128 + t_col % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_129), "f"(wide_1[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(f32_full_addr + (fs) * 8);
                    mbarrier_arrive(tiles_empty_addr + (ls) * 8);
                    ls += 1;
                    if (ls == 2) { ls = 0; _phase_tiles_full ^= 1; }
                    fs += 1;
                    if (fs == 2) { fs = 0; _phase_f32_empty ^= 1; }
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
                for (int j_17 = 0; j_17 < 32; j_17++) {
                    _tmem_load_5[j_17] = _tmem_load_5[j_17] + _tmem_load_6[j_17];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 32; j_18++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)(j_18 * 128 + lane_v))) = _tmem_load_5[j_18];
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
                for (int j_19 = 0; j_19 < 32; j_19++) {
                    _tmem_load_7[j_19] = _tmem_load_7[j_19] + _tmem_load_8[j_19];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 32; j_20++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((32 + j_20) * 128 + lane_v))) = _tmem_load_7[j_20];
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
                for (int j_21 = 0; j_21 < 32; j_21++) {
                    _tmem_load_9[j_21] = _tmem_load_9[j_21] + _tmem_load_10[j_21];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 32; j_22++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((64 + j_22) * 128 + lane_v))) = _tmem_load_9[j_22];
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
                for (int j_23 = 0; j_23 < 32; j_23++) {
                    _tmem_load_11[j_23] = _tmem_load_11[j_23] + _tmem_load_12[j_23];
                }
                if (write_transposed == 1) {
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 32; j_24++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((96 + j_24) * 128 + lane_v))) = _tmem_load_11[j_24];
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
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow = row0_1 + c_1 * 64;
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)half) * 8192, (&w_tma), half * 64, head_1, trow, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)half) * 8192, (&kg_tma), half * 64, head_1, trow, tiles_full_addr + (stage) * 8);
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
                    :: "r"(tmem_b_hi), "r"(_smem_desc_lo_0), "r"(tmem_x_split), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_1 = (((kgf_addr) >> 4) & 0x7FFF) + (fs_1) * 2048;
#else
                    int _smem_desc_lo_1 = (((kgf_addr) >> 4) & 0x3FFF) + (fs_1) * 2048;
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
                    :: "r"((tmem_x_split + (128))), "r"(_smem_desc_lo_1), "r"(tmem_b_hi), "r"(0) : "memory");
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
