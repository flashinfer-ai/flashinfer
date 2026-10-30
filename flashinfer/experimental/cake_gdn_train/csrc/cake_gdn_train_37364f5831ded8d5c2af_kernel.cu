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
#define TMEM_ACC_QH_OFFSET 0
#define TMEM_ACC_AV_OFFSET 256
#define NUM_RING_STAGES 3
#define NUM_ACCS_STAGES 2
#define NUM_TK_PIPE_STAGES 4
#define SMEM_Q_S_OFF 1024
#define SMEM_Q_S_STAGE_BYTES 16384
#define SMEM_Q_S_STRIDE 73728
#define SMEM_H_S_OFF 17408
#define SMEM_H_S_STAGE_BYTES 32768
#define SMEM_H_S_STRIDE 73728
#define SMEM_AQK_P_OFF 50176
#define SMEM_AQK_P_STAGE_BYTES 8192
#define SMEM_AQK_P_STRIDE 73728
#define SMEM_VN_S_OFF 58368
#define SMEM_VN_S_STAGE_BYTES 16384
#define SMEM_VN_S_STRIDE 73728
#define SMEM_Q_K_OFF 1024
#define SMEM_Q_K_STAGE_BYTES 16384
#define SMEM_Q_K_STRIDE 73728
#define SMEM_H_MN_OFF 17408
#define SMEM_H_MN_STAGE_BYTES 32768
#define SMEM_H_MN_STRIDE 73728
#define SMEM_AQK_K_OFF 50176
#define SMEM_AQK_K_STAGE_BYTES 8192
#define SMEM_AQK_K_STRIDE 73728
#define SMEM_VN_MN_OFF 58368
#define SMEM_VN_MN_STAGE_BYTES 16384
#define SMEM_VN_MN_STRIDE 73728
#define SMEM_TK_S_OFF 222208
#define SMEM_TK_S_STAGE_BYTES 16
#define SMEM_TK_S_STRIDE 16
#define SMEM_TOTAL 222336
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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_37364f5831ded8d5c2af(const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap h_tma, const __grid_constant__ CUtensorMap aqk_tma, const __grid_constant__ CUtensorMap vn_tma, float* __restrict__ qscale, __nv_bfloat16* __restrict__ o_out, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, unsigned int* __restrict__ ticket, int num_seqs, int num_heads, int num_qk_heads)
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
    #define tiles_empty_addr (mbar_base + 24)
    #define acc_full_addr (mbar_base + 48)
    #define acc_empty_addr (mbar_base + 64)
    #define tk_full_addr (mbar_base + 80)
    #define tk_empty_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_s_addr = smem + 1024;
    __nv_bfloat16* h_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int h_s_addr = smem + 17408;
    __nv_bfloat16* aqk_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int aqk_p_addr = smem + 50176;
    __nv_bfloat16* vn_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int vn_s_addr = smem + 58368;
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_k_addr = smem + 1024;
    __nv_bfloat16* h_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int h_mn_addr = smem + 17408;
    __nv_bfloat16* aqk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int aqk_k_addr = smem + 50176;
    __nv_bfloat16* vn_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int vn_mn_addr = smem + 58368;
    unsigned int* tk_s = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int tk_s_addr = smem + 222208;
    int nc_k = seq_chunk_start[num_seqs];
    int num_tiles = nc_k * num_heads;
    int grid_k = num_bids;
    int tail_k = num_tiles;
    if (tail_k > grid_k * 4) {
        tail_k = grid_k * 4;
    }
    int big_k = (num_tiles - tail_k) / 8;
    int n_units = big_k + (num_tiles - big_k * 8);
    int dynamic = 0;
    if (n_units > grid_k) {
        dynamic = 1;
    }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // tiles_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'accs' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            mbarrier_init(smem + 72, 128);
            // --- pipeline 'tk_pipe' ---
            // tk_full: 4 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // tk_empty: 4 barriers, init_count=5
            mbarrier_init(smem + 112, 5);
            mbarrier_init(smem + 120, 5);
            mbarrier_init(smem + 128, 5);
            mbarrier_init(smem + 136, 5);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_qh = taddr;
    const int tmem_acc_av = taddr + 256;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_buf = 0;
            unsigned int ek_stage = 0;
            int wq = warp;
            int lane_half = lane >> 4;
            int t_row = wq * 16 + (lane & 15);
            int row_base = wq * 32 << 16;
            unsigned int _phase_tk_full = 0;
            unsigned int _phase_acc_full = 0;
            if (dynamic != 0) {
                while (1) {
                    mbarrier_wait(tk_full_addr + (ek_stage) * 8, _phase_tk_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int tv_e[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_e[0])) : "r"(tk_s_addr + ek_stage * 4) : "memory");
                    int w_e = (int)tv_e[0];
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tk_empty_addr + (ek_stage) * 8);
                    }
                    ek_stage += 1;
                    if (ek_stage == 4) { ek_stage = 0; _phase_tk_full ^= 1; }
                    if (w_e >= n_units) {
                        break;
                    }
                    int u_i = (int)w_e;
                    int start = u_i * 8;
                    int count = 8;
                    if (u_i >= big_k) {
                        start = big_k * 8 + (u_i - big_k);
                        count = 1;
                    }
                    #pragma unroll 1
                    for (int i_e = 0; i_e < count; i_e++) {
                        int t_i = (int)(start + i_e);
                        int chunk = nc_k - 1 - t_i / num_heads;
                        int head = t_i - t_i / num_heads * num_heads;
                        int ext0 = chunk_bos[chunk];
                        int clen = chunk_len[chunk];
                        long long obase = ((long long)(ext0 + t_row) * (long long)num_heads + (long long)head) * 128;
                        float qs = qscale[(long long)(chunk * 64 + t_row) * (long long)num_heads + (long long)head];
                        int buf_col = (int)epi_buf * 128;
                        mbarrier_wait(acc_full_addr + (epi_buf) * 8, _phase_acc_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int col_1_0 = lane_half * 16;
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + (unsigned int)row_base + (unsigned int)buf_col));
                        float _tmem_load_8_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[15]))
                            : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)buf_col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float ov_2_0[16];
                        #pragma unroll
                        for (int j_1_0 = 0; j_1_0 < 16; j_1_0++) {
                            ov_2_0[j_1_0] = __fmaf_rn(_tmem_load_0[j_1_0], qs, _tmem_load_8_0[j_1_0]);
                        }
                        if (t_row < clen) {
                            unsigned int _packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 0], ov_2_0[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 2], ov_2_0[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 4], ov_2_0[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 6], ov_2_0[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 8], ov_2_0[_lp*2+1 + 8]));
                                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 10], ov_2_0[_lp*2+1 + 10]));
                                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 12], ov_2_0[_lp*2+1 + 12]));
                                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_0[_lp*2 + 14], ov_2_0[_lp*2+1 + 14]));
                                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(o_out + (obase + (long long)col_1_0))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                            }
                        }
                        int col_1_1 = 32 + lane_half * 16;
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col + 32)));
                        float _tmem_load_8_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[15]))
                            : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(buf_col + 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float ov_2_1[16];
                        #pragma unroll
                        for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                            ov_2_1[j_1_1] = __fmaf_rn(_tmem_load_1[j_1_1], qs, _tmem_load_8_1[j_1_1]);
                        }
                        if (t_row < clen) {
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 0], ov_2_1[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 2], ov_2_1[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 4], ov_2_1[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 6], ov_2_1[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 8], ov_2_1[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 10], ov_2_1[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 12], ov_2_1[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_1[_lp*2 + 14], ov_2_1[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(o_out + (obase + (long long)col_1_1))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                            }
                        }
                        int col_1_2 = 64 + lane_half * 16;
                        float _tmem_load_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                            : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col + 64)));
                        float _tmem_load_8_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[15]))
                            : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(buf_col + 64)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float ov_2_2[16];
                        #pragma unroll
                        for (int j_1_2 = 0; j_1_2 < 16; j_1_2++) {
                            ov_2_2[j_1_2] = __fmaf_rn(_tmem_load_2[j_1_2], qs, _tmem_load_8_2[j_1_2]);
                        }
                        if (t_row < clen) {
                            unsigned int _packed_2[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 0], ov_2_2[_lp*2+1 + 0]));
                                _packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 2], ov_2_2[_lp*2+1 + 2]));
                                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 4], ov_2_2[_lp*2+1 + 4]));
                                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 6], ov_2_2[_lp*2+1 + 6]));
                                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 8], ov_2_2[_lp*2+1 + 8]));
                                _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 10], ov_2_2[_lp*2+1 + 10]));
                                _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 12], ov_2_2[_lp*2+1 + 12]));
                                _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_2[_lp*2 + 14], ov_2_2[_lp*2+1 + 14]));
                                _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(_packed_2);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(o_out + (obase + (long long)col_1_2))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                            }
                        }
                        int col_1_3 = 96 + lane_half * 16;
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col + 96)));
                        float _tmem_load_8_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[15]))
                            : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(buf_col + 96)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float ov_2_3[16];
                        #pragma unroll
                        for (int j_1_3 = 0; j_1_3 < 16; j_1_3++) {
                            ov_2_3[j_1_3] = __fmaf_rn(_tmem_load_3[j_1_3], qs, _tmem_load_8_3[j_1_3]);
                        }
                        if (t_row < clen) {
                            unsigned int _packed_3[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 0], ov_2_3[_lp*2+1 + 0]));
                                _packed_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 2], ov_2_3[_lp*2+1 + 2]));
                                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 4], ov_2_3[_lp*2+1 + 4]));
                                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 6], ov_2_3[_lp*2+1 + 6]));
                                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 8], ov_2_3[_lp*2+1 + 8]));
                                _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 10], ov_2_3[_lp*2+1 + 10]));
                                _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 12], ov_2_3[_lp*2+1 + 12]));
                                _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_3[_lp*2 + 14], ov_2_3[_lp*2+1 + 14]));
                                _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_3);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(o_out + (obase + (long long)col_1_3))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(acc_empty_addr + (epi_buf) * 8);
                        epi_buf += 1;
                        if (epi_buf == 2) { epi_buf = 0; _phase_acc_full ^= 1; }
                    }
                }
            } else {
                #pragma unroll 1
                for (unsigned int tile = bid; tile < num_tiles; tile += num_bids) {
                    int t_i_1 = (int)tile;
                    int chunk_1 = nc_k - 1 - t_i_1 / num_heads;
                    int head_1 = t_i_1 - t_i_1 / num_heads * num_heads;
                    int ext0_1 = chunk_bos[chunk_1];
                    int clen_1 = chunk_len[chunk_1];
                    long long obase_1 = ((long long)(ext0_1 + t_row) * (long long)num_heads + (long long)head_1) * 128;
                    float qs_1 = qscale[(long long)(chunk_1 * 64 + t_row) * (long long)num_heads + (long long)head_1];
                    int buf_col_1 = (int)epi_buf * 128;
                    mbarrier_wait(acc_full_addr + (epi_buf) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int col_1_4 = lane_half * 16;
                    float _tmem_load_4[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)buf_col_1));
                    float _tmem_load_1_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)buf_col_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_2_4[16];
                    #pragma unroll
                    for (int j_1_4 = 0; j_1_4 < 16; j_1_4++) {
                        ov_2_4[j_1_4] = __fmaf_rn(_tmem_load_4[j_1_4], qs_1, _tmem_load_1_0[j_1_4]);
                    }
                    if (t_row < clen_1) {
                        unsigned int _packed_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 0], ov_2_4[_lp*2+1 + 0]));
                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 2], ov_2_4[_lp*2+1 + 2]));
                            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 4], ov_2_4[_lp*2+1 + 4]));
                            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 6], ov_2_4[_lp*2+1 + 6]));
                            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 8], ov_2_4[_lp*2+1 + 8]));
                            _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 10], ov_2_4[_lp*2+1 + 10]));
                            _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 12], ov_2_4[_lp*2+1 + 12]));
                            _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_4[_lp*2 + 14], ov_2_4[_lp*2+1 + 14]));
                            _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (obase_1 + (long long)col_1_4))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                        }
                    }
                    int col_1_5 = 32 + lane_half * 16;
                    float _tmem_load_5[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col_1 + 32)));
                    float _tmem_load_1_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(buf_col_1 + 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_2_5[16];
                    #pragma unroll
                    for (int j_1_5 = 0; j_1_5 < 16; j_1_5++) {
                        ov_2_5[j_1_5] = __fmaf_rn(_tmem_load_5[j_1_5], qs_1, _tmem_load_1_1[j_1_5]);
                    }
                    if (t_row < clen_1) {
                        unsigned int _packed_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 0], ov_2_5[_lp*2+1 + 0]));
                            _packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 2], ov_2_5[_lp*2+1 + 2]));
                            _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 4], ov_2_5[_lp*2+1 + 4]));
                            _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 6], ov_2_5[_lp*2+1 + 6]));
                            _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 8], ov_2_5[_lp*2+1 + 8]));
                            _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 10], ov_2_5[_lp*2+1 + 10]));
                            _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 12], ov_2_5[_lp*2+1 + 12]));
                            _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_5[_lp*2 + 14], ov_2_5[_lp*2+1 + 14]));
                            _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(_packed_5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (obase_1 + (long long)col_1_5))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                        }
                    }
                    int col_1_6 = 64 + lane_half * 16;
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col_1 + 64)));
                    float _tmem_load_1_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_2[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(buf_col_1 + 64)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_2_6[16];
                    #pragma unroll
                    for (int j_1_6 = 0; j_1_6 < 16; j_1_6++) {
                        ov_2_6[j_1_6] = __fmaf_rn(_tmem_load_6[j_1_6], qs_1, _tmem_load_1_2[j_1_6]);
                    }
                    if (t_row < clen_1) {
                        unsigned int _packed_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 0], ov_2_6[_lp*2+1 + 0]));
                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 2], ov_2_6[_lp*2+1 + 2]));
                            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 4], ov_2_6[_lp*2+1 + 4]));
                            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 6], ov_2_6[_lp*2+1 + 6]));
                            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 8], ov_2_6[_lp*2+1 + 8]));
                            _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 10], ov_2_6[_lp*2+1 + 10]));
                            _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 12], ov_2_6[_lp*2+1 + 12]));
                            _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_6[_lp*2 + 14], ov_2_6[_lp*2+1 + 14]));
                            _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_6 = reinterpret_cast<const unsigned*>(_packed_6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (obase_1 + (long long)col_1_6))), "r"(_raw_stv8_6[0]), "r"(_raw_stv8_6[1]), "r"(_raw_stv8_6[2]), "r"(_raw_stv8_6[3]), "r"(_raw_stv8_6[4]), "r"(_raw_stv8_6[5]), "r"(_raw_stv8_6[6]), "r"(_raw_stv8_6[7]) : "memory");
                        }
                    }
                    int col_1_7 = 96 + lane_half * 16;
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col_1 + 96)));
                    float _tmem_load_1_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_3[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(buf_col_1 + 96)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_2_7[16];
                    #pragma unroll
                    for (int j_1_7 = 0; j_1_7 < 16; j_1_7++) {
                        ov_2_7[j_1_7] = __fmaf_rn(_tmem_load_7[j_1_7], qs_1, _tmem_load_1_3[j_1_7]);
                    }
                    if (t_row < clen_1) {
                        unsigned int _packed_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 0], ov_2_7[_lp*2+1 + 0]));
                            _packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 2], ov_2_7[_lp*2+1 + 2]));
                            _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 4], ov_2_7[_lp*2+1 + 4]));
                            _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 6], ov_2_7[_lp*2+1 + 6]));
                            _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 8], ov_2_7[_lp*2+1 + 8]));
                            _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 10], ov_2_7[_lp*2+1 + 10]));
                            _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 12], ov_2_7[_lp*2+1 + 12]));
                            _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2_7[_lp*2 + 14], ov_2_7[_lp*2+1 + 14]));
                            _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(_packed_7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (obase_1 + (long long)col_1_7))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(acc_empty_addr + (epi_buf) * 8);
                    epi_buf += 1;
                    if (epi_buf == 2) { epi_buf = 0; _phase_acc_full ^= 1; }
                }
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            unsigned int ld_stage = 0;
            unsigned int lk_stage = 0;
            int group_l = num_heads / num_qk_heads;
            unsigned int _phase_tk_empty = 1;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                if (dynamic != 0) {
                    unsigned int limit = (unsigned int)(n_units + num_bids - 1);
                    uint32_t drawn;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(drawn) : "l"(ticket), "r"(static_cast<uint32_t>(limit)) : "memory");
                    while (1) {
                        mbarrier_wait(tk_empty_addr + (lk_stage) * 8, _phase_tk_empty);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(tk_s_addr + lk_stage * 4), "r"(drawn) : "memory");
                        mbarrier_arrive(tk_full_addr + (lk_stage) * 8);
                        lk_stage += 1;
                        if (lk_stage == 4) { lk_stage = 0; _phase_tk_empty ^= 1; }
                        int w_l = (int)drawn;
                        if (w_l >= n_units) {
                            break;
                        }
                        uint32_t _atomic_fetch_inc_0;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_0) : "l"(ticket), "r"(static_cast<uint32_t>(limit)) : "memory");
                        drawn = _atomic_fetch_inc_0;
                        int u_i_1 = (int)w_l;
                        int start_1 = u_i_1 * 8;
                        int count_1 = 8;
                        if (u_i_1 >= big_k) {
                            start_1 = big_k * 8 + (u_i_1 - big_k);
                            count_1 = 1;
                        }
                        #pragma unroll 1
                        for (int i_l = 0; i_l < count_1; i_l++) {
                            int t_i_2 = (int)(start_1 + i_l);
                            int chunk_2 = nc_k - 1 - t_i_2 / num_heads;
                            int head_2 = t_i_2 - t_i_2 / num_heads * num_heads;
                            int qk_head = head_2 / group_l;
                            int row0 = chunk_2 * 64;
                            int hrow0 = (chunk_2 * num_heads + head_2) * 128;
                            int ext0_2 = chunk_bos[chunk_2];
                            mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                            mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 73728);
                            #pragma unroll
                            for (int seg = 0; seg < 2; seg++) {
                                int c0 = seg * 64;
                                tma_3d_gmem2smem(q_s_addr + ld_stage * 73728 + (unsigned int)(seg * 8192), (&q_tma), c0, qk_head, ext0_2, tiles_full_addr + (ld_stage) * 8);
                                tma_3d_gmem2smem(vn_s_addr + ld_stage * 73728 + (unsigned int)(seg * 8192), (&vn_tma), c0, head_2, row0, tiles_full_addr + (ld_stage) * 8);
                                tma_2d_gmem2smem(h_s_addr + ld_stage * 73728 + (unsigned int)(seg * 16384), (&h_tma), c0, hrow0, tiles_full_addr + (ld_stage) * 8);
                            }
                            tma_3d_gmem2smem(aqk_p_addr + ld_stage * 73728, (&aqk_tma), 0, head_2, row0, tiles_full_addr + (ld_stage) * 8);
                            ld_stage += 1;
                            if (ld_stage == 3) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                        }
                    }
                } else {
                    #pragma unroll 1
                    for (unsigned int tile_1 = bid; tile_1 < num_tiles; tile_1 += num_bids) {
                        int t_i_3 = (int)tile_1;
                        int chunk_3 = nc_k - 1 - t_i_3 / num_heads;
                        int head_3 = t_i_3 - t_i_3 / num_heads * num_heads;
                        int qk_head_1 = head_3 / group_l;
                        int row0_1 = chunk_3 * 64;
                        int hrow0_1 = (chunk_3 * num_heads + head_3) * 128;
                        int ext0_3 = chunk_bos[chunk_3];
                        mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 73728);
                        #pragma unroll
                        for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                            int c0_1 = seg_1 * 64;
                            tma_3d_gmem2smem(q_s_addr + ld_stage * 73728 + (unsigned int)(seg_1 * 8192), (&q_tma), c0_1, qk_head_1, ext0_3, tiles_full_addr + (ld_stage) * 8);
                            tma_3d_gmem2smem(vn_s_addr + ld_stage * 73728 + (unsigned int)(seg_1 * 8192), (&vn_tma), c0_1, head_3, row0_1, tiles_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(h_s_addr + ld_stage * 73728 + (unsigned int)(seg_1 * 16384), (&h_tma), c0_1, hrow0_1, tiles_full_addr + (ld_stage) * 8);
                        }
                        tma_3d_gmem2smem(aqk_p_addr + ld_stage * 73728, (&aqk_tma), 0, head_3, row0_1, tiles_full_addr + (ld_stage) * 8);
                        ld_stage += 1;
                        if (ld_stage == 3) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_buf = 0;
            unsigned int mk_stage = 0;
            unsigned int _phase_tk_full_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_acc_empty = 1;
            if (elect_sync()) {
                if (dynamic != 0) {
                    while (1) {
                        mbarrier_wait(tk_full_addr + (mk_stage) * 8, _phase_tk_full_1);
                        unsigned int tv_m[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_m[0])) : "r"(tk_s_addr + mk_stage * 4) : "memory");
                        int w_m = (int)tv_m[0];
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(tk_empty_addr + (mk_stage) * 8);
                        mk_stage += 1;
                        if (mk_stage == 4) { mk_stage = 0; _phase_tk_full_1 ^= 1; }
                        if (w_m >= n_units) {
                            break;
                        }
                        int u_i_2 = (int)w_m;
                        int start_2 = u_i_2 * 8;
                        int count_2 = 8;
                        if (u_i_2 >= big_k) {
                            start_2 = big_k * 8 + (u_i_2 - big_k);
                            count_2 = 1;
                        }
                        #pragma unroll 1
                        for (int _i_m = 0; _i_m < count_2; _i_m++) {
                            mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full);
                            mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_0 = (((q_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 4608;
                            int _smem_desc_lo_1 = ((((h_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 4608;
#else
                            int _smem_desc_lo_0 = (((q_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 4608;
                            int _smem_desc_lo_1 = ((((h_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 4608;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc_qh + (mma_buf * 128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_2 = (((aqk_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 4608;
                            int _smem_desc_lo_3 = ((((vn_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 4608;
#else
                            int _smem_desc_lo_2 = (((aqk_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 4608;
                            int _smem_desc_lo_3 = ((((vn_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 4608;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc_av + (mma_buf * 128))), "r"(0) : "memory");
                            tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                            tcgen05_commit(acc_full_addr + (mma_buf) * 8);
                            mma_stage += 1;
                            if (mma_stage == 3) { mma_stage = 0; _phase_tiles_full ^= 1; }
                            mma_buf += 1;
                            if (mma_buf == 2) { mma_buf = 0; _phase_acc_empty ^= 1; }
                        }
                    }
                } else {
                    #pragma unroll 1
                    for (unsigned int _tile = bid; _tile < num_tiles; _tile += num_bids) {
                        mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full);
                        mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_4 = (((q_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 4608;
                        int _smem_desc_lo_5 = ((((h_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 4608;
#else
                        int _smem_desc_lo_4 = (((q_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 4608;
                        int _smem_desc_lo_5 = ((((h_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 4608;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_acc_qh + (mma_buf * 128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((aqk_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 4608;
                        int _smem_desc_lo_7 = ((((vn_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 4608;
#else
                        int _smem_desc_lo_6 = (((aqk_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 4608;
                        int _smem_desc_lo_7 = ((((vn_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 4608;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_acc_av + (mma_buf * 128))), "r"(0) : "memory");
                        tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                        tcgen05_commit(acc_full_addr + (mma_buf) * 8);
                        mma_stage += 1;
                        if (mma_stage == 3) { mma_stage = 0; _phase_tiles_full ^= 1; }
                        mma_buf += 1;
                        if (mma_buf == 2) { mma_buf = 0; _phase_acc_empty ^= 1; }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
