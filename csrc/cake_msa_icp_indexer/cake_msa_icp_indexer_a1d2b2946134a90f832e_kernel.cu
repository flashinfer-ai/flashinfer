/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_msa_icp_indexer_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define NUM_K_PIPE_STAGES 6
#define NUM_ACC_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 17408
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_TOTAL 115712
#define THREADS 192
#define BLOCK_Q 4

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_msa_icp_indexer_a1d2b2946134a90f832e(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, int* __restrict__ block_table, float* __restrict__ score, uint8_t* __restrict__ valid, int* __restrict__ seq_lens, int* __restrict__ query_start_loc, long long* __restrict__ positions, uint8_t* __restrict__ active_rows, int total_tokens, int token_begin, int token_count, int request_begin, int score_stride, int block_table_stride, int split_count, int rank, int* __restrict__ forced_column, int* __restrict__ pool, int pool_stride, int* __restrict__ row_meta, int* __restrict__ status, int* __restrict__ gen_ctr, int x_region_stride, int x_par_stride)
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
    #define q_full_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define k_empty_addr (mbar_base + 56)
    #define acc_full_addr (mbar_base + 104)
    #define acc_empty_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_K_OFF);
    const int smem_k_addr = smem + SMEM_SMEM_K_OFF;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 6 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // k_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 4 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // acc_empty: 4 barriers, init_count=32
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 0) {
        int _tmem_hold = smem + 168;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epi ----
    if (warp <= 3) {
        { // epi_main
            int b = (int)blockIdx.x + request_begin;
            int split_id = (int)blockIdx.y;
            int seqlen = seq_lens[b];
            int tail = 0;
            if (seqlen % 256 > rank * 128) {
                tail = 1;
            }
            int n_req = seqlen / 256 + tail;
            int n_blk = 0;
            if (split_id < n_req) {
                n_blk = (n_req - split_id + split_count - 1) / split_count;
            }
            int n_tiles = n_blk;
            int w = (int)warp;
            int lane_0 = (int)lane;
            int u = lane_0 / 4;
            int h = lane_0 % 4;
            int qb = query_start_loc[b];
            int qe = query_start_loc[b + 1];
            int t = qb + u;
            int inb = 0;
            if (lane_0 < BLOCK_Q && t < qe && t < total_tokens && t >= token_begin && t < token_begin + token_count) {
                inb = 1;
            }
            int row_off = (t - token_begin) * 4 + h;
            if (split_id == 0 && w == 0 && inb != 0) {
                int lay0[8];
                int ns = ((n_req < split_count) ? n_req : split_count);
                int q = n_req / split_count;
                int r = n_req - q * split_count;
                int ta = q + 1;
                int tb = q + 1 - 1;
                int va = ((ta < 4) ? ta : 4);
                int vb = ((tb < 4) ? tb : 4);
                int ca = (ta + 4 - 1) / 4;
                int cb = (tb + 4 - 1) / 4;
                lay0[0] = r;
                lay0[1] = va;
                lay0[2] = vb;
                lay0[3] = ((ca < 16) ? ca : 16);
                lay0[4] = ((cb < 16) ? cb : 16);
                lay0[5] = r * va;
                lay0[6] = (ns - r) * vb;
                lay0[7] = lay0[3] * lay0[5] + lay0[4] * lay0[6];
                row_meta[row_off] = lay0[7];
            }
            if (n_tiles > 0 && warp < 4) {
                unsigned int lane_base = warp * 32 << 16;
                unsigned int slot_addr = taddr + lane_base + (unsigned int)(warp * 128);
                int live = 0;
                int upper = 0;
                long long p1 = 0;
                long long sl = (long long)seqlen;
                int act = 0;
                int fc = -1;
                if (inb != 0) {
                    p1 = positions[t] + 1;
                    fc = forced_column[t];
                    if ((int)active_rows[t] != 0) {
                        act = 1;
                    }
                }
                if (inb != 0 && act != 0) {
                    if (sl < p1) {
                        p1 = sl;
                    }
                    upper = (int)p1;
                    live = 1;
                }
                unsigned int phase = 0;
                int i0 = 0;
                int c = 0;
                int first_key = 0;
                int nvis = 0;
                float m = 0.0f;
                int off = 0;
                unsigned int top_b[16];
                int top_c[16];
                top_b[0] = 8388607;
                top_b[1] = 8388607;
                top_b[2] = 8388607;
                top_b[3] = 8388607;
                top_b[4] = 8388607;
                top_b[5] = 8388607;
                top_b[6] = 8388607;
                top_b[7] = 8388607;
                top_b[8] = 8388607;
                top_b[9] = 8388607;
                top_b[10] = 8388607;
                top_b[11] = 8388607;
                top_b[12] = 8388607;
                top_b[13] = 8388607;
                top_b[14] = 8388607;
                top_b[15] = 8388607;
                top_c[0] = 2147483647;
                top_c[1] = 2147483647;
                top_c[2] = 2147483647;
                top_c[3] = 2147483647;
                top_c[4] = 2147483647;
                top_c[5] = 2147483647;
                top_c[6] = 2147483647;
                top_c[7] = 2147483647;
                top_c[8] = 2147483647;
                top_c[9] = 2147483647;
                top_c[10] = 2147483647;
                top_c[11] = 2147483647;
                top_c[12] = 2147483647;
                top_c[13] = 2147483647;
                top_c[14] = 2147483647;
                top_c[15] = 2147483647;
                unsigned int buf_b[8];
                int buf_c[8];
                buf_b[0] = 8388607;
                buf_b[1] = 8388607;
                buf_b[2] = 8388607;
                buf_b[3] = 8388607;
                buf_b[4] = 8388607;
                buf_b[5] = 8388607;
                buf_b[6] = 8388607;
                buf_b[7] = 8388607;
                buf_c[0] = 2147483647;
                buf_c[1] = 2147483647;
                buf_c[2] = 2147483647;
                buf_c[3] = 2147483647;
                buf_c[4] = 2147483647;
                buf_c[5] = 2147483647;
                buf_c[6] = 2147483647;
                buf_c[7] = 2147483647;
                int cnt = 0;
                unsigned int cur_b = 0;
                int cur_c = 2147483647;
                unsigned int mb = 0;
                int need = 0;
                #pragma unroll 1
                for (int tile = w; tile < n_tiles; tile += 4) {
                    mbarrier_wait(acc_full_addr + (w) * 8, phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    i0 = tile;
                    if (n_blk > i0) {
                        float _tmem_load_0[128];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63]), "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                            : "r"(slot_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        c = split_id + i0 * split_count;
                        first_key = c * 256 + rank * 128;
                        nvis = upper - first_key;
                        if (nvis > 128) {
                            nvis = 128;
                        }
                        if (nvis < 0) {
                            nvis = 0;
                        }
                        if (nvis < 128) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_0 = nvis;
                                if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                                else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                                }
                            }
                            if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 16))) _tmem_load_0[16] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 17))) _tmem_load_0[17] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 18))) _tmem_load_0[18] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 19))) _tmem_load_0[19] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 20))) _tmem_load_0[20] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 21))) _tmem_load_0[21] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 22))) _tmem_load_0[22] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 23))) _tmem_load_0[23] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 24))) _tmem_load_0[24] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 25))) _tmem_load_0[25] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 26))) _tmem_load_0[26] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 27))) _tmem_load_0[27] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 28))) _tmem_load_0[28] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 29))) _tmem_load_0[29] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 30))) _tmem_load_0[30] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 31))) _tmem_load_0[31] = -CAKE_INF;
                            uint32_t _slice_lo_mask_1;
                            {
                                int _lim_1 = nvis - 32;
                                if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                                }
                            }
                            if (!(_slice_lo_mask_1 & (1u << 0))) _tmem_load_0[32] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 1))) _tmem_load_0[33] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 2))) _tmem_load_0[34] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 3))) _tmem_load_0[35] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 4))) _tmem_load_0[36] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 5))) _tmem_load_0[37] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 6))) _tmem_load_0[38] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 7))) _tmem_load_0[39] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 8))) _tmem_load_0[40] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 9))) _tmem_load_0[41] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 10))) _tmem_load_0[42] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 11))) _tmem_load_0[43] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 12))) _tmem_load_0[44] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 13))) _tmem_load_0[45] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 14))) _tmem_load_0[46] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 15))) _tmem_load_0[47] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 16))) _tmem_load_0[48] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 17))) _tmem_load_0[49] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 18))) _tmem_load_0[50] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 19))) _tmem_load_0[51] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 20))) _tmem_load_0[52] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 21))) _tmem_load_0[53] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 22))) _tmem_load_0[54] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 23))) _tmem_load_0[55] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 24))) _tmem_load_0[56] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 25))) _tmem_load_0[57] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 26))) _tmem_load_0[58] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 27))) _tmem_load_0[59] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 28))) _tmem_load_0[60] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 29))) _tmem_load_0[61] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 30))) _tmem_load_0[62] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 31))) _tmem_load_0[63] = -CAKE_INF;
                            uint32_t _slice_lo_mask_2;
                            {
                                int _lim_2 = nvis - 64;
                                if (_lim_2 <= 0) { _slice_lo_mask_2 = 0u; }
                                else if (_lim_2 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_2));
                                }
                            }
                            if (!(_slice_lo_mask_2 & (1u << 0))) _tmem_load_0[64] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 1))) _tmem_load_0[65] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 2))) _tmem_load_0[66] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 3))) _tmem_load_0[67] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 4))) _tmem_load_0[68] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 5))) _tmem_load_0[69] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 6))) _tmem_load_0[70] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 7))) _tmem_load_0[71] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 8))) _tmem_load_0[72] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 9))) _tmem_load_0[73] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 10))) _tmem_load_0[74] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 11))) _tmem_load_0[75] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 12))) _tmem_load_0[76] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 13))) _tmem_load_0[77] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 14))) _tmem_load_0[78] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 15))) _tmem_load_0[79] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 16))) _tmem_load_0[80] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 17))) _tmem_load_0[81] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 18))) _tmem_load_0[82] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 19))) _tmem_load_0[83] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 20))) _tmem_load_0[84] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 21))) _tmem_load_0[85] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 22))) _tmem_load_0[86] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 23))) _tmem_load_0[87] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 24))) _tmem_load_0[88] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 25))) _tmem_load_0[89] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 26))) _tmem_load_0[90] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 27))) _tmem_load_0[91] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 28))) _tmem_load_0[92] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 29))) _tmem_load_0[93] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 30))) _tmem_load_0[94] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 31))) _tmem_load_0[95] = -CAKE_INF;
                            uint32_t _slice_lo_mask_3;
                            {
                                int _lim_3 = nvis - 96;
                                if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                                else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
                                }
                            }
                            if (!(_slice_lo_mask_3 & (1u << 0))) _tmem_load_0[96] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 1))) _tmem_load_0[97] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 2))) _tmem_load_0[98] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 3))) _tmem_load_0[99] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 4))) _tmem_load_0[100] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 5))) _tmem_load_0[101] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 6))) _tmem_load_0[102] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 7))) _tmem_load_0[103] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 8))) _tmem_load_0[104] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 9))) _tmem_load_0[105] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 10))) _tmem_load_0[106] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 11))) _tmem_load_0[107] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 12))) _tmem_load_0[108] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 13))) _tmem_load_0[109] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 14))) _tmem_load_0[110] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 15))) _tmem_load_0[111] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 16))) _tmem_load_0[112] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 17))) _tmem_load_0[113] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 18))) _tmem_load_0[114] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 19))) _tmem_load_0[115] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 20))) _tmem_load_0[116] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 21))) _tmem_load_0[117] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 22))) _tmem_load_0[118] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 23))) _tmem_load_0[119] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 24))) _tmem_load_0[120] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 25))) _tmem_load_0[121] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 26))) _tmem_load_0[122] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 27))) _tmem_load_0[123] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 28))) _tmem_load_0[124] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 29))) _tmem_load_0[125] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 30))) _tmem_load_0[126] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 31))) _tmem_load_0[127] = -CAKE_INF;
                        }
                        float _reg_reduce_max_nan_4_0_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_0) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]));
                        float _reg_reduce_max_nan_4_0_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_1) : "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]));
                        float _reg_reduce_max_nan_4_0_2;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_2) : "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]));
                        float _reg_reduce_max_nan_4_0_3;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_3) : "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]));
                        float _reg_reduce_max_nan_4_0_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_4) : "f"(_tmem_load_0[8]), "f"(_tmem_load_0[9]));
                        float _reg_reduce_max_nan_4_0_5;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_5) : "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]));
                        float _reg_reduce_max_nan_4_0_6;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_6) : "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]));
                        float _reg_reduce_max_nan_4_0_7;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_7) : "f"(_tmem_load_0[14]), "f"(_tmem_load_0[15]));
                        float _reg_reduce_max_nan_4_0_8;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_8) : "f"(_tmem_load_0[16]), "f"(_tmem_load_0[17]));
                        float _reg_reduce_max_nan_4_0_9;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_9) : "f"(_tmem_load_0[18]), "f"(_tmem_load_0[19]));
                        float _reg_reduce_max_nan_4_0_10;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_10) : "f"(_tmem_load_0[20]), "f"(_tmem_load_0[21]));
                        float _reg_reduce_max_nan_4_0_11;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_11) : "f"(_tmem_load_0[22]), "f"(_tmem_load_0[23]));
                        float _reg_reduce_max_nan_4_0_12;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_12) : "f"(_tmem_load_0[24]), "f"(_tmem_load_0[25]));
                        float _reg_reduce_max_nan_4_0_13;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_13) : "f"(_tmem_load_0[26]), "f"(_tmem_load_0[27]));
                        float _reg_reduce_max_nan_4_0_14;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_14) : "f"(_tmem_load_0[28]), "f"(_tmem_load_0[29]));
                        float _reg_reduce_max_nan_4_0_15;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_15) : "f"(_tmem_load_0[30]), "f"(_tmem_load_0[31]));
                        float _reg_reduce_max_nan_4_0_16;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_16) : "f"(_tmem_load_0[32]), "f"(_tmem_load_0[33]));
                        float _reg_reduce_max_nan_4_0_17;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_17) : "f"(_tmem_load_0[34]), "f"(_tmem_load_0[35]));
                        float _reg_reduce_max_nan_4_0_18;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_18) : "f"(_tmem_load_0[36]), "f"(_tmem_load_0[37]));
                        float _reg_reduce_max_nan_4_0_19;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_19) : "f"(_tmem_load_0[38]), "f"(_tmem_load_0[39]));
                        float _reg_reduce_max_nan_4_0_20;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_20) : "f"(_tmem_load_0[40]), "f"(_tmem_load_0[41]));
                        float _reg_reduce_max_nan_4_0_21;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_21) : "f"(_tmem_load_0[42]), "f"(_tmem_load_0[43]));
                        float _reg_reduce_max_nan_4_0_22;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_22) : "f"(_tmem_load_0[44]), "f"(_tmem_load_0[45]));
                        float _reg_reduce_max_nan_4_0_23;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_23) : "f"(_tmem_load_0[46]), "f"(_tmem_load_0[47]));
                        float _reg_reduce_max_nan_4_0_24;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_24) : "f"(_tmem_load_0[48]), "f"(_tmem_load_0[49]));
                        float _reg_reduce_max_nan_4_0_25;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_25) : "f"(_tmem_load_0[50]), "f"(_tmem_load_0[51]));
                        float _reg_reduce_max_nan_4_0_26;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_26) : "f"(_tmem_load_0[52]), "f"(_tmem_load_0[53]));
                        float _reg_reduce_max_nan_4_0_27;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_27) : "f"(_tmem_load_0[54]), "f"(_tmem_load_0[55]));
                        float _reg_reduce_max_nan_4_0_28;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_28) : "f"(_tmem_load_0[56]), "f"(_tmem_load_0[57]));
                        float _reg_reduce_max_nan_4_0_29;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_29) : "f"(_tmem_load_0[58]), "f"(_tmem_load_0[59]));
                        float _reg_reduce_max_nan_4_0_30;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_30) : "f"(_tmem_load_0[60]), "f"(_tmem_load_0[61]));
                        float _reg_reduce_max_nan_4_0_31;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_31) : "f"(_tmem_load_0[62]), "f"(_tmem_load_0[63]));
                        float _reg_reduce_max_nan_4_0_32;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_32) : "f"(_tmem_load_0[64]), "f"(_tmem_load_0[65]));
                        float _reg_reduce_max_nan_4_0_33;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_33) : "f"(_tmem_load_0[66]), "f"(_tmem_load_0[67]));
                        float _reg_reduce_max_nan_4_0_34;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_34) : "f"(_tmem_load_0[68]), "f"(_tmem_load_0[69]));
                        float _reg_reduce_max_nan_4_0_35;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_35) : "f"(_tmem_load_0[70]), "f"(_tmem_load_0[71]));
                        float _reg_reduce_max_nan_4_0_36;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_36) : "f"(_tmem_load_0[72]), "f"(_tmem_load_0[73]));
                        float _reg_reduce_max_nan_4_0_37;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_37) : "f"(_tmem_load_0[74]), "f"(_tmem_load_0[75]));
                        float _reg_reduce_max_nan_4_0_38;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_38) : "f"(_tmem_load_0[76]), "f"(_tmem_load_0[77]));
                        float _reg_reduce_max_nan_4_0_39;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_39) : "f"(_tmem_load_0[78]), "f"(_tmem_load_0[79]));
                        float _reg_reduce_max_nan_4_0_40;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_40) : "f"(_tmem_load_0[80]), "f"(_tmem_load_0[81]));
                        float _reg_reduce_max_nan_4_0_41;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_41) : "f"(_tmem_load_0[82]), "f"(_tmem_load_0[83]));
                        float _reg_reduce_max_nan_4_0_42;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_42) : "f"(_tmem_load_0[84]), "f"(_tmem_load_0[85]));
                        float _reg_reduce_max_nan_4_0_43;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_43) : "f"(_tmem_load_0[86]), "f"(_tmem_load_0[87]));
                        float _reg_reduce_max_nan_4_0_44;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_44) : "f"(_tmem_load_0[88]), "f"(_tmem_load_0[89]));
                        float _reg_reduce_max_nan_4_0_45;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_45) : "f"(_tmem_load_0[90]), "f"(_tmem_load_0[91]));
                        float _reg_reduce_max_nan_4_0_46;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_46) : "f"(_tmem_load_0[92]), "f"(_tmem_load_0[93]));
                        float _reg_reduce_max_nan_4_0_47;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_47) : "f"(_tmem_load_0[94]), "f"(_tmem_load_0[95]));
                        float _reg_reduce_max_nan_4_0_48;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_48) : "f"(_tmem_load_0[96]), "f"(_tmem_load_0[97]));
                        float _reg_reduce_max_nan_4_0_49;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_49) : "f"(_tmem_load_0[98]), "f"(_tmem_load_0[99]));
                        float _reg_reduce_max_nan_4_0_50;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_50) : "f"(_tmem_load_0[100]), "f"(_tmem_load_0[101]));
                        float _reg_reduce_max_nan_4_0_51;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_51) : "f"(_tmem_load_0[102]), "f"(_tmem_load_0[103]));
                        float _reg_reduce_max_nan_4_0_52;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_52) : "f"(_tmem_load_0[104]), "f"(_tmem_load_0[105]));
                        float _reg_reduce_max_nan_4_0_53;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_53) : "f"(_tmem_load_0[106]), "f"(_tmem_load_0[107]));
                        float _reg_reduce_max_nan_4_0_54;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_54) : "f"(_tmem_load_0[108]), "f"(_tmem_load_0[109]));
                        float _reg_reduce_max_nan_4_0_55;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_55) : "f"(_tmem_load_0[110]), "f"(_tmem_load_0[111]));
                        float _reg_reduce_max_nan_4_0_56;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_56) : "f"(_tmem_load_0[112]), "f"(_tmem_load_0[113]));
                        float _reg_reduce_max_nan_4_0_57;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_57) : "f"(_tmem_load_0[114]), "f"(_tmem_load_0[115]));
                        float _reg_reduce_max_nan_4_0_58;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_58) : "f"(_tmem_load_0[116]), "f"(_tmem_load_0[117]));
                        float _reg_reduce_max_nan_4_0_59;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_59) : "f"(_tmem_load_0[118]), "f"(_tmem_load_0[119]));
                        float _reg_reduce_max_nan_4_0_60;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_60) : "f"(_tmem_load_0[120]), "f"(_tmem_load_0[121]));
                        float _reg_reduce_max_nan_4_0_61;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_61) : "f"(_tmem_load_0[122]), "f"(_tmem_load_0[123]));
                        float _reg_reduce_max_nan_4_0_62;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_62) : "f"(_tmem_load_0[124]), "f"(_tmem_load_0[125]));
                        float _reg_reduce_max_nan_4_0_63;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_0_63) : "f"(_tmem_load_0[126]), "f"(_tmem_load_0[127]));
                        float _reg_reduce_max_nan_4_1_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_0) : "f"(_reg_reduce_max_nan_4_0_0), "f"(_reg_reduce_max_nan_4_0_1));
                        float _reg_reduce_max_nan_4_1_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_1) : "f"(_reg_reduce_max_nan_4_0_2), "f"(_reg_reduce_max_nan_4_0_3));
                        float _reg_reduce_max_nan_4_1_2;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_2) : "f"(_reg_reduce_max_nan_4_0_4), "f"(_reg_reduce_max_nan_4_0_5));
                        float _reg_reduce_max_nan_4_1_3;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_3) : "f"(_reg_reduce_max_nan_4_0_6), "f"(_reg_reduce_max_nan_4_0_7));
                        float _reg_reduce_max_nan_4_1_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_4) : "f"(_reg_reduce_max_nan_4_0_8), "f"(_reg_reduce_max_nan_4_0_9));
                        float _reg_reduce_max_nan_4_1_5;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_5) : "f"(_reg_reduce_max_nan_4_0_10), "f"(_reg_reduce_max_nan_4_0_11));
                        float _reg_reduce_max_nan_4_1_6;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_6) : "f"(_reg_reduce_max_nan_4_0_12), "f"(_reg_reduce_max_nan_4_0_13));
                        float _reg_reduce_max_nan_4_1_7;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_7) : "f"(_reg_reduce_max_nan_4_0_14), "f"(_reg_reduce_max_nan_4_0_15));
                        float _reg_reduce_max_nan_4_1_8;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_8) : "f"(_reg_reduce_max_nan_4_0_16), "f"(_reg_reduce_max_nan_4_0_17));
                        float _reg_reduce_max_nan_4_1_9;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_9) : "f"(_reg_reduce_max_nan_4_0_18), "f"(_reg_reduce_max_nan_4_0_19));
                        float _reg_reduce_max_nan_4_1_10;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_10) : "f"(_reg_reduce_max_nan_4_0_20), "f"(_reg_reduce_max_nan_4_0_21));
                        float _reg_reduce_max_nan_4_1_11;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_11) : "f"(_reg_reduce_max_nan_4_0_22), "f"(_reg_reduce_max_nan_4_0_23));
                        float _reg_reduce_max_nan_4_1_12;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_12) : "f"(_reg_reduce_max_nan_4_0_24), "f"(_reg_reduce_max_nan_4_0_25));
                        float _reg_reduce_max_nan_4_1_13;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_13) : "f"(_reg_reduce_max_nan_4_0_26), "f"(_reg_reduce_max_nan_4_0_27));
                        float _reg_reduce_max_nan_4_1_14;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_14) : "f"(_reg_reduce_max_nan_4_0_28), "f"(_reg_reduce_max_nan_4_0_29));
                        float _reg_reduce_max_nan_4_1_15;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_15) : "f"(_reg_reduce_max_nan_4_0_30), "f"(_reg_reduce_max_nan_4_0_31));
                        float _reg_reduce_max_nan_4_1_16;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_16) : "f"(_reg_reduce_max_nan_4_0_32), "f"(_reg_reduce_max_nan_4_0_33));
                        float _reg_reduce_max_nan_4_1_17;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_17) : "f"(_reg_reduce_max_nan_4_0_34), "f"(_reg_reduce_max_nan_4_0_35));
                        float _reg_reduce_max_nan_4_1_18;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_18) : "f"(_reg_reduce_max_nan_4_0_36), "f"(_reg_reduce_max_nan_4_0_37));
                        float _reg_reduce_max_nan_4_1_19;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_19) : "f"(_reg_reduce_max_nan_4_0_38), "f"(_reg_reduce_max_nan_4_0_39));
                        float _reg_reduce_max_nan_4_1_20;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_20) : "f"(_reg_reduce_max_nan_4_0_40), "f"(_reg_reduce_max_nan_4_0_41));
                        float _reg_reduce_max_nan_4_1_21;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_21) : "f"(_reg_reduce_max_nan_4_0_42), "f"(_reg_reduce_max_nan_4_0_43));
                        float _reg_reduce_max_nan_4_1_22;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_22) : "f"(_reg_reduce_max_nan_4_0_44), "f"(_reg_reduce_max_nan_4_0_45));
                        float _reg_reduce_max_nan_4_1_23;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_23) : "f"(_reg_reduce_max_nan_4_0_46), "f"(_reg_reduce_max_nan_4_0_47));
                        float _reg_reduce_max_nan_4_1_24;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_24) : "f"(_reg_reduce_max_nan_4_0_48), "f"(_reg_reduce_max_nan_4_0_49));
                        float _reg_reduce_max_nan_4_1_25;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_25) : "f"(_reg_reduce_max_nan_4_0_50), "f"(_reg_reduce_max_nan_4_0_51));
                        float _reg_reduce_max_nan_4_1_26;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_26) : "f"(_reg_reduce_max_nan_4_0_52), "f"(_reg_reduce_max_nan_4_0_53));
                        float _reg_reduce_max_nan_4_1_27;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_27) : "f"(_reg_reduce_max_nan_4_0_54), "f"(_reg_reduce_max_nan_4_0_55));
                        float _reg_reduce_max_nan_4_1_28;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_28) : "f"(_reg_reduce_max_nan_4_0_56), "f"(_reg_reduce_max_nan_4_0_57));
                        float _reg_reduce_max_nan_4_1_29;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_29) : "f"(_reg_reduce_max_nan_4_0_58), "f"(_reg_reduce_max_nan_4_0_59));
                        float _reg_reduce_max_nan_4_1_30;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_30) : "f"(_reg_reduce_max_nan_4_0_60), "f"(_reg_reduce_max_nan_4_0_61));
                        float _reg_reduce_max_nan_4_1_31;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_1_31) : "f"(_reg_reduce_max_nan_4_0_62), "f"(_reg_reduce_max_nan_4_0_63));
                        float _reg_reduce_max_nan_4_2_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_0) : "f"(_reg_reduce_max_nan_4_1_0), "f"(_reg_reduce_max_nan_4_1_1));
                        float _reg_reduce_max_nan_4_2_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_1) : "f"(_reg_reduce_max_nan_4_1_2), "f"(_reg_reduce_max_nan_4_1_3));
                        float _reg_reduce_max_nan_4_2_2;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_2) : "f"(_reg_reduce_max_nan_4_1_4), "f"(_reg_reduce_max_nan_4_1_5));
                        float _reg_reduce_max_nan_4_2_3;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_3) : "f"(_reg_reduce_max_nan_4_1_6), "f"(_reg_reduce_max_nan_4_1_7));
                        float _reg_reduce_max_nan_4_2_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_4) : "f"(_reg_reduce_max_nan_4_1_8), "f"(_reg_reduce_max_nan_4_1_9));
                        float _reg_reduce_max_nan_4_2_5;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_5) : "f"(_reg_reduce_max_nan_4_1_10), "f"(_reg_reduce_max_nan_4_1_11));
                        float _reg_reduce_max_nan_4_2_6;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_6) : "f"(_reg_reduce_max_nan_4_1_12), "f"(_reg_reduce_max_nan_4_1_13));
                        float _reg_reduce_max_nan_4_2_7;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_7) : "f"(_reg_reduce_max_nan_4_1_14), "f"(_reg_reduce_max_nan_4_1_15));
                        float _reg_reduce_max_nan_4_2_8;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_8) : "f"(_reg_reduce_max_nan_4_1_16), "f"(_reg_reduce_max_nan_4_1_17));
                        float _reg_reduce_max_nan_4_2_9;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_9) : "f"(_reg_reduce_max_nan_4_1_18), "f"(_reg_reduce_max_nan_4_1_19));
                        float _reg_reduce_max_nan_4_2_10;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_10) : "f"(_reg_reduce_max_nan_4_1_20), "f"(_reg_reduce_max_nan_4_1_21));
                        float _reg_reduce_max_nan_4_2_11;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_11) : "f"(_reg_reduce_max_nan_4_1_22), "f"(_reg_reduce_max_nan_4_1_23));
                        float _reg_reduce_max_nan_4_2_12;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_12) : "f"(_reg_reduce_max_nan_4_1_24), "f"(_reg_reduce_max_nan_4_1_25));
                        float _reg_reduce_max_nan_4_2_13;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_13) : "f"(_reg_reduce_max_nan_4_1_26), "f"(_reg_reduce_max_nan_4_1_27));
                        float _reg_reduce_max_nan_4_2_14;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_14) : "f"(_reg_reduce_max_nan_4_1_28), "f"(_reg_reduce_max_nan_4_1_29));
                        float _reg_reduce_max_nan_4_2_15;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_2_15) : "f"(_reg_reduce_max_nan_4_1_30), "f"(_reg_reduce_max_nan_4_1_31));
                        float _reg_reduce_max_nan_4_3_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_0) : "f"(_reg_reduce_max_nan_4_2_0), "f"(_reg_reduce_max_nan_4_2_1));
                        float _reg_reduce_max_nan_4_3_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_1) : "f"(_reg_reduce_max_nan_4_2_2), "f"(_reg_reduce_max_nan_4_2_3));
                        float _reg_reduce_max_nan_4_3_2;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_2) : "f"(_reg_reduce_max_nan_4_2_4), "f"(_reg_reduce_max_nan_4_2_5));
                        float _reg_reduce_max_nan_4_3_3;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_3) : "f"(_reg_reduce_max_nan_4_2_6), "f"(_reg_reduce_max_nan_4_2_7));
                        float _reg_reduce_max_nan_4_3_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_4) : "f"(_reg_reduce_max_nan_4_2_8), "f"(_reg_reduce_max_nan_4_2_9));
                        float _reg_reduce_max_nan_4_3_5;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_5) : "f"(_reg_reduce_max_nan_4_2_10), "f"(_reg_reduce_max_nan_4_2_11));
                        float _reg_reduce_max_nan_4_3_6;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_6) : "f"(_reg_reduce_max_nan_4_2_12), "f"(_reg_reduce_max_nan_4_2_13));
                        float _reg_reduce_max_nan_4_3_7;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_3_7) : "f"(_reg_reduce_max_nan_4_2_14), "f"(_reg_reduce_max_nan_4_2_15));
                        float _reg_reduce_max_nan_4_4_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_4_0) : "f"(_reg_reduce_max_nan_4_3_0), "f"(_reg_reduce_max_nan_4_3_1));
                        float _reg_reduce_max_nan_4_4_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_4_1) : "f"(_reg_reduce_max_nan_4_3_2), "f"(_reg_reduce_max_nan_4_3_3));
                        float _reg_reduce_max_nan_4_4_2;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_4_2) : "f"(_reg_reduce_max_nan_4_3_4), "f"(_reg_reduce_max_nan_4_3_5));
                        float _reg_reduce_max_nan_4_4_3;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_4_3) : "f"(_reg_reduce_max_nan_4_3_6), "f"(_reg_reduce_max_nan_4_3_7));
                        float _reg_reduce_max_nan_4_5_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_5_0) : "f"(_reg_reduce_max_nan_4_4_0), "f"(_reg_reduce_max_nan_4_4_1));
                        float _reg_reduce_max_nan_4_5_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_5_1) : "f"(_reg_reduce_max_nan_4_4_2), "f"(_reg_reduce_max_nan_4_4_3));
                        float _reg_reduce_max_nan_4_6_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_reg_reduce_max_nan_4_6_0) : "f"(_reg_reduce_max_nan_4_5_0), "f"(_reg_reduce_max_nan_4_5_1));
                        float _tmem_load_0_max = _reg_reduce_max_nan_4_6_0;
                        m = _tmem_load_0_max;
                        need = 0;
                        cur_b = 8388607;
                        cur_c = 2147483647;
                        if (live != 0 && nvis > 0 && c != fc) {
                            mb = reinterpret_cast<unsigned int*>(&m)[0];
                            if ((mb & 2147483647) > 2139095040) {
                                status[0] = 1;
                            }
                            if ((mb & 2147483647) == 0) {
                                mb = 0;
                            }
                            cur_b = (((mb & 2147483648u) != 0) ? ~mb : mb ^ 2147483648u);
                            cur_c = c * 2 + rank;
                            if (cur_b > top_b[15]) {
                                need = 1;
                            }
                            if (cur_b == top_b[15] && cur_c < top_c[15]) {
                                need = 1;
                            }
                        }
                        int _vote_0 = __any_sync(0xFFFFFFFF, need != 0);
                        if (_vote_0 != 0) {
                            buf_b[0] = ((need != 0 && cnt == 0) ? cur_b : buf_b[0]);
                            buf_c[0] = ((need != 0 && cnt == 0) ? cur_c : buf_c[0]);
                            buf_b[1] = ((need != 0 && cnt == 1) ? cur_b : buf_b[1]);
                            buf_c[1] = ((need != 0 && cnt == 1) ? cur_c : buf_c[1]);
                            buf_b[2] = ((need != 0 && cnt == 2) ? cur_b : buf_b[2]);
                            buf_c[2] = ((need != 0 && cnt == 2) ? cur_c : buf_c[2]);
                            buf_b[3] = ((need != 0 && cnt == 3) ? cur_b : buf_b[3]);
                            buf_c[3] = ((need != 0 && cnt == 3) ? cur_c : buf_c[3]);
                            buf_b[4] = ((need != 0 && cnt == 4) ? cur_b : buf_b[4]);
                            buf_c[4] = ((need != 0 && cnt == 4) ? cur_c : buf_c[4]);
                            buf_b[5] = ((need != 0 && cnt == 5) ? cur_b : buf_b[5]);
                            buf_c[5] = ((need != 0 && cnt == 5) ? cur_c : buf_c[5]);
                            buf_b[6] = ((need != 0 && cnt == 6) ? cur_b : buf_b[6]);
                            buf_c[6] = ((need != 0 && cnt == 6) ? cur_c : buf_c[6]);
                            buf_b[7] = ((need != 0 && cnt == 7) ? cur_b : buf_b[7]);
                            buf_c[7] = ((need != 0 && cnt == 7) ? cur_c : buf_c[7]);
                            cnt = cnt + need;
                            int _vote_1 = __any_sync(0xFFFFFFFF, cnt == 8);
                            if (_vote_1 != 0) {
                                unsigned int xk = buf_b[0];
                                int xi = buf_c[0];
                                unsigned int yk = buf_b[1];
                                int yi = buf_c[1];
                                int take = (int)(yk > xk || yk == xk && yi < xi);
                                buf_b[0] = ((take != 0) ? yk : xk);
                                buf_c[0] = ((take != 0) ? yi : xi);
                                buf_b[1] = ((take != 0) ? xk : yk);
                                buf_c[1] = ((take != 0) ? xi : yi);
                                unsigned int xk_0 = buf_b[2];
                                int xi_1 = buf_c[2];
                                unsigned int yk_2 = buf_b[3];
                                int yi_3 = buf_c[3];
                                int take_4 = (int)(yk_2 > xk_0 || yk_2 == xk_0 && yi_3 < xi_1);
                                buf_b[2] = ((take_4 != 0) ? yk_2 : xk_0);
                                buf_c[2] = ((take_4 != 0) ? yi_3 : xi_1);
                                buf_b[3] = ((take_4 != 0) ? xk_0 : yk_2);
                                buf_c[3] = ((take_4 != 0) ? xi_1 : yi_3);
                                unsigned int xk_5 = buf_b[4];
                                int xi_6 = buf_c[4];
                                unsigned int yk_7 = buf_b[5];
                                int yi_8 = buf_c[5];
                                int take_9 = (int)(yk_7 > xk_5 || yk_7 == xk_5 && yi_8 < xi_6);
                                buf_b[4] = ((take_9 != 0) ? yk_7 : xk_5);
                                buf_c[4] = ((take_9 != 0) ? yi_8 : xi_6);
                                buf_b[5] = ((take_9 != 0) ? xk_5 : yk_7);
                                buf_c[5] = ((take_9 != 0) ? xi_6 : yi_8);
                                unsigned int xk_10 = buf_b[6];
                                int xi_11 = buf_c[6];
                                unsigned int yk_12 = buf_b[7];
                                int yi_13 = buf_c[7];
                                int take_14 = (int)(yk_12 > xk_10 || yk_12 == xk_10 && yi_13 < xi_11);
                                buf_b[6] = ((take_14 != 0) ? yk_12 : xk_10);
                                buf_c[6] = ((take_14 != 0) ? yi_13 : xi_11);
                                buf_b[7] = ((take_14 != 0) ? xk_10 : yk_12);
                                buf_c[7] = ((take_14 != 0) ? xi_11 : yi_13);
                                unsigned int xk_15 = buf_b[0];
                                int xi_16 = buf_c[0];
                                unsigned int yk_17 = buf_b[2];
                                int yi_18 = buf_c[2];
                                int take_19 = (int)(yk_17 > xk_15 || yk_17 == xk_15 && yi_18 < xi_16);
                                buf_b[0] = ((take_19 != 0) ? yk_17 : xk_15);
                                buf_c[0] = ((take_19 != 0) ? yi_18 : xi_16);
                                buf_b[2] = ((take_19 != 0) ? xk_15 : yk_17);
                                buf_c[2] = ((take_19 != 0) ? xi_16 : yi_18);
                                unsigned int xk_20 = buf_b[1];
                                int xi_21 = buf_c[1];
                                unsigned int yk_22 = buf_b[3];
                                int yi_23 = buf_c[3];
                                int take_24 = (int)(yk_22 > xk_20 || yk_22 == xk_20 && yi_23 < xi_21);
                                buf_b[1] = ((take_24 != 0) ? yk_22 : xk_20);
                                buf_c[1] = ((take_24 != 0) ? yi_23 : xi_21);
                                buf_b[3] = ((take_24 != 0) ? xk_20 : yk_22);
                                buf_c[3] = ((take_24 != 0) ? xi_21 : yi_23);
                                unsigned int xk_25 = buf_b[4];
                                int xi_26 = buf_c[4];
                                unsigned int yk_27 = buf_b[6];
                                int yi_28 = buf_c[6];
                                int take_29 = (int)(yk_27 > xk_25 || yk_27 == xk_25 && yi_28 < xi_26);
                                buf_b[4] = ((take_29 != 0) ? yk_27 : xk_25);
                                buf_c[4] = ((take_29 != 0) ? yi_28 : xi_26);
                                buf_b[6] = ((take_29 != 0) ? xk_25 : yk_27);
                                buf_c[6] = ((take_29 != 0) ? xi_26 : yi_28);
                                unsigned int xk_30 = buf_b[5];
                                int xi_31 = buf_c[5];
                                unsigned int yk_32 = buf_b[7];
                                int yi_33 = buf_c[7];
                                int take_34 = (int)(yk_32 > xk_30 || yk_32 == xk_30 && yi_33 < xi_31);
                                buf_b[5] = ((take_34 != 0) ? yk_32 : xk_30);
                                buf_c[5] = ((take_34 != 0) ? yi_33 : xi_31);
                                buf_b[7] = ((take_34 != 0) ? xk_30 : yk_32);
                                buf_c[7] = ((take_34 != 0) ? xi_31 : yi_33);
                                unsigned int xk_35 = buf_b[1];
                                int xi_36 = buf_c[1];
                                unsigned int yk_37 = buf_b[2];
                                int yi_38 = buf_c[2];
                                int take_39 = (int)(yk_37 > xk_35 || yk_37 == xk_35 && yi_38 < xi_36);
                                buf_b[1] = ((take_39 != 0) ? yk_37 : xk_35);
                                buf_c[1] = ((take_39 != 0) ? yi_38 : xi_36);
                                buf_b[2] = ((take_39 != 0) ? xk_35 : yk_37);
                                buf_c[2] = ((take_39 != 0) ? xi_36 : yi_38);
                                unsigned int xk_40 = buf_b[5];
                                int xi_41 = buf_c[5];
                                unsigned int yk_42 = buf_b[6];
                                int yi_43 = buf_c[6];
                                int take_44 = (int)(yk_42 > xk_40 || yk_42 == xk_40 && yi_43 < xi_41);
                                buf_b[5] = ((take_44 != 0) ? yk_42 : xk_40);
                                buf_c[5] = ((take_44 != 0) ? yi_43 : xi_41);
                                buf_b[6] = ((take_44 != 0) ? xk_40 : yk_42);
                                buf_c[6] = ((take_44 != 0) ? xi_41 : yi_43);
                                unsigned int xk_45 = buf_b[0];
                                int xi_46 = buf_c[0];
                                unsigned int yk_47 = buf_b[4];
                                int yi_48 = buf_c[4];
                                int take_49 = (int)(yk_47 > xk_45 || yk_47 == xk_45 && yi_48 < xi_46);
                                buf_b[0] = ((take_49 != 0) ? yk_47 : xk_45);
                                buf_c[0] = ((take_49 != 0) ? yi_48 : xi_46);
                                buf_b[4] = ((take_49 != 0) ? xk_45 : yk_47);
                                buf_c[4] = ((take_49 != 0) ? xi_46 : yi_48);
                                unsigned int xk_50 = buf_b[1];
                                int xi_51 = buf_c[1];
                                unsigned int yk_52 = buf_b[5];
                                int yi_53 = buf_c[5];
                                int take_54 = (int)(yk_52 > xk_50 || yk_52 == xk_50 && yi_53 < xi_51);
                                buf_b[1] = ((take_54 != 0) ? yk_52 : xk_50);
                                buf_c[1] = ((take_54 != 0) ? yi_53 : xi_51);
                                buf_b[5] = ((take_54 != 0) ? xk_50 : yk_52);
                                buf_c[5] = ((take_54 != 0) ? xi_51 : yi_53);
                                unsigned int xk_55 = buf_b[2];
                                int xi_56 = buf_c[2];
                                unsigned int yk_57 = buf_b[6];
                                int yi_58 = buf_c[6];
                                int take_59 = (int)(yk_57 > xk_55 || yk_57 == xk_55 && yi_58 < xi_56);
                                buf_b[2] = ((take_59 != 0) ? yk_57 : xk_55);
                                buf_c[2] = ((take_59 != 0) ? yi_58 : xi_56);
                                buf_b[6] = ((take_59 != 0) ? xk_55 : yk_57);
                                buf_c[6] = ((take_59 != 0) ? xi_56 : yi_58);
                                unsigned int xk_60 = buf_b[3];
                                int xi_61 = buf_c[3];
                                unsigned int yk_62 = buf_b[7];
                                int yi_63 = buf_c[7];
                                int take_64 = (int)(yk_62 > xk_60 || yk_62 == xk_60 && yi_63 < xi_61);
                                buf_b[3] = ((take_64 != 0) ? yk_62 : xk_60);
                                buf_c[3] = ((take_64 != 0) ? yi_63 : xi_61);
                                buf_b[7] = ((take_64 != 0) ? xk_60 : yk_62);
                                buf_c[7] = ((take_64 != 0) ? xi_61 : yi_63);
                                unsigned int xk_65 = buf_b[2];
                                int xi_66 = buf_c[2];
                                unsigned int yk_67 = buf_b[4];
                                int yi_68 = buf_c[4];
                                int take_69 = (int)(yk_67 > xk_65 || yk_67 == xk_65 && yi_68 < xi_66);
                                buf_b[2] = ((take_69 != 0) ? yk_67 : xk_65);
                                buf_c[2] = ((take_69 != 0) ? yi_68 : xi_66);
                                buf_b[4] = ((take_69 != 0) ? xk_65 : yk_67);
                                buf_c[4] = ((take_69 != 0) ? xi_66 : yi_68);
                                unsigned int xk_70 = buf_b[3];
                                int xi_71 = buf_c[3];
                                unsigned int yk_72 = buf_b[5];
                                int yi_73 = buf_c[5];
                                int take_74 = (int)(yk_72 > xk_70 || yk_72 == xk_70 && yi_73 < xi_71);
                                buf_b[3] = ((take_74 != 0) ? yk_72 : xk_70);
                                buf_c[3] = ((take_74 != 0) ? yi_73 : xi_71);
                                buf_b[5] = ((take_74 != 0) ? xk_70 : yk_72);
                                buf_c[5] = ((take_74 != 0) ? xi_71 : yi_73);
                                unsigned int xk_75 = buf_b[1];
                                int xi_76 = buf_c[1];
                                unsigned int yk_77 = buf_b[2];
                                int yi_78 = buf_c[2];
                                int take_79 = (int)(yk_77 > xk_75 || yk_77 == xk_75 && yi_78 < xi_76);
                                buf_b[1] = ((take_79 != 0) ? yk_77 : xk_75);
                                buf_c[1] = ((take_79 != 0) ? yi_78 : xi_76);
                                buf_b[2] = ((take_79 != 0) ? xk_75 : yk_77);
                                buf_c[2] = ((take_79 != 0) ? xi_76 : yi_78);
                                unsigned int xk_80 = buf_b[3];
                                int xi_81 = buf_c[3];
                                unsigned int yk_82 = buf_b[4];
                                int yi_83 = buf_c[4];
                                int take_84 = (int)(yk_82 > xk_80 || yk_82 == xk_80 && yi_83 < xi_81);
                                buf_b[3] = ((take_84 != 0) ? yk_82 : xk_80);
                                buf_c[3] = ((take_84 != 0) ? yi_83 : xi_81);
                                buf_b[4] = ((take_84 != 0) ? xk_80 : yk_82);
                                buf_c[4] = ((take_84 != 0) ? xi_81 : yi_83);
                                unsigned int xk_85 = buf_b[5];
                                int xi_86 = buf_c[5];
                                unsigned int yk_87 = buf_b[6];
                                int yi_88 = buf_c[6];
                                int take_89 = (int)(yk_87 > xk_85 || yk_87 == xk_85 && yi_88 < xi_86);
                                buf_b[5] = ((take_89 != 0) ? yk_87 : xk_85);
                                buf_c[5] = ((take_89 != 0) ? yi_88 : xi_86);
                                buf_b[6] = ((take_89 != 0) ? xk_85 : yk_87);
                                buf_c[6] = ((take_89 != 0) ? xi_86 : yi_88);
                                unsigned int xk_90 = top_b[15];
                                int xi_91 = top_c[15];
                                unsigned int yk_92 = buf_b[0];
                                int yi_93 = buf_c[0];
                                int take_94 = (int)(yk_92 > xk_90 || yk_92 == xk_90 && yi_93 < xi_91);
                                top_b[15] = ((take_94 != 0) ? yk_92 : xk_90);
                                top_c[15] = ((take_94 != 0) ? yi_93 : xi_91);
                                unsigned int xk_95 = top_b[14];
                                int xi_96 = top_c[14];
                                unsigned int yk_97 = buf_b[1];
                                int yi_98 = buf_c[1];
                                int take_99 = (int)(yk_97 > xk_95 || yk_97 == xk_95 && yi_98 < xi_96);
                                top_b[14] = ((take_99 != 0) ? yk_97 : xk_95);
                                top_c[14] = ((take_99 != 0) ? yi_98 : xi_96);
                                unsigned int xk_100 = top_b[13];
                                int xi_101 = top_c[13];
                                unsigned int yk_102 = buf_b[2];
                                int yi_103 = buf_c[2];
                                int take_104 = (int)(yk_102 > xk_100 || yk_102 == xk_100 && yi_103 < xi_101);
                                top_b[13] = ((take_104 != 0) ? yk_102 : xk_100);
                                top_c[13] = ((take_104 != 0) ? yi_103 : xi_101);
                                unsigned int xk_105 = top_b[12];
                                int xi_106 = top_c[12];
                                unsigned int yk_107 = buf_b[3];
                                int yi_108 = buf_c[3];
                                int take_109 = (int)(yk_107 > xk_105 || yk_107 == xk_105 && yi_108 < xi_106);
                                top_b[12] = ((take_109 != 0) ? yk_107 : xk_105);
                                top_c[12] = ((take_109 != 0) ? yi_108 : xi_106);
                                unsigned int xk_110 = top_b[11];
                                int xi_111 = top_c[11];
                                unsigned int yk_112 = buf_b[4];
                                int yi_113 = buf_c[4];
                                int take_114 = (int)(yk_112 > xk_110 || yk_112 == xk_110 && yi_113 < xi_111);
                                top_b[11] = ((take_114 != 0) ? yk_112 : xk_110);
                                top_c[11] = ((take_114 != 0) ? yi_113 : xi_111);
                                unsigned int xk_115 = top_b[10];
                                int xi_116 = top_c[10];
                                unsigned int yk_117 = buf_b[5];
                                int yi_118 = buf_c[5];
                                int take_119 = (int)(yk_117 > xk_115 || yk_117 == xk_115 && yi_118 < xi_116);
                                top_b[10] = ((take_119 != 0) ? yk_117 : xk_115);
                                top_c[10] = ((take_119 != 0) ? yi_118 : xi_116);
                                unsigned int xk_120 = top_b[9];
                                int xi_121 = top_c[9];
                                unsigned int yk_122 = buf_b[6];
                                int yi_123 = buf_c[6];
                                int take_124 = (int)(yk_122 > xk_120 || yk_122 == xk_120 && yi_123 < xi_121);
                                top_b[9] = ((take_124 != 0) ? yk_122 : xk_120);
                                top_c[9] = ((take_124 != 0) ? yi_123 : xi_121);
                                unsigned int xk_125 = top_b[8];
                                int xi_126 = top_c[8];
                                unsigned int yk_127 = buf_b[7];
                                int yi_128 = buf_c[7];
                                int take_129 = (int)(yk_127 > xk_125 || yk_127 == xk_125 && yi_128 < xi_126);
                                top_b[8] = ((take_129 != 0) ? yk_127 : xk_125);
                                top_c[8] = ((take_129 != 0) ? yi_128 : xi_126);
                                unsigned int xk_130 = top_b[0];
                                int xi_131 = top_c[0];
                                unsigned int yk_132 = top_b[8];
                                int yi_133 = top_c[8];
                                int take_134 = (int)(yk_132 > xk_130 || yk_132 == xk_130 && yi_133 < xi_131);
                                top_b[0] = ((take_134 != 0) ? yk_132 : xk_130);
                                top_c[0] = ((take_134 != 0) ? yi_133 : xi_131);
                                top_b[8] = ((take_134 != 0) ? xk_130 : yk_132);
                                top_c[8] = ((take_134 != 0) ? xi_131 : yi_133);
                                unsigned int xk_135 = top_b[1];
                                int xi_136 = top_c[1];
                                unsigned int yk_137 = top_b[9];
                                int yi_138 = top_c[9];
                                int take_139 = (int)(yk_137 > xk_135 || yk_137 == xk_135 && yi_138 < xi_136);
                                top_b[1] = ((take_139 != 0) ? yk_137 : xk_135);
                                top_c[1] = ((take_139 != 0) ? yi_138 : xi_136);
                                top_b[9] = ((take_139 != 0) ? xk_135 : yk_137);
                                top_c[9] = ((take_139 != 0) ? xi_136 : yi_138);
                                unsigned int xk_140 = top_b[2];
                                int xi_141 = top_c[2];
                                unsigned int yk_142 = top_b[10];
                                int yi_143 = top_c[10];
                                int take_144 = (int)(yk_142 > xk_140 || yk_142 == xk_140 && yi_143 < xi_141);
                                top_b[2] = ((take_144 != 0) ? yk_142 : xk_140);
                                top_c[2] = ((take_144 != 0) ? yi_143 : xi_141);
                                top_b[10] = ((take_144 != 0) ? xk_140 : yk_142);
                                top_c[10] = ((take_144 != 0) ? xi_141 : yi_143);
                                unsigned int xk_145 = top_b[3];
                                int xi_146 = top_c[3];
                                unsigned int yk_147 = top_b[11];
                                int yi_148 = top_c[11];
                                int take_149 = (int)(yk_147 > xk_145 || yk_147 == xk_145 && yi_148 < xi_146);
                                top_b[3] = ((take_149 != 0) ? yk_147 : xk_145);
                                top_c[3] = ((take_149 != 0) ? yi_148 : xi_146);
                                top_b[11] = ((take_149 != 0) ? xk_145 : yk_147);
                                top_c[11] = ((take_149 != 0) ? xi_146 : yi_148);
                                unsigned int xk_150 = top_b[4];
                                int xi_151 = top_c[4];
                                unsigned int yk_152 = top_b[12];
                                int yi_153 = top_c[12];
                                int take_154 = (int)(yk_152 > xk_150 || yk_152 == xk_150 && yi_153 < xi_151);
                                top_b[4] = ((take_154 != 0) ? yk_152 : xk_150);
                                top_c[4] = ((take_154 != 0) ? yi_153 : xi_151);
                                top_b[12] = ((take_154 != 0) ? xk_150 : yk_152);
                                top_c[12] = ((take_154 != 0) ? xi_151 : yi_153);
                                unsigned int xk_155 = top_b[5];
                                int xi_156 = top_c[5];
                                unsigned int yk_157 = top_b[13];
                                int yi_158 = top_c[13];
                                int take_159 = (int)(yk_157 > xk_155 || yk_157 == xk_155 && yi_158 < xi_156);
                                top_b[5] = ((take_159 != 0) ? yk_157 : xk_155);
                                top_c[5] = ((take_159 != 0) ? yi_158 : xi_156);
                                top_b[13] = ((take_159 != 0) ? xk_155 : yk_157);
                                top_c[13] = ((take_159 != 0) ? xi_156 : yi_158);
                                unsigned int xk_160 = top_b[6];
                                int xi_161 = top_c[6];
                                unsigned int yk_162 = top_b[14];
                                int yi_163 = top_c[14];
                                int take_164 = (int)(yk_162 > xk_160 || yk_162 == xk_160 && yi_163 < xi_161);
                                top_b[6] = ((take_164 != 0) ? yk_162 : xk_160);
                                top_c[6] = ((take_164 != 0) ? yi_163 : xi_161);
                                top_b[14] = ((take_164 != 0) ? xk_160 : yk_162);
                                top_c[14] = ((take_164 != 0) ? xi_161 : yi_163);
                                unsigned int xk_165 = top_b[7];
                                int xi_166 = top_c[7];
                                unsigned int yk_167 = top_b[15];
                                int yi_168 = top_c[15];
                                int take_169 = (int)(yk_167 > xk_165 || yk_167 == xk_165 && yi_168 < xi_166);
                                top_b[7] = ((take_169 != 0) ? yk_167 : xk_165);
                                top_c[7] = ((take_169 != 0) ? yi_168 : xi_166);
                                top_b[15] = ((take_169 != 0) ? xk_165 : yk_167);
                                top_c[15] = ((take_169 != 0) ? xi_166 : yi_168);
                                unsigned int xk_170 = top_b[0];
                                int xi_171 = top_c[0];
                                unsigned int yk_172 = top_b[4];
                                int yi_173 = top_c[4];
                                int take_174 = (int)(yk_172 > xk_170 || yk_172 == xk_170 && yi_173 < xi_171);
                                top_b[0] = ((take_174 != 0) ? yk_172 : xk_170);
                                top_c[0] = ((take_174 != 0) ? yi_173 : xi_171);
                                top_b[4] = ((take_174 != 0) ? xk_170 : yk_172);
                                top_c[4] = ((take_174 != 0) ? xi_171 : yi_173);
                                unsigned int xk_175 = top_b[1];
                                int xi_176 = top_c[1];
                                unsigned int yk_177 = top_b[5];
                                int yi_178 = top_c[5];
                                int take_179 = (int)(yk_177 > xk_175 || yk_177 == xk_175 && yi_178 < xi_176);
                                top_b[1] = ((take_179 != 0) ? yk_177 : xk_175);
                                top_c[1] = ((take_179 != 0) ? yi_178 : xi_176);
                                top_b[5] = ((take_179 != 0) ? xk_175 : yk_177);
                                top_c[5] = ((take_179 != 0) ? xi_176 : yi_178);
                                unsigned int xk_180 = top_b[2];
                                int xi_181 = top_c[2];
                                unsigned int yk_182 = top_b[6];
                                int yi_183 = top_c[6];
                                int take_184 = (int)(yk_182 > xk_180 || yk_182 == xk_180 && yi_183 < xi_181);
                                top_b[2] = ((take_184 != 0) ? yk_182 : xk_180);
                                top_c[2] = ((take_184 != 0) ? yi_183 : xi_181);
                                top_b[6] = ((take_184 != 0) ? xk_180 : yk_182);
                                top_c[6] = ((take_184 != 0) ? xi_181 : yi_183);
                                unsigned int xk_185 = top_b[3];
                                int xi_186 = top_c[3];
                                unsigned int yk_187 = top_b[7];
                                int yi_188 = top_c[7];
                                int take_189 = (int)(yk_187 > xk_185 || yk_187 == xk_185 && yi_188 < xi_186);
                                top_b[3] = ((take_189 != 0) ? yk_187 : xk_185);
                                top_c[3] = ((take_189 != 0) ? yi_188 : xi_186);
                                top_b[7] = ((take_189 != 0) ? xk_185 : yk_187);
                                top_c[7] = ((take_189 != 0) ? xi_186 : yi_188);
                                unsigned int xk_190 = top_b[8];
                                int xi_191 = top_c[8];
                                unsigned int yk_192 = top_b[12];
                                int yi_193 = top_c[12];
                                int take_194 = (int)(yk_192 > xk_190 || yk_192 == xk_190 && yi_193 < xi_191);
                                top_b[8] = ((take_194 != 0) ? yk_192 : xk_190);
                                top_c[8] = ((take_194 != 0) ? yi_193 : xi_191);
                                top_b[12] = ((take_194 != 0) ? xk_190 : yk_192);
                                top_c[12] = ((take_194 != 0) ? xi_191 : yi_193);
                                unsigned int xk_195 = top_b[9];
                                int xi_196 = top_c[9];
                                unsigned int yk_197 = top_b[13];
                                int yi_198 = top_c[13];
                                int take_199 = (int)(yk_197 > xk_195 || yk_197 == xk_195 && yi_198 < xi_196);
                                top_b[9] = ((take_199 != 0) ? yk_197 : xk_195);
                                top_c[9] = ((take_199 != 0) ? yi_198 : xi_196);
                                top_b[13] = ((take_199 != 0) ? xk_195 : yk_197);
                                top_c[13] = ((take_199 != 0) ? xi_196 : yi_198);
                                unsigned int xk_200 = top_b[10];
                                int xi_201 = top_c[10];
                                unsigned int yk_202 = top_b[14];
                                int yi_203 = top_c[14];
                                int take_204 = (int)(yk_202 > xk_200 || yk_202 == xk_200 && yi_203 < xi_201);
                                top_b[10] = ((take_204 != 0) ? yk_202 : xk_200);
                                top_c[10] = ((take_204 != 0) ? yi_203 : xi_201);
                                top_b[14] = ((take_204 != 0) ? xk_200 : yk_202);
                                top_c[14] = ((take_204 != 0) ? xi_201 : yi_203);
                                unsigned int xk_205 = top_b[11];
                                int xi_206 = top_c[11];
                                unsigned int yk_207 = top_b[15];
                                int yi_208 = top_c[15];
                                int take_209 = (int)(yk_207 > xk_205 || yk_207 == xk_205 && yi_208 < xi_206);
                                top_b[11] = ((take_209 != 0) ? yk_207 : xk_205);
                                top_c[11] = ((take_209 != 0) ? yi_208 : xi_206);
                                top_b[15] = ((take_209 != 0) ? xk_205 : yk_207);
                                top_c[15] = ((take_209 != 0) ? xi_206 : yi_208);
                                unsigned int xk_210 = top_b[0];
                                int xi_211 = top_c[0];
                                unsigned int yk_212 = top_b[2];
                                int yi_213 = top_c[2];
                                int take_214 = (int)(yk_212 > xk_210 || yk_212 == xk_210 && yi_213 < xi_211);
                                top_b[0] = ((take_214 != 0) ? yk_212 : xk_210);
                                top_c[0] = ((take_214 != 0) ? yi_213 : xi_211);
                                top_b[2] = ((take_214 != 0) ? xk_210 : yk_212);
                                top_c[2] = ((take_214 != 0) ? xi_211 : yi_213);
                                unsigned int xk_215 = top_b[1];
                                int xi_216 = top_c[1];
                                unsigned int yk_217 = top_b[3];
                                int yi_218 = top_c[3];
                                int take_219 = (int)(yk_217 > xk_215 || yk_217 == xk_215 && yi_218 < xi_216);
                                top_b[1] = ((take_219 != 0) ? yk_217 : xk_215);
                                top_c[1] = ((take_219 != 0) ? yi_218 : xi_216);
                                top_b[3] = ((take_219 != 0) ? xk_215 : yk_217);
                                top_c[3] = ((take_219 != 0) ? xi_216 : yi_218);
                                unsigned int xk_220 = top_b[4];
                                int xi_221 = top_c[4];
                                unsigned int yk_222 = top_b[6];
                                int yi_223 = top_c[6];
                                int take_224 = (int)(yk_222 > xk_220 || yk_222 == xk_220 && yi_223 < xi_221);
                                top_b[4] = ((take_224 != 0) ? yk_222 : xk_220);
                                top_c[4] = ((take_224 != 0) ? yi_223 : xi_221);
                                top_b[6] = ((take_224 != 0) ? xk_220 : yk_222);
                                top_c[6] = ((take_224 != 0) ? xi_221 : yi_223);
                                unsigned int xk_225 = top_b[5];
                                int xi_226 = top_c[5];
                                unsigned int yk_227 = top_b[7];
                                int yi_228 = top_c[7];
                                int take_229 = (int)(yk_227 > xk_225 || yk_227 == xk_225 && yi_228 < xi_226);
                                top_b[5] = ((take_229 != 0) ? yk_227 : xk_225);
                                top_c[5] = ((take_229 != 0) ? yi_228 : xi_226);
                                top_b[7] = ((take_229 != 0) ? xk_225 : yk_227);
                                top_c[7] = ((take_229 != 0) ? xi_226 : yi_228);
                                unsigned int xk_230 = top_b[8];
                                int xi_231 = top_c[8];
                                unsigned int yk_232 = top_b[10];
                                int yi_233 = top_c[10];
                                int take_234 = (int)(yk_232 > xk_230 || yk_232 == xk_230 && yi_233 < xi_231);
                                top_b[8] = ((take_234 != 0) ? yk_232 : xk_230);
                                top_c[8] = ((take_234 != 0) ? yi_233 : xi_231);
                                top_b[10] = ((take_234 != 0) ? xk_230 : yk_232);
                                top_c[10] = ((take_234 != 0) ? xi_231 : yi_233);
                                unsigned int xk_235 = top_b[9];
                                int xi_236 = top_c[9];
                                unsigned int yk_237 = top_b[11];
                                int yi_238 = top_c[11];
                                int take_239 = (int)(yk_237 > xk_235 || yk_237 == xk_235 && yi_238 < xi_236);
                                top_b[9] = ((take_239 != 0) ? yk_237 : xk_235);
                                top_c[9] = ((take_239 != 0) ? yi_238 : xi_236);
                                top_b[11] = ((take_239 != 0) ? xk_235 : yk_237);
                                top_c[11] = ((take_239 != 0) ? xi_236 : yi_238);
                                unsigned int xk_240 = top_b[12];
                                int xi_241 = top_c[12];
                                unsigned int yk_242 = top_b[14];
                                int yi_243 = top_c[14];
                                int take_244 = (int)(yk_242 > xk_240 || yk_242 == xk_240 && yi_243 < xi_241);
                                top_b[12] = ((take_244 != 0) ? yk_242 : xk_240);
                                top_c[12] = ((take_244 != 0) ? yi_243 : xi_241);
                                top_b[14] = ((take_244 != 0) ? xk_240 : yk_242);
                                top_c[14] = ((take_244 != 0) ? xi_241 : yi_243);
                                unsigned int xk_245 = top_b[13];
                                int xi_246 = top_c[13];
                                unsigned int yk_247 = top_b[15];
                                int yi_248 = top_c[15];
                                int take_249 = (int)(yk_247 > xk_245 || yk_247 == xk_245 && yi_248 < xi_246);
                                top_b[13] = ((take_249 != 0) ? yk_247 : xk_245);
                                top_c[13] = ((take_249 != 0) ? yi_248 : xi_246);
                                top_b[15] = ((take_249 != 0) ? xk_245 : yk_247);
                                top_c[15] = ((take_249 != 0) ? xi_246 : yi_248);
                                unsigned int xk_250 = top_b[0];
                                int xi_251 = top_c[0];
                                unsigned int yk_252 = top_b[1];
                                int yi_253 = top_c[1];
                                int take_254 = (int)(yk_252 > xk_250 || yk_252 == xk_250 && yi_253 < xi_251);
                                top_b[0] = ((take_254 != 0) ? yk_252 : xk_250);
                                top_c[0] = ((take_254 != 0) ? yi_253 : xi_251);
                                top_b[1] = ((take_254 != 0) ? xk_250 : yk_252);
                                top_c[1] = ((take_254 != 0) ? xi_251 : yi_253);
                                unsigned int xk_255 = top_b[2];
                                int xi_256 = top_c[2];
                                unsigned int yk_257 = top_b[3];
                                int yi_258 = top_c[3];
                                int take_259 = (int)(yk_257 > xk_255 || yk_257 == xk_255 && yi_258 < xi_256);
                                top_b[2] = ((take_259 != 0) ? yk_257 : xk_255);
                                top_c[2] = ((take_259 != 0) ? yi_258 : xi_256);
                                top_b[3] = ((take_259 != 0) ? xk_255 : yk_257);
                                top_c[3] = ((take_259 != 0) ? xi_256 : yi_258);
                                unsigned int xk_260 = top_b[4];
                                int xi_261 = top_c[4];
                                unsigned int yk_262 = top_b[5];
                                int yi_263 = top_c[5];
                                int take_264 = (int)(yk_262 > xk_260 || yk_262 == xk_260 && yi_263 < xi_261);
                                top_b[4] = ((take_264 != 0) ? yk_262 : xk_260);
                                top_c[4] = ((take_264 != 0) ? yi_263 : xi_261);
                                top_b[5] = ((take_264 != 0) ? xk_260 : yk_262);
                                top_c[5] = ((take_264 != 0) ? xi_261 : yi_263);
                                unsigned int xk_265 = top_b[6];
                                int xi_266 = top_c[6];
                                unsigned int yk_267 = top_b[7];
                                int yi_268 = top_c[7];
                                int take_269 = (int)(yk_267 > xk_265 || yk_267 == xk_265 && yi_268 < xi_266);
                                top_b[6] = ((take_269 != 0) ? yk_267 : xk_265);
                                top_c[6] = ((take_269 != 0) ? yi_268 : xi_266);
                                top_b[7] = ((take_269 != 0) ? xk_265 : yk_267);
                                top_c[7] = ((take_269 != 0) ? xi_266 : yi_268);
                                unsigned int xk_270 = top_b[8];
                                int xi_271 = top_c[8];
                                unsigned int yk_272 = top_b[9];
                                int yi_273 = top_c[9];
                                int take_274 = (int)(yk_272 > xk_270 || yk_272 == xk_270 && yi_273 < xi_271);
                                top_b[8] = ((take_274 != 0) ? yk_272 : xk_270);
                                top_c[8] = ((take_274 != 0) ? yi_273 : xi_271);
                                top_b[9] = ((take_274 != 0) ? xk_270 : yk_272);
                                top_c[9] = ((take_274 != 0) ? xi_271 : yi_273);
                                unsigned int xk_275 = top_b[10];
                                int xi_276 = top_c[10];
                                unsigned int yk_277 = top_b[11];
                                int yi_278 = top_c[11];
                                int take_279 = (int)(yk_277 > xk_275 || yk_277 == xk_275 && yi_278 < xi_276);
                                top_b[10] = ((take_279 != 0) ? yk_277 : xk_275);
                                top_c[10] = ((take_279 != 0) ? yi_278 : xi_276);
                                top_b[11] = ((take_279 != 0) ? xk_275 : yk_277);
                                top_c[11] = ((take_279 != 0) ? xi_276 : yi_278);
                                unsigned int xk_280 = top_b[12];
                                int xi_281 = top_c[12];
                                unsigned int yk_282 = top_b[13];
                                int yi_283 = top_c[13];
                                int take_284 = (int)(yk_282 > xk_280 || yk_282 == xk_280 && yi_283 < xi_281);
                                top_b[12] = ((take_284 != 0) ? yk_282 : xk_280);
                                top_c[12] = ((take_284 != 0) ? yi_283 : xi_281);
                                top_b[13] = ((take_284 != 0) ? xk_280 : yk_282);
                                top_c[13] = ((take_284 != 0) ? xi_281 : yi_283);
                                unsigned int xk_285 = top_b[14];
                                int xi_286 = top_c[14];
                                unsigned int yk_287 = top_b[15];
                                int yi_288 = top_c[15];
                                int take_289 = (int)(yk_287 > xk_285 || yk_287 == xk_285 && yi_288 < xi_286);
                                top_b[14] = ((take_289 != 0) ? yk_287 : xk_285);
                                top_c[14] = ((take_289 != 0) ? yi_288 : xi_286);
                                top_b[15] = ((take_289 != 0) ? xk_285 : yk_287);
                                top_c[15] = ((take_289 != 0) ? xi_286 : yi_288);
                                buf_b[0] = 8388607;
                                buf_b[1] = 8388607;
                                buf_b[2] = 8388607;
                                buf_b[3] = 8388607;
                                buf_b[4] = 8388607;
                                buf_b[5] = 8388607;
                                buf_b[6] = 8388607;
                                buf_b[7] = 8388607;
                                buf_c[0] = 2147483647;
                                buf_c[1] = 2147483647;
                                buf_c[2] = 2147483647;
                                buf_c[3] = 2147483647;
                                buf_c[4] = 2147483647;
                                buf_c[5] = 2147483647;
                                buf_c[6] = 2147483647;
                                buf_c[7] = 2147483647;
                                cnt = 0;
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(acc_empty_addr + (w) * 8);
                    phase = phase ^ 1;
                }
                int _vote_2 = __any_sync(0xFFFFFFFF, cnt != 0);
                if (_vote_2 != 0) {
                    unsigned int xk_1 = buf_b[0];
                    int xi_2 = buf_c[0];
                    unsigned int yk_1 = buf_b[1];
                    int yi_1 = buf_c[1];
                    int take_1 = (int)(yk_1 > xk_1 || yk_1 == xk_1 && yi_1 < xi_2);
                    buf_b[0] = ((take_1 != 0) ? yk_1 : xk_1);
                    buf_c[0] = ((take_1 != 0) ? yi_1 : xi_2);
                    buf_b[1] = ((take_1 != 0) ? xk_1 : yk_1);
                    buf_c[1] = ((take_1 != 0) ? xi_2 : yi_1);
                    unsigned int xk_0_1 = buf_b[2];
                    int xi_1_1 = buf_c[2];
                    unsigned int yk_2_1 = buf_b[3];
                    int yi_3_1 = buf_c[3];
                    int take_4_1 = (int)(yk_2_1 > xk_0_1 || yk_2_1 == xk_0_1 && yi_3_1 < xi_1_1);
                    buf_b[2] = ((take_4_1 != 0) ? yk_2_1 : xk_0_1);
                    buf_c[2] = ((take_4_1 != 0) ? yi_3_1 : xi_1_1);
                    buf_b[3] = ((take_4_1 != 0) ? xk_0_1 : yk_2_1);
                    buf_c[3] = ((take_4_1 != 0) ? xi_1_1 : yi_3_1);
                    unsigned int xk_5_1 = buf_b[4];
                    int xi_6_1 = buf_c[4];
                    unsigned int yk_7_1 = buf_b[5];
                    int yi_8_1 = buf_c[5];
                    int take_9_1 = (int)(yk_7_1 > xk_5_1 || yk_7_1 == xk_5_1 && yi_8_1 < xi_6_1);
                    buf_b[4] = ((take_9_1 != 0) ? yk_7_1 : xk_5_1);
                    buf_c[4] = ((take_9_1 != 0) ? yi_8_1 : xi_6_1);
                    buf_b[5] = ((take_9_1 != 0) ? xk_5_1 : yk_7_1);
                    buf_c[5] = ((take_9_1 != 0) ? xi_6_1 : yi_8_1);
                    unsigned int xk_10_1 = buf_b[6];
                    int xi_11_1 = buf_c[6];
                    unsigned int yk_12_1 = buf_b[7];
                    int yi_13_1 = buf_c[7];
                    int take_14_1 = (int)(yk_12_1 > xk_10_1 || yk_12_1 == xk_10_1 && yi_13_1 < xi_11_1);
                    buf_b[6] = ((take_14_1 != 0) ? yk_12_1 : xk_10_1);
                    buf_c[6] = ((take_14_1 != 0) ? yi_13_1 : xi_11_1);
                    buf_b[7] = ((take_14_1 != 0) ? xk_10_1 : yk_12_1);
                    buf_c[7] = ((take_14_1 != 0) ? xi_11_1 : yi_13_1);
                    unsigned int xk_15_1 = buf_b[0];
                    int xi_16_1 = buf_c[0];
                    unsigned int yk_17_1 = buf_b[2];
                    int yi_18_1 = buf_c[2];
                    int take_19_1 = (int)(yk_17_1 > xk_15_1 || yk_17_1 == xk_15_1 && yi_18_1 < xi_16_1);
                    buf_b[0] = ((take_19_1 != 0) ? yk_17_1 : xk_15_1);
                    buf_c[0] = ((take_19_1 != 0) ? yi_18_1 : xi_16_1);
                    buf_b[2] = ((take_19_1 != 0) ? xk_15_1 : yk_17_1);
                    buf_c[2] = ((take_19_1 != 0) ? xi_16_1 : yi_18_1);
                    unsigned int xk_20_1 = buf_b[1];
                    int xi_21_1 = buf_c[1];
                    unsigned int yk_22_1 = buf_b[3];
                    int yi_23_1 = buf_c[3];
                    int take_24_1 = (int)(yk_22_1 > xk_20_1 || yk_22_1 == xk_20_1 && yi_23_1 < xi_21_1);
                    buf_b[1] = ((take_24_1 != 0) ? yk_22_1 : xk_20_1);
                    buf_c[1] = ((take_24_1 != 0) ? yi_23_1 : xi_21_1);
                    buf_b[3] = ((take_24_1 != 0) ? xk_20_1 : yk_22_1);
                    buf_c[3] = ((take_24_1 != 0) ? xi_21_1 : yi_23_1);
                    unsigned int xk_25_1 = buf_b[4];
                    int xi_26_1 = buf_c[4];
                    unsigned int yk_27_1 = buf_b[6];
                    int yi_28_1 = buf_c[6];
                    int take_29_1 = (int)(yk_27_1 > xk_25_1 || yk_27_1 == xk_25_1 && yi_28_1 < xi_26_1);
                    buf_b[4] = ((take_29_1 != 0) ? yk_27_1 : xk_25_1);
                    buf_c[4] = ((take_29_1 != 0) ? yi_28_1 : xi_26_1);
                    buf_b[6] = ((take_29_1 != 0) ? xk_25_1 : yk_27_1);
                    buf_c[6] = ((take_29_1 != 0) ? xi_26_1 : yi_28_1);
                    unsigned int xk_30_1 = buf_b[5];
                    int xi_31_1 = buf_c[5];
                    unsigned int yk_32_1 = buf_b[7];
                    int yi_33_1 = buf_c[7];
                    int take_34_1 = (int)(yk_32_1 > xk_30_1 || yk_32_1 == xk_30_1 && yi_33_1 < xi_31_1);
                    buf_b[5] = ((take_34_1 != 0) ? yk_32_1 : xk_30_1);
                    buf_c[5] = ((take_34_1 != 0) ? yi_33_1 : xi_31_1);
                    buf_b[7] = ((take_34_1 != 0) ? xk_30_1 : yk_32_1);
                    buf_c[7] = ((take_34_1 != 0) ? xi_31_1 : yi_33_1);
                    unsigned int xk_35_1 = buf_b[1];
                    int xi_36_1 = buf_c[1];
                    unsigned int yk_37_1 = buf_b[2];
                    int yi_38_1 = buf_c[2];
                    int take_39_1 = (int)(yk_37_1 > xk_35_1 || yk_37_1 == xk_35_1 && yi_38_1 < xi_36_1);
                    buf_b[1] = ((take_39_1 != 0) ? yk_37_1 : xk_35_1);
                    buf_c[1] = ((take_39_1 != 0) ? yi_38_1 : xi_36_1);
                    buf_b[2] = ((take_39_1 != 0) ? xk_35_1 : yk_37_1);
                    buf_c[2] = ((take_39_1 != 0) ? xi_36_1 : yi_38_1);
                    unsigned int xk_40_1 = buf_b[5];
                    int xi_41_1 = buf_c[5];
                    unsigned int yk_42_1 = buf_b[6];
                    int yi_43_1 = buf_c[6];
                    int take_44_1 = (int)(yk_42_1 > xk_40_1 || yk_42_1 == xk_40_1 && yi_43_1 < xi_41_1);
                    buf_b[5] = ((take_44_1 != 0) ? yk_42_1 : xk_40_1);
                    buf_c[5] = ((take_44_1 != 0) ? yi_43_1 : xi_41_1);
                    buf_b[6] = ((take_44_1 != 0) ? xk_40_1 : yk_42_1);
                    buf_c[6] = ((take_44_1 != 0) ? xi_41_1 : yi_43_1);
                    unsigned int xk_45_1 = buf_b[0];
                    int xi_46_1 = buf_c[0];
                    unsigned int yk_47_1 = buf_b[4];
                    int yi_48_1 = buf_c[4];
                    int take_49_1 = (int)(yk_47_1 > xk_45_1 || yk_47_1 == xk_45_1 && yi_48_1 < xi_46_1);
                    buf_b[0] = ((take_49_1 != 0) ? yk_47_1 : xk_45_1);
                    buf_c[0] = ((take_49_1 != 0) ? yi_48_1 : xi_46_1);
                    buf_b[4] = ((take_49_1 != 0) ? xk_45_1 : yk_47_1);
                    buf_c[4] = ((take_49_1 != 0) ? xi_46_1 : yi_48_1);
                    unsigned int xk_50_1 = buf_b[1];
                    int xi_51_1 = buf_c[1];
                    unsigned int yk_52_1 = buf_b[5];
                    int yi_53_1 = buf_c[5];
                    int take_54_1 = (int)(yk_52_1 > xk_50_1 || yk_52_1 == xk_50_1 && yi_53_1 < xi_51_1);
                    buf_b[1] = ((take_54_1 != 0) ? yk_52_1 : xk_50_1);
                    buf_c[1] = ((take_54_1 != 0) ? yi_53_1 : xi_51_1);
                    buf_b[5] = ((take_54_1 != 0) ? xk_50_1 : yk_52_1);
                    buf_c[5] = ((take_54_1 != 0) ? xi_51_1 : yi_53_1);
                    unsigned int xk_55_1 = buf_b[2];
                    int xi_56_1 = buf_c[2];
                    unsigned int yk_57_1 = buf_b[6];
                    int yi_58_1 = buf_c[6];
                    int take_59_1 = (int)(yk_57_1 > xk_55_1 || yk_57_1 == xk_55_1 && yi_58_1 < xi_56_1);
                    buf_b[2] = ((take_59_1 != 0) ? yk_57_1 : xk_55_1);
                    buf_c[2] = ((take_59_1 != 0) ? yi_58_1 : xi_56_1);
                    buf_b[6] = ((take_59_1 != 0) ? xk_55_1 : yk_57_1);
                    buf_c[6] = ((take_59_1 != 0) ? xi_56_1 : yi_58_1);
                    unsigned int xk_60_1 = buf_b[3];
                    int xi_61_1 = buf_c[3];
                    unsigned int yk_62_1 = buf_b[7];
                    int yi_63_1 = buf_c[7];
                    int take_64_1 = (int)(yk_62_1 > xk_60_1 || yk_62_1 == xk_60_1 && yi_63_1 < xi_61_1);
                    buf_b[3] = ((take_64_1 != 0) ? yk_62_1 : xk_60_1);
                    buf_c[3] = ((take_64_1 != 0) ? yi_63_1 : xi_61_1);
                    buf_b[7] = ((take_64_1 != 0) ? xk_60_1 : yk_62_1);
                    buf_c[7] = ((take_64_1 != 0) ? xi_61_1 : yi_63_1);
                    unsigned int xk_65_1 = buf_b[2];
                    int xi_66_1 = buf_c[2];
                    unsigned int yk_67_1 = buf_b[4];
                    int yi_68_1 = buf_c[4];
                    int take_69_1 = (int)(yk_67_1 > xk_65_1 || yk_67_1 == xk_65_1 && yi_68_1 < xi_66_1);
                    buf_b[2] = ((take_69_1 != 0) ? yk_67_1 : xk_65_1);
                    buf_c[2] = ((take_69_1 != 0) ? yi_68_1 : xi_66_1);
                    buf_b[4] = ((take_69_1 != 0) ? xk_65_1 : yk_67_1);
                    buf_c[4] = ((take_69_1 != 0) ? xi_66_1 : yi_68_1);
                    unsigned int xk_70_1 = buf_b[3];
                    int xi_71_1 = buf_c[3];
                    unsigned int yk_72_1 = buf_b[5];
                    int yi_73_1 = buf_c[5];
                    int take_74_1 = (int)(yk_72_1 > xk_70_1 || yk_72_1 == xk_70_1 && yi_73_1 < xi_71_1);
                    buf_b[3] = ((take_74_1 != 0) ? yk_72_1 : xk_70_1);
                    buf_c[3] = ((take_74_1 != 0) ? yi_73_1 : xi_71_1);
                    buf_b[5] = ((take_74_1 != 0) ? xk_70_1 : yk_72_1);
                    buf_c[5] = ((take_74_1 != 0) ? xi_71_1 : yi_73_1);
                    unsigned int xk_75_1 = buf_b[1];
                    int xi_76_1 = buf_c[1];
                    unsigned int yk_77_1 = buf_b[2];
                    int yi_78_1 = buf_c[2];
                    int take_79_1 = (int)(yk_77_1 > xk_75_1 || yk_77_1 == xk_75_1 && yi_78_1 < xi_76_1);
                    buf_b[1] = ((take_79_1 != 0) ? yk_77_1 : xk_75_1);
                    buf_c[1] = ((take_79_1 != 0) ? yi_78_1 : xi_76_1);
                    buf_b[2] = ((take_79_1 != 0) ? xk_75_1 : yk_77_1);
                    buf_c[2] = ((take_79_1 != 0) ? xi_76_1 : yi_78_1);
                    unsigned int xk_80_1 = buf_b[3];
                    int xi_81_1 = buf_c[3];
                    unsigned int yk_82_1 = buf_b[4];
                    int yi_83_1 = buf_c[4];
                    int take_84_1 = (int)(yk_82_1 > xk_80_1 || yk_82_1 == xk_80_1 && yi_83_1 < xi_81_1);
                    buf_b[3] = ((take_84_1 != 0) ? yk_82_1 : xk_80_1);
                    buf_c[3] = ((take_84_1 != 0) ? yi_83_1 : xi_81_1);
                    buf_b[4] = ((take_84_1 != 0) ? xk_80_1 : yk_82_1);
                    buf_c[4] = ((take_84_1 != 0) ? xi_81_1 : yi_83_1);
                    unsigned int xk_85_1 = buf_b[5];
                    int xi_86_1 = buf_c[5];
                    unsigned int yk_87_1 = buf_b[6];
                    int yi_88_1 = buf_c[6];
                    int take_89_1 = (int)(yk_87_1 > xk_85_1 || yk_87_1 == xk_85_1 && yi_88_1 < xi_86_1);
                    buf_b[5] = ((take_89_1 != 0) ? yk_87_1 : xk_85_1);
                    buf_c[5] = ((take_89_1 != 0) ? yi_88_1 : xi_86_1);
                    buf_b[6] = ((take_89_1 != 0) ? xk_85_1 : yk_87_1);
                    buf_c[6] = ((take_89_1 != 0) ? xi_86_1 : yi_88_1);
                    unsigned int xk_90_1 = top_b[15];
                    int xi_91_1 = top_c[15];
                    unsigned int yk_92_1 = buf_b[0];
                    int yi_93_1 = buf_c[0];
                    int take_94_1 = (int)(yk_92_1 > xk_90_1 || yk_92_1 == xk_90_1 && yi_93_1 < xi_91_1);
                    top_b[15] = ((take_94_1 != 0) ? yk_92_1 : xk_90_1);
                    top_c[15] = ((take_94_1 != 0) ? yi_93_1 : xi_91_1);
                    unsigned int xk_95_1 = top_b[14];
                    int xi_96_1 = top_c[14];
                    unsigned int yk_97_1 = buf_b[1];
                    int yi_98_1 = buf_c[1];
                    int take_99_1 = (int)(yk_97_1 > xk_95_1 || yk_97_1 == xk_95_1 && yi_98_1 < xi_96_1);
                    top_b[14] = ((take_99_1 != 0) ? yk_97_1 : xk_95_1);
                    top_c[14] = ((take_99_1 != 0) ? yi_98_1 : xi_96_1);
                    unsigned int xk_100_1 = top_b[13];
                    int xi_101_1 = top_c[13];
                    unsigned int yk_102_1 = buf_b[2];
                    int yi_103_1 = buf_c[2];
                    int take_104_1 = (int)(yk_102_1 > xk_100_1 || yk_102_1 == xk_100_1 && yi_103_1 < xi_101_1);
                    top_b[13] = ((take_104_1 != 0) ? yk_102_1 : xk_100_1);
                    top_c[13] = ((take_104_1 != 0) ? yi_103_1 : xi_101_1);
                    unsigned int xk_105_1 = top_b[12];
                    int xi_106_1 = top_c[12];
                    unsigned int yk_107_1 = buf_b[3];
                    int yi_108_1 = buf_c[3];
                    int take_109_1 = (int)(yk_107_1 > xk_105_1 || yk_107_1 == xk_105_1 && yi_108_1 < xi_106_1);
                    top_b[12] = ((take_109_1 != 0) ? yk_107_1 : xk_105_1);
                    top_c[12] = ((take_109_1 != 0) ? yi_108_1 : xi_106_1);
                    unsigned int xk_110_1 = top_b[11];
                    int xi_111_1 = top_c[11];
                    unsigned int yk_112_1 = buf_b[4];
                    int yi_113_1 = buf_c[4];
                    int take_114_1 = (int)(yk_112_1 > xk_110_1 || yk_112_1 == xk_110_1 && yi_113_1 < xi_111_1);
                    top_b[11] = ((take_114_1 != 0) ? yk_112_1 : xk_110_1);
                    top_c[11] = ((take_114_1 != 0) ? yi_113_1 : xi_111_1);
                    unsigned int xk_115_1 = top_b[10];
                    int xi_116_1 = top_c[10];
                    unsigned int yk_117_1 = buf_b[5];
                    int yi_118_1 = buf_c[5];
                    int take_119_1 = (int)(yk_117_1 > xk_115_1 || yk_117_1 == xk_115_1 && yi_118_1 < xi_116_1);
                    top_b[10] = ((take_119_1 != 0) ? yk_117_1 : xk_115_1);
                    top_c[10] = ((take_119_1 != 0) ? yi_118_1 : xi_116_1);
                    unsigned int xk_120_1 = top_b[9];
                    int xi_121_1 = top_c[9];
                    unsigned int yk_122_1 = buf_b[6];
                    int yi_123_1 = buf_c[6];
                    int take_124_1 = (int)(yk_122_1 > xk_120_1 || yk_122_1 == xk_120_1 && yi_123_1 < xi_121_1);
                    top_b[9] = ((take_124_1 != 0) ? yk_122_1 : xk_120_1);
                    top_c[9] = ((take_124_1 != 0) ? yi_123_1 : xi_121_1);
                    unsigned int xk_125_1 = top_b[8];
                    int xi_126_1 = top_c[8];
                    unsigned int yk_127_1 = buf_b[7];
                    int yi_128_1 = buf_c[7];
                    int take_129_1 = (int)(yk_127_1 > xk_125_1 || yk_127_1 == xk_125_1 && yi_128_1 < xi_126_1);
                    top_b[8] = ((take_129_1 != 0) ? yk_127_1 : xk_125_1);
                    top_c[8] = ((take_129_1 != 0) ? yi_128_1 : xi_126_1);
                    unsigned int xk_130_1 = top_b[0];
                    int xi_131_1 = top_c[0];
                    unsigned int yk_132_1 = top_b[8];
                    int yi_133_1 = top_c[8];
                    int take_134_1 = (int)(yk_132_1 > xk_130_1 || yk_132_1 == xk_130_1 && yi_133_1 < xi_131_1);
                    top_b[0] = ((take_134_1 != 0) ? yk_132_1 : xk_130_1);
                    top_c[0] = ((take_134_1 != 0) ? yi_133_1 : xi_131_1);
                    top_b[8] = ((take_134_1 != 0) ? xk_130_1 : yk_132_1);
                    top_c[8] = ((take_134_1 != 0) ? xi_131_1 : yi_133_1);
                    unsigned int xk_135_1 = top_b[1];
                    int xi_136_1 = top_c[1];
                    unsigned int yk_137_1 = top_b[9];
                    int yi_138_1 = top_c[9];
                    int take_139_1 = (int)(yk_137_1 > xk_135_1 || yk_137_1 == xk_135_1 && yi_138_1 < xi_136_1);
                    top_b[1] = ((take_139_1 != 0) ? yk_137_1 : xk_135_1);
                    top_c[1] = ((take_139_1 != 0) ? yi_138_1 : xi_136_1);
                    top_b[9] = ((take_139_1 != 0) ? xk_135_1 : yk_137_1);
                    top_c[9] = ((take_139_1 != 0) ? xi_136_1 : yi_138_1);
                    unsigned int xk_140_1 = top_b[2];
                    int xi_141_1 = top_c[2];
                    unsigned int yk_142_1 = top_b[10];
                    int yi_143_1 = top_c[10];
                    int take_144_1 = (int)(yk_142_1 > xk_140_1 || yk_142_1 == xk_140_1 && yi_143_1 < xi_141_1);
                    top_b[2] = ((take_144_1 != 0) ? yk_142_1 : xk_140_1);
                    top_c[2] = ((take_144_1 != 0) ? yi_143_1 : xi_141_1);
                    top_b[10] = ((take_144_1 != 0) ? xk_140_1 : yk_142_1);
                    top_c[10] = ((take_144_1 != 0) ? xi_141_1 : yi_143_1);
                    unsigned int xk_145_1 = top_b[3];
                    int xi_146_1 = top_c[3];
                    unsigned int yk_147_1 = top_b[11];
                    int yi_148_1 = top_c[11];
                    int take_149_1 = (int)(yk_147_1 > xk_145_1 || yk_147_1 == xk_145_1 && yi_148_1 < xi_146_1);
                    top_b[3] = ((take_149_1 != 0) ? yk_147_1 : xk_145_1);
                    top_c[3] = ((take_149_1 != 0) ? yi_148_1 : xi_146_1);
                    top_b[11] = ((take_149_1 != 0) ? xk_145_1 : yk_147_1);
                    top_c[11] = ((take_149_1 != 0) ? xi_146_1 : yi_148_1);
                    unsigned int xk_150_1 = top_b[4];
                    int xi_151_1 = top_c[4];
                    unsigned int yk_152_1 = top_b[12];
                    int yi_153_1 = top_c[12];
                    int take_154_1 = (int)(yk_152_1 > xk_150_1 || yk_152_1 == xk_150_1 && yi_153_1 < xi_151_1);
                    top_b[4] = ((take_154_1 != 0) ? yk_152_1 : xk_150_1);
                    top_c[4] = ((take_154_1 != 0) ? yi_153_1 : xi_151_1);
                    top_b[12] = ((take_154_1 != 0) ? xk_150_1 : yk_152_1);
                    top_c[12] = ((take_154_1 != 0) ? xi_151_1 : yi_153_1);
                    unsigned int xk_155_1 = top_b[5];
                    int xi_156_1 = top_c[5];
                    unsigned int yk_157_1 = top_b[13];
                    int yi_158_1 = top_c[13];
                    int take_159_1 = (int)(yk_157_1 > xk_155_1 || yk_157_1 == xk_155_1 && yi_158_1 < xi_156_1);
                    top_b[5] = ((take_159_1 != 0) ? yk_157_1 : xk_155_1);
                    top_c[5] = ((take_159_1 != 0) ? yi_158_1 : xi_156_1);
                    top_b[13] = ((take_159_1 != 0) ? xk_155_1 : yk_157_1);
                    top_c[13] = ((take_159_1 != 0) ? xi_156_1 : yi_158_1);
                    unsigned int xk_160_1 = top_b[6];
                    int xi_161_1 = top_c[6];
                    unsigned int yk_162_1 = top_b[14];
                    int yi_163_1 = top_c[14];
                    int take_164_1 = (int)(yk_162_1 > xk_160_1 || yk_162_1 == xk_160_1 && yi_163_1 < xi_161_1);
                    top_b[6] = ((take_164_1 != 0) ? yk_162_1 : xk_160_1);
                    top_c[6] = ((take_164_1 != 0) ? yi_163_1 : xi_161_1);
                    top_b[14] = ((take_164_1 != 0) ? xk_160_1 : yk_162_1);
                    top_c[14] = ((take_164_1 != 0) ? xi_161_1 : yi_163_1);
                    unsigned int xk_165_1 = top_b[7];
                    int xi_166_1 = top_c[7];
                    unsigned int yk_167_1 = top_b[15];
                    int yi_168_1 = top_c[15];
                    int take_169_1 = (int)(yk_167_1 > xk_165_1 || yk_167_1 == xk_165_1 && yi_168_1 < xi_166_1);
                    top_b[7] = ((take_169_1 != 0) ? yk_167_1 : xk_165_1);
                    top_c[7] = ((take_169_1 != 0) ? yi_168_1 : xi_166_1);
                    top_b[15] = ((take_169_1 != 0) ? xk_165_1 : yk_167_1);
                    top_c[15] = ((take_169_1 != 0) ? xi_166_1 : yi_168_1);
                    unsigned int xk_170_1 = top_b[0];
                    int xi_171_1 = top_c[0];
                    unsigned int yk_172_1 = top_b[4];
                    int yi_173_1 = top_c[4];
                    int take_174_1 = (int)(yk_172_1 > xk_170_1 || yk_172_1 == xk_170_1 && yi_173_1 < xi_171_1);
                    top_b[0] = ((take_174_1 != 0) ? yk_172_1 : xk_170_1);
                    top_c[0] = ((take_174_1 != 0) ? yi_173_1 : xi_171_1);
                    top_b[4] = ((take_174_1 != 0) ? xk_170_1 : yk_172_1);
                    top_c[4] = ((take_174_1 != 0) ? xi_171_1 : yi_173_1);
                    unsigned int xk_175_1 = top_b[1];
                    int xi_176_1 = top_c[1];
                    unsigned int yk_177_1 = top_b[5];
                    int yi_178_1 = top_c[5];
                    int take_179_1 = (int)(yk_177_1 > xk_175_1 || yk_177_1 == xk_175_1 && yi_178_1 < xi_176_1);
                    top_b[1] = ((take_179_1 != 0) ? yk_177_1 : xk_175_1);
                    top_c[1] = ((take_179_1 != 0) ? yi_178_1 : xi_176_1);
                    top_b[5] = ((take_179_1 != 0) ? xk_175_1 : yk_177_1);
                    top_c[5] = ((take_179_1 != 0) ? xi_176_1 : yi_178_1);
                    unsigned int xk_180_1 = top_b[2];
                    int xi_181_1 = top_c[2];
                    unsigned int yk_182_1 = top_b[6];
                    int yi_183_1 = top_c[6];
                    int take_184_1 = (int)(yk_182_1 > xk_180_1 || yk_182_1 == xk_180_1 && yi_183_1 < xi_181_1);
                    top_b[2] = ((take_184_1 != 0) ? yk_182_1 : xk_180_1);
                    top_c[2] = ((take_184_1 != 0) ? yi_183_1 : xi_181_1);
                    top_b[6] = ((take_184_1 != 0) ? xk_180_1 : yk_182_1);
                    top_c[6] = ((take_184_1 != 0) ? xi_181_1 : yi_183_1);
                    unsigned int xk_185_1 = top_b[3];
                    int xi_186_1 = top_c[3];
                    unsigned int yk_187_1 = top_b[7];
                    int yi_188_1 = top_c[7];
                    int take_189_1 = (int)(yk_187_1 > xk_185_1 || yk_187_1 == xk_185_1 && yi_188_1 < xi_186_1);
                    top_b[3] = ((take_189_1 != 0) ? yk_187_1 : xk_185_1);
                    top_c[3] = ((take_189_1 != 0) ? yi_188_1 : xi_186_1);
                    top_b[7] = ((take_189_1 != 0) ? xk_185_1 : yk_187_1);
                    top_c[7] = ((take_189_1 != 0) ? xi_186_1 : yi_188_1);
                    unsigned int xk_190_1 = top_b[8];
                    int xi_191_1 = top_c[8];
                    unsigned int yk_192_1 = top_b[12];
                    int yi_193_1 = top_c[12];
                    int take_194_1 = (int)(yk_192_1 > xk_190_1 || yk_192_1 == xk_190_1 && yi_193_1 < xi_191_1);
                    top_b[8] = ((take_194_1 != 0) ? yk_192_1 : xk_190_1);
                    top_c[8] = ((take_194_1 != 0) ? yi_193_1 : xi_191_1);
                    top_b[12] = ((take_194_1 != 0) ? xk_190_1 : yk_192_1);
                    top_c[12] = ((take_194_1 != 0) ? xi_191_1 : yi_193_1);
                    unsigned int xk_195_1 = top_b[9];
                    int xi_196_1 = top_c[9];
                    unsigned int yk_197_1 = top_b[13];
                    int yi_198_1 = top_c[13];
                    int take_199_1 = (int)(yk_197_1 > xk_195_1 || yk_197_1 == xk_195_1 && yi_198_1 < xi_196_1);
                    top_b[9] = ((take_199_1 != 0) ? yk_197_1 : xk_195_1);
                    top_c[9] = ((take_199_1 != 0) ? yi_198_1 : xi_196_1);
                    top_b[13] = ((take_199_1 != 0) ? xk_195_1 : yk_197_1);
                    top_c[13] = ((take_199_1 != 0) ? xi_196_1 : yi_198_1);
                    unsigned int xk_200_1 = top_b[10];
                    int xi_201_1 = top_c[10];
                    unsigned int yk_202_1 = top_b[14];
                    int yi_203_1 = top_c[14];
                    int take_204_1 = (int)(yk_202_1 > xk_200_1 || yk_202_1 == xk_200_1 && yi_203_1 < xi_201_1);
                    top_b[10] = ((take_204_1 != 0) ? yk_202_1 : xk_200_1);
                    top_c[10] = ((take_204_1 != 0) ? yi_203_1 : xi_201_1);
                    top_b[14] = ((take_204_1 != 0) ? xk_200_1 : yk_202_1);
                    top_c[14] = ((take_204_1 != 0) ? xi_201_1 : yi_203_1);
                    unsigned int xk_205_1 = top_b[11];
                    int xi_206_1 = top_c[11];
                    unsigned int yk_207_1 = top_b[15];
                    int yi_208_1 = top_c[15];
                    int take_209_1 = (int)(yk_207_1 > xk_205_1 || yk_207_1 == xk_205_1 && yi_208_1 < xi_206_1);
                    top_b[11] = ((take_209_1 != 0) ? yk_207_1 : xk_205_1);
                    top_c[11] = ((take_209_1 != 0) ? yi_208_1 : xi_206_1);
                    top_b[15] = ((take_209_1 != 0) ? xk_205_1 : yk_207_1);
                    top_c[15] = ((take_209_1 != 0) ? xi_206_1 : yi_208_1);
                    unsigned int xk_210_1 = top_b[0];
                    int xi_211_1 = top_c[0];
                    unsigned int yk_212_1 = top_b[2];
                    int yi_213_1 = top_c[2];
                    int take_214_1 = (int)(yk_212_1 > xk_210_1 || yk_212_1 == xk_210_1 && yi_213_1 < xi_211_1);
                    top_b[0] = ((take_214_1 != 0) ? yk_212_1 : xk_210_1);
                    top_c[0] = ((take_214_1 != 0) ? yi_213_1 : xi_211_1);
                    top_b[2] = ((take_214_1 != 0) ? xk_210_1 : yk_212_1);
                    top_c[2] = ((take_214_1 != 0) ? xi_211_1 : yi_213_1);
                    unsigned int xk_215_1 = top_b[1];
                    int xi_216_1 = top_c[1];
                    unsigned int yk_217_1 = top_b[3];
                    int yi_218_1 = top_c[3];
                    int take_219_1 = (int)(yk_217_1 > xk_215_1 || yk_217_1 == xk_215_1 && yi_218_1 < xi_216_1);
                    top_b[1] = ((take_219_1 != 0) ? yk_217_1 : xk_215_1);
                    top_c[1] = ((take_219_1 != 0) ? yi_218_1 : xi_216_1);
                    top_b[3] = ((take_219_1 != 0) ? xk_215_1 : yk_217_1);
                    top_c[3] = ((take_219_1 != 0) ? xi_216_1 : yi_218_1);
                    unsigned int xk_220_1 = top_b[4];
                    int xi_221_1 = top_c[4];
                    unsigned int yk_222_1 = top_b[6];
                    int yi_223_1 = top_c[6];
                    int take_224_1 = (int)(yk_222_1 > xk_220_1 || yk_222_1 == xk_220_1 && yi_223_1 < xi_221_1);
                    top_b[4] = ((take_224_1 != 0) ? yk_222_1 : xk_220_1);
                    top_c[4] = ((take_224_1 != 0) ? yi_223_1 : xi_221_1);
                    top_b[6] = ((take_224_1 != 0) ? xk_220_1 : yk_222_1);
                    top_c[6] = ((take_224_1 != 0) ? xi_221_1 : yi_223_1);
                    unsigned int xk_225_1 = top_b[5];
                    int xi_226_1 = top_c[5];
                    unsigned int yk_227_1 = top_b[7];
                    int yi_228_1 = top_c[7];
                    int take_229_1 = (int)(yk_227_1 > xk_225_1 || yk_227_1 == xk_225_1 && yi_228_1 < xi_226_1);
                    top_b[5] = ((take_229_1 != 0) ? yk_227_1 : xk_225_1);
                    top_c[5] = ((take_229_1 != 0) ? yi_228_1 : xi_226_1);
                    top_b[7] = ((take_229_1 != 0) ? xk_225_1 : yk_227_1);
                    top_c[7] = ((take_229_1 != 0) ? xi_226_1 : yi_228_1);
                    unsigned int xk_230_1 = top_b[8];
                    int xi_231_1 = top_c[8];
                    unsigned int yk_232_1 = top_b[10];
                    int yi_233_1 = top_c[10];
                    int take_234_1 = (int)(yk_232_1 > xk_230_1 || yk_232_1 == xk_230_1 && yi_233_1 < xi_231_1);
                    top_b[8] = ((take_234_1 != 0) ? yk_232_1 : xk_230_1);
                    top_c[8] = ((take_234_1 != 0) ? yi_233_1 : xi_231_1);
                    top_b[10] = ((take_234_1 != 0) ? xk_230_1 : yk_232_1);
                    top_c[10] = ((take_234_1 != 0) ? xi_231_1 : yi_233_1);
                    unsigned int xk_235_1 = top_b[9];
                    int xi_236_1 = top_c[9];
                    unsigned int yk_237_1 = top_b[11];
                    int yi_238_1 = top_c[11];
                    int take_239_1 = (int)(yk_237_1 > xk_235_1 || yk_237_1 == xk_235_1 && yi_238_1 < xi_236_1);
                    top_b[9] = ((take_239_1 != 0) ? yk_237_1 : xk_235_1);
                    top_c[9] = ((take_239_1 != 0) ? yi_238_1 : xi_236_1);
                    top_b[11] = ((take_239_1 != 0) ? xk_235_1 : yk_237_1);
                    top_c[11] = ((take_239_1 != 0) ? xi_236_1 : yi_238_1);
                    unsigned int xk_240_1 = top_b[12];
                    int xi_241_1 = top_c[12];
                    unsigned int yk_242_1 = top_b[14];
                    int yi_243_1 = top_c[14];
                    int take_244_1 = (int)(yk_242_1 > xk_240_1 || yk_242_1 == xk_240_1 && yi_243_1 < xi_241_1);
                    top_b[12] = ((take_244_1 != 0) ? yk_242_1 : xk_240_1);
                    top_c[12] = ((take_244_1 != 0) ? yi_243_1 : xi_241_1);
                    top_b[14] = ((take_244_1 != 0) ? xk_240_1 : yk_242_1);
                    top_c[14] = ((take_244_1 != 0) ? xi_241_1 : yi_243_1);
                    unsigned int xk_245_1 = top_b[13];
                    int xi_246_1 = top_c[13];
                    unsigned int yk_247_1 = top_b[15];
                    int yi_248_1 = top_c[15];
                    int take_249_1 = (int)(yk_247_1 > xk_245_1 || yk_247_1 == xk_245_1 && yi_248_1 < xi_246_1);
                    top_b[13] = ((take_249_1 != 0) ? yk_247_1 : xk_245_1);
                    top_c[13] = ((take_249_1 != 0) ? yi_248_1 : xi_246_1);
                    top_b[15] = ((take_249_1 != 0) ? xk_245_1 : yk_247_1);
                    top_c[15] = ((take_249_1 != 0) ? xi_246_1 : yi_248_1);
                    unsigned int xk_250_1 = top_b[0];
                    int xi_251_1 = top_c[0];
                    unsigned int yk_252_1 = top_b[1];
                    int yi_253_1 = top_c[1];
                    int take_254_1 = (int)(yk_252_1 > xk_250_1 || yk_252_1 == xk_250_1 && yi_253_1 < xi_251_1);
                    top_b[0] = ((take_254_1 != 0) ? yk_252_1 : xk_250_1);
                    top_c[0] = ((take_254_1 != 0) ? yi_253_1 : xi_251_1);
                    top_b[1] = ((take_254_1 != 0) ? xk_250_1 : yk_252_1);
                    top_c[1] = ((take_254_1 != 0) ? xi_251_1 : yi_253_1);
                    unsigned int xk_255_1 = top_b[2];
                    int xi_256_1 = top_c[2];
                    unsigned int yk_257_1 = top_b[3];
                    int yi_258_1 = top_c[3];
                    int take_259_1 = (int)(yk_257_1 > xk_255_1 || yk_257_1 == xk_255_1 && yi_258_1 < xi_256_1);
                    top_b[2] = ((take_259_1 != 0) ? yk_257_1 : xk_255_1);
                    top_c[2] = ((take_259_1 != 0) ? yi_258_1 : xi_256_1);
                    top_b[3] = ((take_259_1 != 0) ? xk_255_1 : yk_257_1);
                    top_c[3] = ((take_259_1 != 0) ? xi_256_1 : yi_258_1);
                    unsigned int xk_260_1 = top_b[4];
                    int xi_261_1 = top_c[4];
                    unsigned int yk_262_1 = top_b[5];
                    int yi_263_1 = top_c[5];
                    int take_264_1 = (int)(yk_262_1 > xk_260_1 || yk_262_1 == xk_260_1 && yi_263_1 < xi_261_1);
                    top_b[4] = ((take_264_1 != 0) ? yk_262_1 : xk_260_1);
                    top_c[4] = ((take_264_1 != 0) ? yi_263_1 : xi_261_1);
                    top_b[5] = ((take_264_1 != 0) ? xk_260_1 : yk_262_1);
                    top_c[5] = ((take_264_1 != 0) ? xi_261_1 : yi_263_1);
                    unsigned int xk_265_1 = top_b[6];
                    int xi_266_1 = top_c[6];
                    unsigned int yk_267_1 = top_b[7];
                    int yi_268_1 = top_c[7];
                    int take_269_1 = (int)(yk_267_1 > xk_265_1 || yk_267_1 == xk_265_1 && yi_268_1 < xi_266_1);
                    top_b[6] = ((take_269_1 != 0) ? yk_267_1 : xk_265_1);
                    top_c[6] = ((take_269_1 != 0) ? yi_268_1 : xi_266_1);
                    top_b[7] = ((take_269_1 != 0) ? xk_265_1 : yk_267_1);
                    top_c[7] = ((take_269_1 != 0) ? xi_266_1 : yi_268_1);
                    unsigned int xk_270_1 = top_b[8];
                    int xi_271_1 = top_c[8];
                    unsigned int yk_272_1 = top_b[9];
                    int yi_273_1 = top_c[9];
                    int take_274_1 = (int)(yk_272_1 > xk_270_1 || yk_272_1 == xk_270_1 && yi_273_1 < xi_271_1);
                    top_b[8] = ((take_274_1 != 0) ? yk_272_1 : xk_270_1);
                    top_c[8] = ((take_274_1 != 0) ? yi_273_1 : xi_271_1);
                    top_b[9] = ((take_274_1 != 0) ? xk_270_1 : yk_272_1);
                    top_c[9] = ((take_274_1 != 0) ? xi_271_1 : yi_273_1);
                    unsigned int xk_275_1 = top_b[10];
                    int xi_276_1 = top_c[10];
                    unsigned int yk_277_1 = top_b[11];
                    int yi_278_1 = top_c[11];
                    int take_279_1 = (int)(yk_277_1 > xk_275_1 || yk_277_1 == xk_275_1 && yi_278_1 < xi_276_1);
                    top_b[10] = ((take_279_1 != 0) ? yk_277_1 : xk_275_1);
                    top_c[10] = ((take_279_1 != 0) ? yi_278_1 : xi_276_1);
                    top_b[11] = ((take_279_1 != 0) ? xk_275_1 : yk_277_1);
                    top_c[11] = ((take_279_1 != 0) ? xi_276_1 : yi_278_1);
                    unsigned int xk_280_1 = top_b[12];
                    int xi_281_1 = top_c[12];
                    unsigned int yk_282_1 = top_b[13];
                    int yi_283_1 = top_c[13];
                    int take_284_1 = (int)(yk_282_1 > xk_280_1 || yk_282_1 == xk_280_1 && yi_283_1 < xi_281_1);
                    top_b[12] = ((take_284_1 != 0) ? yk_282_1 : xk_280_1);
                    top_c[12] = ((take_284_1 != 0) ? yi_283_1 : xi_281_1);
                    top_b[13] = ((take_284_1 != 0) ? xk_280_1 : yk_282_1);
                    top_c[13] = ((take_284_1 != 0) ? xi_281_1 : yi_283_1);
                    unsigned int xk_285_1 = top_b[14];
                    int xi_286_1 = top_c[14];
                    unsigned int yk_287_1 = top_b[15];
                    int yi_288_1 = top_c[15];
                    int take_289_1 = (int)(yk_287_1 > xk_285_1 || yk_287_1 == xk_285_1 && yi_288_1 < xi_286_1);
                    top_b[14] = ((take_289_1 != 0) ? yk_287_1 : xk_285_1);
                    top_c[14] = ((take_289_1 != 0) ? yi_288_1 : xi_286_1);
                    top_b[15] = ((take_289_1 != 0) ? xk_285_1 : yk_287_1);
                    top_c[15] = ((take_289_1 != 0) ? xi_286_1 : yi_288_1);
                    buf_b[0] = 8388607;
                    buf_b[1] = 8388607;
                    buf_b[2] = 8388607;
                    buf_b[3] = 8388607;
                    buf_b[4] = 8388607;
                    buf_b[5] = 8388607;
                    buf_b[6] = 8388607;
                    buf_b[7] = 8388607;
                    buf_c[0] = 2147483647;
                    buf_c[1] = 2147483647;
                    buf_c[2] = 2147483647;
                    buf_c[3] = 2147483647;
                    buf_c[4] = 2147483647;
                    buf_c[5] = 2147483647;
                    buf_c[6] = 2147483647;
                    buf_c[7] = 2147483647;
                }
                if (w < n_tiles && inb != 0) {
                    int lay[8];
                    int ns_1 = ((n_req < split_count) ? n_req : split_count);
                    int q_1 = n_req / split_count;
                    int r_1 = n_req - q_1 * split_count;
                    int ta_1 = q_1 + 1;
                    int tb_1 = q_1 + 1 - 1;
                    int va_1 = ((ta_1 < 4) ? ta_1 : 4);
                    int vb_1 = ((tb_1 < 4) ? tb_1 : 4);
                    int ca_1 = (ta_1 + 4 - 1) / 4;
                    int cb_1 = (tb_1 + 4 - 1) / 4;
                    lay[0] = r_1;
                    lay[1] = va_1;
                    lay[2] = vb_1;
                    lay[3] = ((ca_1 < 16) ? ca_1 : 16);
                    lay[4] = ((cb_1 < 16) ? cb_1 : 16);
                    lay[5] = r_1 * va_1;
                    lay[6] = (ns_1 - r_1) * vb_1;
                    lay[7] = lay[3] * lay[5] + lay[4] * lay[6];
                    if (lay[7] > pool_stride) {
                        status[0] = 2;
                    } else {
                        int in_a = (int)(split_id < lay[0]);
                        int l_idx = ((in_a != 0) ? split_id * lay[1] + w : (split_id - lay[0]) * lay[2] + w);
                        int n_cap = ((in_a != 0) ? lay[3] : lay[4]);
                        int l_cnt = ((in_a != 0) ? lay[5] : lay[6]);
                        int e_base = ((in_a != 0) ? 0 : lay[3] * lay[5]);
                        long long p_off = ((long long)row_off * (long long)pool_stride + (long long)(e_base + l_idx)) * 2;
                        long long l_stride = (long long)l_cnt * 2;
                        int pair[2];
                        int k_bits = 0;
                        if (n_cap > 0) {
                            k_bits = reinterpret_cast<int*>(&top_b[0])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[0];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + p_off + 0) = _iv2;
                            }
                        }
                        if (n_cap > 1) {
                            k_bits = reinterpret_cast<int*>(&top_b[1])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[1];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 2) {
                            k_bits = reinterpret_cast<int*>(&top_b[2])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[2];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 2 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 3) {
                            k_bits = reinterpret_cast<int*>(&top_b[3])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[3];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 3 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 4) {
                            k_bits = reinterpret_cast<int*>(&top_b[4])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[4];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 4 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 5) {
                            k_bits = reinterpret_cast<int*>(&top_b[5])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[5];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 5 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 6) {
                            k_bits = reinterpret_cast<int*>(&top_b[6])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[6];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 6 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 7) {
                            k_bits = reinterpret_cast<int*>(&top_b[7])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[7];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 7 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 8) {
                            k_bits = reinterpret_cast<int*>(&top_b[8])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[8];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 8 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 9) {
                            k_bits = reinterpret_cast<int*>(&top_b[9])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[9];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 9 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 10) {
                            k_bits = reinterpret_cast<int*>(&top_b[10])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[10];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 10 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 11) {
                            k_bits = reinterpret_cast<int*>(&top_b[11])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[11];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 11 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 12) {
                            k_bits = reinterpret_cast<int*>(&top_b[12])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[12];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 12 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 13) {
                            k_bits = reinterpret_cast<int*>(&top_b[13])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[13];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 13 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 14) {
                            k_bits = reinterpret_cast<int*>(&top_b[14])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[14];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 14 * l_stride) + 0) = _iv2;
                            }
                        }
                        if (n_cap > 15) {
                            k_bits = reinterpret_cast<int*>(&top_b[15])[0];
                            pair[0] = k_bits;
                            pair[1] = top_c[15];
                            {
                                int2 _iv2 = make_int2(pair[0 + 0], pair[0 + 1]);
                                *reinterpret_cast<int2*>(pool + (p_off + 15 * l_stride) + 0) = _iv2;
                            }
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int b_1 = (int)blockIdx.x + request_begin;
            int split_id_1 = (int)blockIdx.y;
            int seqlen_1 = seq_lens[b_1];
            int tail_1 = 0;
            if (seqlen_1 % 256 > rank * 128) {
                tail_1 = 1;
            }
            int n_req_1 = seqlen_1 / 256 + tail_1;
            int n_blk_1 = 0;
            if (split_id_1 < n_req_1) {
                n_blk_1 = (n_req_1 - split_id_1 + split_count - 1) / split_count;
            }
            int n_tiles_1 = n_blk_1;
            int qb_1 = query_start_loc[b_1];
            int bt_base = b_1 * block_table_stride;
            int lane_0_1 = (int)lane;
            int ids = 0;
            int ids_next = 0;
            int gi = lane_0_1;
            int bi = split_id_1 + lane_0_1 * split_count;
            if (bi < block_table_stride) {
                ids = block_table[bt_base + bi];
            }
            unsigned int _phase_k_empty = 1;
            if (n_tiles_1 > 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, BLOCK_Q * 512);
                    #pragma unroll
                    for (int w_1 = 0; w_1 < 4; w_1++) {
                        tma_2d_gmem2smem(smem_q_addr + (unsigned int)(w_1 * 32 * 128), (&Q), 0, qb_1 * 4, q_full_addr);
                    }
                }
                unsigned int k_stage = 0;
                int i0_1 = 0;
                int nb = 0;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < n_tiles_1; tile_1++) {
                    i0_1 = tile_1;
                    if (i0_1 % 32 == 0) {
                        gi = i0_1 + 32 + lane_0_1;
                        ids_next = 0;
                        if (gi < n_blk_1) {
                            ids_next = block_table[bt_base + split_id_1 + gi * split_count];
                        }
                    }
                    nb = n_blk_1 - i0_1;
                    if (nb > 1) {
                        nb = 1;
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, ids, i0_1 % 32);
                    mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (k_stage) * 8, (unsigned int)nb * 16384);
                        if (nb > 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 16384), "l"((&K)), "r"(0), "r"(0), "r"(_shfl_0),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    if ((i0_1 + 1) % 32 == 0) {
                        ids = ids_next;
                    }
                    k_stage += 1;
                    if (k_stage == 6) { k_stage = 0; _phase_k_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int b_2 = (int)blockIdx.x + request_begin;
            int split_id_2 = (int)blockIdx.y;
            int seqlen_2 = seq_lens[b_2];
            int tail_2 = 0;
            if (seqlen_2 % 256 > rank * 128) {
                tail_2 = 1;
            }
            int n_req_2 = seqlen_2 / 256 + tail_2;
            int n_blk_2 = 0;
            if (split_id_2 < n_req_2) {
                n_blk_2 = (n_req_2 - split_id_2 + split_count - 1) / split_count;
            }
            int n_tiles_2 = n_blk_2;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_acc_empty = 1;
            if (n_tiles_2 > 0) {
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                unsigned int k_stage_1 = 0;
                unsigned int slot = 0;
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < n_tiles_2; tile_2++) {
                    mbarrier_wait(k_full_addr + (k_stage_1) * 8, _phase_k_full);
                    mbarrier_wait(acc_empty_addr + (slot) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                        }
                    }
                    elect_commit2(acc_full_addr + (slot) * 8, k_empty_addr + (k_stage_1) * 8);
                    k_stage_1 += 1;
                    if (k_stage_1 == 6) { k_stage_1 = 0; _phase_k_full ^= 1; }
                    slot += 1;
                    if (slot == 4) { slot = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
