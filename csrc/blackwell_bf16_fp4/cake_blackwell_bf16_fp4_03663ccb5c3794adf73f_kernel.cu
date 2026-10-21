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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_blackwell_bf16_fp4_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 8
#define NUM_PACK_PIPE_STAGES 8
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_SF_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 24576
#define SMEM_SMEM_B_CVT_OFF 17408
#define SMEM_SMEM_B_CVT_STAGE_BYTES 8192
#define SMEM_SMEM_B_CVT_STRIDE 24576
#define SMEM_SMEM_BPACK_OFF 197632
#define SMEM_SMEM_BPACK_STAGE_BYTES 2048
#define SMEM_SMEM_BPACK_STRIDE 2048
#define SMEM_SMEM_SF_OFF 214016
#define SMEM_SMEM_SF_STAGE_BYTES 1024
#define SMEM_SMEM_SF_STRIDE 1024
#define SMEM_TOTAL 218112
#define num_tiles (grid_m * grid_n)

extern "C" {

__global__ __launch_bounds__(512) __cluster_dims__(2,1,1) void
kernel_cake_blackwell_bf16_fp4_03663ccb5c3794adf73f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K, int grid_m, int grid_n)
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
    #define smem_a_full_addr (mbar_base + 0)
    #define smem_a_done_addr (mbar_base + 64)
    #define bsrc_full_addr (mbar_base + 128)
    #define bsrc_done_addr (mbar_base + 192)
    #define smem_b_full_addr (mbar_base + 256)
    #define smem_b_done_addr (mbar_base + 320)
    #define sf_full_addr (mbar_base + 384)
    #define sf_done_addr (mbar_base + 416)
    #define mainloop_done_addr (mbar_base + 448)
    #define epilogue_done_addr (mbar_base + 464)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b_cvt = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_CVT_OFF);
    const int smem_b_cvt_addr = smem + SMEM_SMEM_B_CVT_OFF;
    uint8_t* smem_bpack = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_BPACK_OFF);
    const int smem_bpack_addr = smem + SMEM_SMEM_BPACK_OFF;
    uint8_t* smem_sf = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SF_OFF);
    const int smem_sf_addr = smem + SMEM_SMEM_SF_OFF;
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 6) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 6) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_descale))) : "memory"); }

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 60 barriers)
    // Mbarriers at smem_raw[0..480)

    if (warp == 0) {
        // --- pipeline 'tma_pipe' ---
        // smem_a_full: 8 barriers, init_count=2
        // smem_a_done: 8 barriers, init_count=1
        // --- pipeline 'pack_pipe' ---
        // bsrc_full: 8 barriers, init_count=1
        // bsrc_done: 8 barriers, init_count=2
        // --- pipeline 'tma_pipe' ---
        // smem_b_full: 8 barriers, init_count=4
        // smem_b_done: 8 barriers, init_count=1
        // --- pipeline 'sf_pipe' ---
        // sf_full: 4 barriers, init_count=1
        // sf_done: 4 barriers, init_count=8
        // --- pipeline 'mainloop_pipe' ---
        // mainloop_done: 2 barriers, init_count=1
        // epilogue_done: 2 barriers, init_count=8
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 2;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(24), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(8), "r"((uint32_t)(2)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 8;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(26), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(24), "r"((uint32_t)(8)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(4)));
        if (lane < 28) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 480);
    if (warp == 0) {
        int _tmem_hold = smem + 480;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    // Partial post-allocation TMEM rendezvous (512 threads on named barrier 7)
    if (warp <= 15) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(7), "r"(512) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp <= 15) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            float alpha_value = 1.0f;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            alpha_value = alpha[0];
            int vec_ok = ((N % 16 == 0) ? 1 : 0);
            float vals[16];
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < num_tiles; tile_idx += num_clusters) {
                int m_tile = tile_idx / (unsigned int)grid_n;
                int n_tile = tile_idx % (unsigned int)grid_n;
                int off_m = m_tile * 256;
                int tmem_row = epi_warp * 32;
                off_m = off_m + cta_rank * 128;
                tmem_row = tmem_row + cta_rank * 128;
                int off_n = n_tile * 128;
                int row_global = off_m + epi_tid;
                long long row_linear = (long long)row_global * (long long)N;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int n_chunk = 0; n_chunk < 2; n_chunk++) {
                    int tmem_col = epi_stage * 128 + (unsigned int)(n_chunk * 64);
                    int tmem_addr = taddr + (unsigned int)(tmem_row << 16) + (unsigned int)tmem_col;
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(tmem_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(tmem_addr + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int sub16 = 0; sub16 < 4; sub16++) {
                        int col_global = off_n + n_chunk * 64 + sub16 * 16;
                        if (row_global < M && col_global < N) {
                            #pragma unroll
                            for (int i = 0; i < 16; i++) {
                                vals[i] = _tmem_load_0[sub16 * 16 + i] * alpha_value;
                            }
                            if (vec_ok == 1 && col_global + 16 <= N) {
                                {
                                    {
                                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(vals[0 + 8], vals[0 + 9]);
                                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(vals[0 + 10], vals[0 + 11]);
                                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(vals[0 + 12], vals[0 + 13]);
                                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(vals[0 + 14], vals[0 + 15]);
                                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(&((__nv_bfloat16*)(C + (row_linear + (long long)col_global)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                    }
                                }
                            } else {
                                #pragma unroll
                                for (int i_1 = 0; i_1 < 16; i_1++) {
                                    if (col_global + i_1 < N) {
                                        float value = vals[i_1];
                                        *(reinterpret_cast<__nv_bfloat16*>(C + (row_linear + (long long)col_global + (long long)i_1)) + (0)) = __float2bfloat16_rn(value);
                                    }
                                }
                            }
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            int num_iters = K / 64;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_smem_a_full = 0;
            unsigned int _phase_smem_b_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < num_tiles; tile_idx_1 += num_clusters) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
                        mbarrier_wait(smem_a_full_addr + (mma_tma_stage) * 8, _phase_smem_a_full);
                        mbarrier_wait(smem_b_full_addr + (mma_tma_stage) * 8, _phase_smem_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 1536;
                        int _mma_b_lo_0 = (((smem_b_cvt_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 1536;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 128))), "r"(((init_flag) ? 0 : 1)));
                        elect_commit_cg2_multicast(smem_a_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(smem_b_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 8) { mma_tma_stage = 0; _phase_smem_a_full ^= 1; _phase_smem_b_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            unsigned int load_a_stage = 0;
            int num_iters_1 = K / 64;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_smem_a_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < num_tiles; tile_idx_2 += num_clusters) {
                    int m_tile_1 = tile_idx_2 / (unsigned int)grid_n;
                    int off_m_1 = m_tile_1 * 256;
                    off_m_1 = off_m_1 + cta_rank * 128;
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_iters_1; iter_k_1++) {
                        mbarrier_wait(smem_a_done_addr + (load_a_stage) * 8, _phase_smem_a_done);
                        int a_k = iter_k_1 * 64;
                        tma_2d_gmem2smem_cta2(smem_a_addr + load_a_stage * 24576, (&A), a_k, off_m_1, ((smem_a_full_addr + (load_a_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((smem_a_full_addr + (load_a_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        load_a_stage += 1;
                        if (load_a_stage == 8) { load_a_stage = 0; _phase_smem_a_done ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 6) {
        { // load_b_main
            unsigned int load_b_stage = 0;
            unsigned int sf_stage = 0;
            int num_groups = K / 256;
            unsigned int _phase_sf_done = 1;
            unsigned int _phase_bsrc_done = 1;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_descale))) : "memory");
                #pragma unroll 1
                for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < num_tiles; tile_idx_3 += num_clusters) {
                    int n_tile_1 = tile_idx_3 % (unsigned int)grid_n;
                    int b_row = n_tile_1 * 128;
                    b_row = b_row + cta_rank * 64;
                    #pragma unroll 1
                    for (int group = 0; group < num_groups; group++) {
                        mbarrier_wait(sf_done_addr + (sf_stage) * 8, _phase_sf_done);
                        tma_2d_gmem2smem(smem_sf_addr + sf_stage * 1024, (&B_descale), b_row, group * 16, sf_full_addr + (sf_stage) * 8);
                        mbarrier_arrive_expect_tx(sf_full_addr + (sf_stage) * 8, 1024);
                        sf_stage += 1;
                        if (sf_stage == 4) { sf_stage = 0; _phase_sf_done ^= 1; }
                        #pragma unroll 1
                        for (int sub = 0; sub < 4; sub++) {
                            mbarrier_wait(bsrc_done_addr + (load_b_stage) * 8, _phase_bsrc_done);
                            tma_2d_gmem2smem(smem_bpack_addr + load_b_stage * 2048, (&B), b_row * 2, (group * 4 + sub) * 4, bsrc_full_addr + (load_b_stage) * 8);
                            mbarrier_arrive_expect_tx(bsrc_full_addr + (load_b_stage) * 8, 2048);
                            load_b_stage += 1;
                            if (load_b_stage == 8) { load_b_stage = 0; _phase_bsrc_done ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int warp_id_in_role = (warp - 8);
            int group_1 = warp_id_in_role / 2;
            int group_lead = group_1 * 2;
            unsigned int cvt_stage = (unsigned int)group_1;
            unsigned int cvt_pack_stage = (unsigned int)group_1;
            unsigned int cvt_sf_stage = 0;
            int kgroup_first = 0;
            int num_groups_1 = K / 256;
            int convert_tid = (warp_id_in_role - group_lead) * 32 + lane;
            int row0 = convert_tid / 4;
            int k_block = convert_tid - row0 * 4;
            int swz = row0 & 7;
            int chunk_lo = k_block * 2 ^ swz;
            int chunk_hi = k_block * 2 + 1 ^ swz;
            unsigned int raw_words[2];
            unsigned int scale_word[1];
            unsigned int raw8[8];
            int cute_j = convert_tid & 15;
            int cute_unit0 = convert_tid >> 4;
            int cute_swz = cute_j & 7;
            int cute_sf_shift = (cute_j & 3) * 8;
            unsigned int _phase_sf_full = 0;
            unsigned int _phase_bsrc_full = 0;
            unsigned int _phase_smem_b_done = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < num_tiles; tile_idx_4 += num_clusters) {
                #pragma unroll 1
                for (int kgroup = kgroup_first; kgroup < num_groups_1; kgroup++) {
                    mbarrier_wait(sf_full_addr + (cvt_sf_stage) * 8, _phase_sf_full);
                    int sf_base = smem_sf_addr + cvt_sf_stage * 1024;
                    #pragma unroll 1
                    for (int pass_idx = 0; pass_idx < 1; pass_idx++) {
                        int sub_1 = pass_idx * 4 + group_1 & 3;
                        mbarrier_wait(bsrc_full_addr + (cvt_pack_stage) * 8, _phase_bsrc_full);
                        mbarrier_wait(smem_b_done_addr + (cvt_stage) * 8, _phase_smem_b_done);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int packed_base = smem_bpack_addr + cvt_pack_stage * 2048;
                        int bcvt_base = smem_b_cvt_addr + cvt_stage * 24576;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw8[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw8[(0) + 3]))
                            : "r"(packed_base + (cute_unit0 & 3) * 512 + (cute_unit0 >> 2) * 512 + cute_j * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw8[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw8[(4) + 3]))
                            : "r"(packed_base + (cute_unit0 & 3) * 512 + (cute_unit0 >> 2) * 512 + cute_j * 32 + 16));
                        uint32_t _prmt_b32_0;
                        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_b32_0) : "r"(raw8[0]), "r"(raw8[2]));
                        uint32_t _prmt_b32_1;
                        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_b32_1) : "r"(raw8[0]), "r"(raw8[2]));
                        uint32_t _prmt_b32_2;
                        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_b32_2) : "r"(raw8[4]), "r"(raw8[6]));
                        uint32_t _prmt_b32_3;
                        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_b32_3) : "r"(raw8[4]), "r"(raw8[6]));
                        uint32_t _prmt_b32_4;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_4) : "r"(_prmt_b32_0), "r"(_prmt_b32_2));
                        uint32_t _prmt_b32_5;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_5) : "r"(_prmt_b32_0), "r"(_prmt_b32_2));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + (sub_1 * 4 + (cute_unit0 & 3)) * 64 + (cute_unit0 >> 2) * 64 + (cute_j & 12)));
                        uint32_t _fp4_dequant_block16_bf16_0[8];
                        {
                            uint32_t _scale_byte = ((uint32_t)((uint8_t)(scale_word[0] >> (unsigned int)cute_sf_shift & 255))) & 0xFFu;
                            uint32_t _scale_f16x2;
                            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                            uint32_t _scale_bf16x2;
                            uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                            uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                            float _scale_f0;
                            float _scale_f1;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_scale_f1), "f"(_scale_f0));
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_4)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_4)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_4)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_4)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_5)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_5)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_5)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_5)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j) * 128 + (((cute_unit0 & 3) * 2 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j) * 128 + (((cute_unit0 & 3) * 2 + 1 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(4) + 3])));
                        uint32_t _prmt_b32_6;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_6) : "r"(_prmt_b32_1), "r"(_prmt_b32_3));
                        uint32_t _prmt_b32_7;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_7) : "r"(_prmt_b32_1), "r"(_prmt_b32_3));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + (sub_1 * 4 + (cute_unit0 & 3)) * 64 + (cute_unit0 >> 2) * 64 + (cute_j & 12) + 16));
                        uint32_t _fp4_dequant_block16_bf16_1[8];
                        {
                            uint32_t _scale_byte = ((uint32_t)((uint8_t)(scale_word[0] >> (unsigned int)cute_sf_shift & 255))) & 0xFFu;
                            uint32_t _scale_f16x2;
                            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                            uint32_t _scale_bf16x2;
                            uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                            uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                            float _scale_f0;
                            float _scale_f1;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_scale_f1), "f"(_scale_f0));
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_6)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_6)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_6)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_6)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_7)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_7)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_7)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_7)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j + 16) * 128 + (((cute_unit0 & 3) * 2 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j + 16) * 128 + (((cute_unit0 & 3) * 2 + 1 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(4) + 3])));
                        uint32_t _prmt_b32_8;
                        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_b32_8) : "r"(raw8[1]), "r"(raw8[3]));
                        uint32_t _prmt_b32_9;
                        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_b32_9) : "r"(raw8[1]), "r"(raw8[3]));
                        uint32_t _prmt_b32_10;
                        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_b32_10) : "r"(raw8[5]), "r"(raw8[7]));
                        uint32_t _prmt_b32_11;
                        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_b32_11) : "r"(raw8[5]), "r"(raw8[7]));
                        uint32_t _prmt_b32_12;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_12) : "r"(_prmt_b32_8), "r"(_prmt_b32_10));
                        uint32_t _prmt_b32_13;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_13) : "r"(_prmt_b32_8), "r"(_prmt_b32_10));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + (sub_1 * 4 + (cute_unit0 & 3)) * 64 + (cute_unit0 >> 2) * 64 + (cute_j & 12) + 32));
                        uint32_t _fp4_dequant_block16_bf16_2[8];
                        {
                            uint32_t _scale_byte = ((uint32_t)((uint8_t)(scale_word[0] >> (unsigned int)cute_sf_shift & 255))) & 0xFFu;
                            uint32_t _scale_f16x2;
                            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                            uint32_t _scale_bf16x2;
                            uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                            uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                            float _scale_f0;
                            float _scale_f1;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_scale_f1), "f"(_scale_f0));
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_12)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_12)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_12)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_12)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_13)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_13)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_13)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_13)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j + 32) * 128 + (((cute_unit0 & 3) * 2 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j + 32) * 128 + (((cute_unit0 & 3) * 2 + 1 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(4) + 3])));
                        uint32_t _prmt_b32_14;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_14) : "r"(_prmt_b32_9), "r"(_prmt_b32_11));
                        uint32_t _prmt_b32_15;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_15) : "r"(_prmt_b32_9), "r"(_prmt_b32_11));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + (sub_1 * 4 + (cute_unit0 & 3)) * 64 + (cute_unit0 >> 2) * 64 + (cute_j & 12) + 48));
                        uint32_t _fp4_dequant_block16_bf16_3[8];
                        {
                            uint32_t _scale_byte = ((uint32_t)((uint8_t)(scale_word[0] >> (unsigned int)cute_sf_shift & 255))) & 0xFFu;
                            uint32_t _scale_f16x2;
                            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                            uint32_t _scale_bf16x2;
                            uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                            uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                            float _scale_f0;
                            float _scale_f1;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_scale_f1), "f"(_scale_f0));
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_14)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_14)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_14)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_14)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_15)) >> 0));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_15)) >> 8));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_15)) >> 16));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                            {
                                uint32_t _fp4_bf16x2 = cvt_bf16x2_e2m1x2((((uint32_t)(_prmt_b32_15)) >> 24));
                                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j + 48) * 128 + (((cute_unit0 & 3) * 2 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(bcvt_base + ((cute_unit0 >> 2) * 64 + cute_j + 48) * 128 + (((cute_unit0 & 3) * 2 + 1 ^ cute_swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(4) + 3])));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(bsrc_done_addr + (cvt_pack_stage) * 8);
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((smem_b_full_addr + (cvt_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        cvt_stage += 1;
                        if (cvt_stage == 8) { cvt_stage = 0; _phase_smem_b_done ^= 1; }
                        cvt_stage += 1;
                        if (cvt_stage == 8) { cvt_stage = 0; _phase_smem_b_done ^= 1; }
                        cvt_stage += 1;
                        if (cvt_stage == 8) { cvt_stage = 0; _phase_smem_b_done ^= 1; }
                        cvt_stage += 1;
                        if (cvt_stage == 8) { cvt_stage = 0; _phase_smem_b_done ^= 1; }
                        cvt_pack_stage += 1;
                        if (cvt_pack_stage == 8) { cvt_pack_stage = 0; _phase_bsrc_full ^= 1; }
                        cvt_pack_stage += 1;
                        if (cvt_pack_stage == 8) { cvt_pack_stage = 0; _phase_bsrc_full ^= 1; }
                        cvt_pack_stage += 1;
                        if (cvt_pack_stage == 8) { cvt_pack_stage = 0; _phase_bsrc_full ^= 1; }
                        cvt_pack_stage += 1;
                        if (cvt_pack_stage == 8) { cvt_pack_stage = 0; _phase_bsrc_full ^= 1; }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(sf_done_addr + (cvt_sf_stage) * 8);
                    }
                    cvt_sf_stage += 1;
                    if (cvt_sf_stage == 4) { cvt_sf_stage = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
