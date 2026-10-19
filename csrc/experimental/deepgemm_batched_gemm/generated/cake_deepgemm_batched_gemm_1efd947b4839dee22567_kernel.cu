/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_batched_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 256
#define TMEM_SF_B_OFFSET 260
#define NUM_MAIN_PIPE_STAGES 5
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_CD_OFF 0
#define SMEM_CD_STAGE_BYTES 16384
#define SMEM_CD_STRIDE 16384
#define SMEM_A_OFF 32768
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 114688
#define SMEM_B_STAGE_BYTES 16384
#define SMEM_B_STRIDE 16384
#define SMEM_SFA_OFF 196608
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 199168
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_TOTAL 201984
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_batched_gemm_1efd947b4839dee22567(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, unsigned int* __restrict__ SFD, unsigned int M, unsigned int grid_m, unsigned long long sfd_stride, float alpha)
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

    const int mbar_base = smem + 201728;
    #define full_addr (mbar_base + 0)
    #define sf_full_addr (mbar_base + 40)
    #define empty_addr (mbar_base + 80)
    #define acc_full_addr (mbar_base + 120)
    #define acc_empty_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* cd = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_CD_OFF);
    const int cd_addr = smem + SMEM_CD_OFF;
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_A_OFF);
    const int a_addr = smem + SMEM_A_OFF;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_B_OFF);
    const int b_addr = smem + SMEM_B_OFF;
    unsigned int* sfa = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SFA_OFF);
    const int sfa_addr = smem + SMEM_SFA_OFF;
    unsigned int* sfb = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SFB_OFF);
    const int sfb_addr = smem + SMEM_SFB_OFF;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[201728..201880)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // full: 5 barriers, init_count=33
            mbarrier_init(smem + 201728, 33);
            mbarrier_init(smem + 201736, 33);
            mbarrier_init(smem + 201744, 33);
            mbarrier_init(smem + 201752, 33);
            mbarrier_init(smem + 201760, 33);
            // sf_full: 5 barriers, init_count=1
            mbarrier_init(smem + 201768, 1);
            mbarrier_init(smem + 201776, 1);
            mbarrier_init(smem + 201784, 1);
            mbarrier_init(smem + 201792, 1);
            mbarrier_init(smem + 201800, 1);
            // empty: 5 barriers, init_count=1
            mbarrier_init(smem + 201808, 1);
            mbarrier_init(smem + 201816, 1);
            mbarrier_init(smem + 201824, 1);
            mbarrier_init(smem + 201832, 1);
            mbarrier_init(smem + 201840, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 201848, 1);
            mbarrier_init(smem + 201856, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 201864, 128);
            mbarrier_init(smem + 201872, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 264 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 201880);
    if (warp == 2) {
        int _tmem_hold = smem + 201880;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 256;
    const int tmem_sf_b = taddr + 260;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < 8 * grid_m * 8; tile += num_bids) {
                unsigned int head = (unsigned int)tile / (grid_m * 8);
                unsigned int local = (unsigned int)tile % (grid_m * 8);
                unsigned int off_m = local / 8 * 128;
                unsigned int off_n = local % 8 * 128;
                #pragma unroll 4
                for (int kt = 0; kt < 32; kt++) {
                    mbarrier_wait(empty_addr + (stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        unsigned int sf_bytes = 0;
                        if (kt % 4 == 0) {
                            tma_2d_gmem2smem(sfa_addr + stage * 512, (&SFA), off_m, head * 8 + (unsigned int)(kt / 4), sf_full_addr + (stage) * 8);
                            tma_2d_gmem2smem(sfb_addr + stage * 512, (&SFB), off_n, head * 8 + (unsigned int)(kt / 4), sf_full_addr + (stage) * 8);
                            sf_bytes = 1024;
                        }
                        mbarrier_arrive_expect_tx(sf_full_addr + (stage) * 8, sf_bytes);
                        tma_3d_gmem2smem(a_addr + stage * 16384, (&A), kt * 128, off_m, head, full_addr + (stage) * 8);
                        tma_3d_gmem2smem(b_addr + stage * 16384, (&B), kt * 128, off_n, head, full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 32768);
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < 8 * grid_m * 8; tile_1 += num_bids) {
                mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 4
                for (int kt_1 = 0; kt_1 < 32; kt_1++) {
                    mbarrier_wait(full_addr + (stage_1) * 8, _phase_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        if (kt_1 % 4 == 0) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (stage_1) * 32)));
                            tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (stage_1) * 32)));
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        int _mma_a_lo_0 = (((a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _mma_b_lo_0 = (((b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, ((kt_1 == 0) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 2, b_desc + 2,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 4, b_desc + 4,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 6, b_desc + 6,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, 1);
                        }
                    }
                    __syncwarp();
                    elect_commit(empty_addr + (stage_1) * 8);
                    if (kt_1 == 31) {
                        elect_commit(acc_full_addr + (acc_stage) * 8);
                    }
                    __syncwarp();
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_full ^= 1; }
                }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_empty ^= 1; }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int stage_2 = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (int tile_2 = bid; tile_2 < 8 * grid_m * 8; tile_2 += num_bids) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < 32; kt_2++) {
                    mbarrier_wait(sf_full_addr + (stage_2) * 8, _phase_sf_full);
                    if (kt_2 % 4 == 0) {
                        unsigned int _sf_v[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v[3])));
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v_0[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v_0[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v_0[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v_0[3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(full_addr + (stage_2) * 8);
                    stage_2 += 1;
                    if (stage_2 == 5) { stage_2 = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: store ----
    if (warp >= 4 && warp <= 7) {
        { // store_main
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            unsigned int warp_0 = warp - 4;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int tile_3 = bid; tile_3 < 8 * grid_m * 8; tile_3 += num_bids) {
                unsigned int head_1 = (unsigned int)tile_3 / (grid_m * 8);
                unsigned int local_1 = (unsigned int)tile_3 % (grid_m * 8);
                unsigned int off_m_1 = local_1 / 8 * 128;
                unsigned int off_n_1 = local_1 % 8 * 128;
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int atom = 0; atom < 2; atom++) {
                    if (warp == 4) {
                        asm volatile("cp.async.bulk.wait_group 1;");
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    unsigned int row = off_m_1 + warp_0 * 32 + (unsigned int)lane;
                    unsigned int sf_word = 0;
                    #pragma unroll
                    for (int chunk = 0; chunk < 8; chunk++) {
                        unsigned int col = atom * 64 + chunk * 8;
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], (unsigned int)tmem_acc + (acc_stage_1 * 128 + col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            float2 _f2_0 = make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]);
                            float2 _f2_1 = make_float2(alpha, alpha);
                            float2 _mul_f32x2_0;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
                            float2 scaled = _mul_f32x2_0;
                            _tmem_load_0[pair * 2] = scaled.x;
                            _tmem_load_0[pair * 2 + 1] = scaled.y;
                        }
                        if (atom == 1 && chunk == 7) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(acc_empty_addr + (acc_stage_1) * 8);
                        }
                        unsigned int smem_row = store_stage * 128 + warp_0 * 32 + (unsigned int)lane;
                        unsigned int packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((cd_addr + (smem_row * 128 + (unsigned int)(chunk * 16) ^ (smem_row * 128 + (unsigned int)(chunk * 16) >> 7 & 7) << 4))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_3d((&D), off_n_1 + (unsigned int)(atom * 64), off_m_1, head_1, cd_addr + store_stage * 16384);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    store_stage = (store_stage + 1) % 2;
                    __syncwarp();
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.sync 0, 256;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
    }

    // Cleanup
}

} // extern "C"
