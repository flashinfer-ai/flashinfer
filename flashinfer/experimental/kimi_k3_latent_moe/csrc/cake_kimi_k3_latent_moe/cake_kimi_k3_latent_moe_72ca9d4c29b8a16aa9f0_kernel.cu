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
#include "cake_kimi_k3_latent_moe_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 11
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 4096
#define SMEM_LAND_STRIDE 4096
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 6144
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 186368
#define SMEM_SMEM_B_STAGE_BYTES 1024
#define SMEM_SMEM_B_STRIDE 1024
#define SMEM_SMEM_BN_OFF 197632
#define SMEM_SMEM_BN_STAGE_BYTES 1024
#define SMEM_SMEM_BN_STRIDE 1024
#define SMEM_SMEM_AG_OFF 6144
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 14336
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 201728

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_72ca9d4c29b8a16aa9f0(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_empty_addr (mbar_base + 88)
    #define acc_full_addr (mbar_base + 176)
    #define acc_empty_addr (mbar_base + 184)
    #define red_bar_addr (mbar_base + 192)
    #define issue_bar_addr (mbar_base + 200)
    #define rows_bar_addr (mbar_base + 208)
    #define bn_bar_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_a_addr = smem + 6144;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 186368);
    const int smem_b_addr = smem + 186368;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_bn_addr = smem + 197632;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_ag_addr = smem + 6144;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 14336);
    const int smem_au_addr = smem + 14336;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 11 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // a_empty: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 192, 4096);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
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
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    int rank_e = 0;
                    {
                        rank_e = g_e % 2;
                    }
                    int c_lo = k1_off + ((rank_e == 0) ? 0 : 4);
                    int c_cnt = ((rank_e == 0) ? 4 : 3);
                    int col_b = lane % 8 * 16;
                    unsigned long long partial_stride = (unsigned long long)num_tokens * 3584;
                    unsigned int zero_w = 0;
                    unsigned int nwp[56];
                    unsigned int xp[112];
                    {
                        #pragma unroll
                        for (int i = 0; i < 14; i++) {
                            int kw = (lane + i * 32) * 8;
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(norm_w + kw + 0);
                                uint4* _vdst_0 = reinterpret_cast<uint4*>(&nwp[i * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
                                }
                            }
                        }
                    }
                    int r_ld = epi_warp;
                    if (r_ld < num_tokens) {
                        unsigned long long nrow_ld = (unsigned long long)r_ld * 3584;
                        {
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 14; i_1++) {
                                int kk0 = (lane + i_1 * 32) * 8;
                                {
                                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(routed + (nrow_ld + (unsigned long long)kk0) + 0);
                                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&xp[i_1 * 4]);
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                            : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    int r_ld_0 = epi_warp + 4;
                    if (r_ld_0 < num_tokens) {
                        unsigned long long nrow_ld_1 = (unsigned long long)r_ld_0 * 3584;
                        {
                            #pragma unroll
                            for (int i_2 = 0; i_2 < 14; i_2++) {
                                int kk0_1 = (lane + i_2 * 32) * 8;
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(routed + (nrow_ld_1 + (unsigned long long)kk0_1) + 0);
                                    uint4* _vdst_2 = reinterpret_cast<uint4*>(&xp[56 + i_2 * 4]);
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                            : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(issue_bar_addr);
                        }
                    }
                    int r = epi_warp;
                    float nacc[112];
                    if (r < num_tokens) {
                        unsigned long long nrow_base = (unsigned long long)r * 3584;
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 56; i_3++) {
                            {
                                float _bf16x2_add_f32_0[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(xp[i_3]), "f"(0.0f), "f"(0.0f));
                                nacc[2 * i_3] = _bf16x2_add_f32_0[0];
                                nacc[2 * i_3 + 1] = _bf16x2_add_f32_0[1];
                            }
                        }
                        {
                            #pragma unroll 1
                            for (int p = 1; p < num_partials; p++) {
                                unsigned long long src_base = (unsigned long long)p * partial_stride + nrow_base;
                                #pragma unroll
                                for (int i_4 = 0; i_4 < 14; i_4++) {
                                    int kk = (lane + i_4 * 32) * 8;
                                    float _vec_load_0[8];
                                    {
                                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(routed + (src_base + (unsigned long long)kk) + 0);
                                        uint4 _vld_3[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_3[_blk] = _vptr_3[_blk];
                                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_3[_pair]));
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int j = 0; j < 8; j++) {
                                        nacc[i_4 * 8 + j] = nacc[i_4 * 8 + j] + _vec_load_0[j];
                                    }
                                }
                            }
                        }
                        float sum_sq = 0.0f;
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 112; i_5++) {
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(nacc[i_5]);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            float sv = _cvt_f32_0;
                            sum_sq += sv * sv;
                        }
                        float _warp_reduce_0 = sum_sq;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                        float total = _warp_reduce_0;
                        float _rsqrt_0 = rsqrtf(total / 3584.0f + eps);
                        float rstd = _rsqrt_0;
                        int own = (r - g_e) % 112;
                        #pragma unroll
                        for (int i_6 = 0; i_6 < 14; i_6++) {
                            int kk2 = (lane + i_6 * 32) * 8;
                            float nvals[8];
                            #pragma unroll
                            for (int jj = 0; jj < 4; jj++) {
                                float _bf16x2_add_f32_1[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(nwp[i_6 * 4 + jj]), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(nacc[i_6 * 8 + jj * 2 + h]);
                                    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                    float s2 = _cvt_f32_1;
                                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(s2 * rstd);
                                    float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                    float normed = _cvt_f32_2;
                                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_bf16x2_add_f32_1[h] * normed);
                                    float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                    nvals[jj * 2 + h] = _cvt_f32_3;
                                }
                            }
                            if (own == 0) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(nvals[0 + 0], nvals[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(nvals[0 + 2], nvals[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(nvals[0 + 4], nvals[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(nvals[0 + 6], nvals[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y_out + (nrow_base + (unsigned long long)kk2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                            int c_rel = i_6 * 4 + lane / 8 - c_lo;
                            if (c_rel >= 0) {
                                if (c_rel < c_cnt) {
                                    uint32_t nvals_bf16[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nvals[_lp*2 + 0], nvals[_lp*2+1 + 0]));
                                        nvals_bf16[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_rel * 1024) + (unsigned int)(r * 128 + col_b ^ (r * 128 + col_b >> 7 & 7) << 4))), "r"(nvals_bf16[0]), "r"(nvals_bf16[1]), "r"(nvals_bf16[2]), "r"(nvals_bf16[3]) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i_7 = 0; i_7 < 14; i_7++) {
                            int c_relz = i_7 * 4 + lane / 8 - c_lo;
                            if (c_relz >= 0) {
                                if (c_relz < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_relz * 1024) + (unsigned int)(r * 128 + col_b ^ (r * 128 + col_b >> 7 & 7) << 4))), "r"(zero_w), "r"(zero_w), "r"(zero_w), "r"(zero_w) : "memory");
                                }
                            }
                        }
                    }
                    int r_1 = epi_warp + 4;
                    float nacc_2[112];
                    if (r_1 < num_tokens) {
                        unsigned long long nrow_base_1 = (unsigned long long)r_1 * 3584;
                        #pragma unroll
                        for (int i_8 = 0; i_8 < 56; i_8++) {
                            {
                                float _bf16x2_add_f32_2[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_2[0]), "=&f"(_bf16x2_add_f32_2[1]) : "r"(xp[56 + i_8]), "f"(0.0f), "f"(0.0f));
                                nacc_2[2 * i_8] = _bf16x2_add_f32_2[0];
                                nacc_2[2 * i_8 + 1] = _bf16x2_add_f32_2[1];
                            }
                        }
                        {
                            #pragma unroll 1
                            for (int p_1 = 1; p_1 < num_partials; p_1++) {
                                unsigned long long src_base_1 = (unsigned long long)p_1 * partial_stride + nrow_base_1;
                                #pragma unroll
                                for (int i_9 = 0; i_9 < 14; i_9++) {
                                    int kk_1 = (lane + i_9 * 32) * 8;
                                    float _vec_load_1[8];
                                    {
                                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(routed + (src_base_1 + (unsigned long long)kk_1) + 0);
                                        uint4 _vld_4[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_4[_blk] = _vptr_4[_blk];
                                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_4[_pair]));
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int j_1 = 0; j_1 < 8; j_1++) {
                                        nacc_2[i_9 * 8 + j_1] = nacc_2[i_9 * 8 + j_1] + _vec_load_1[j_1];
                                    }
                                }
                            }
                        }
                        float sum_sq_1 = 0.0f;
                        #pragma unroll
                        for (int i_10 = 0; i_10 < 112; i_10++) {
                            __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(nacc_2[i_10]);
                            float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                            float sv_1 = _cvt_f32_4;
                            sum_sq_1 += sv_1 * sv_1;
                        }
                        float _warp_reduce_1 = sum_sq_1;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                        float total_1 = _warp_reduce_1;
                        float _rsqrt_1 = rsqrtf(total_1 / 3584.0f + eps);
                        float rstd_1 = _rsqrt_1;
                        int own_1 = (r_1 - g_e) % 112;
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 14; i_11++) {
                            int kk2_1 = (lane + i_11 * 32) * 8;
                            float nvals_1[8];
                            #pragma unroll
                            for (int jj_1 = 0; jj_1 < 4; jj_1++) {
                                float _bf16x2_add_f32_3[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_3[0]), "=&f"(_bf16x2_add_f32_3[1]) : "r"(nwp[i_11 * 4 + jj_1]), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int h_1 = 0; h_1 < 2; h_1++) {
                                    __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(nacc_2[i_11 * 8 + jj_1 * 2 + h_1]);
                                    float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                                    float s2_1 = _cvt_f32_5;
                                    __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(s2_1 * rstd_1);
                                    float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                                    float normed_1 = _cvt_f32_6;
                                    __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_bf16x2_add_f32_3[h_1] * normed_1);
                                    float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                                    nvals_1[jj_1 * 2 + h_1] = _cvt_f32_7;
                                }
                            }
                            if (own_1 == 0) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(nvals_1[0 + 0], nvals_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(nvals_1[0 + 2], nvals_1[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(nvals_1[0 + 4], nvals_1[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(nvals_1[0 + 6], nvals_1[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y_out + (nrow_base_1 + (unsigned long long)kk2_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                            int c_rel_1 = i_11 * 4 + lane / 8 - c_lo;
                            if (c_rel_1 >= 0) {
                                if (c_rel_1 < c_cnt) {
                                    uint32_t nvals_bf16_1[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nvals_1[_lp*2 + 0], nvals_1[_lp*2+1 + 0]));
                                        nvals_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_rel_1 * 1024) + (unsigned int)(r_1 * 128 + col_b ^ (r_1 * 128 + col_b >> 7 & 7) << 4))), "r"(nvals_bf16_1[0]), "r"(nvals_bf16_1[1]), "r"(nvals_bf16_1[2]), "r"(nvals_bf16_1[3]) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i_12 = 0; i_12 < 14; i_12++) {
                            int c_relz_1 = i_12 * 4 + lane / 8 - c_lo;
                            if (c_relz_1 >= 0) {
                                if (c_relz_1 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_relz_1 * 1024) + (unsigned int)(r_1 * 128 + col_b ^ (r_1 * 128 + col_b >> 7 & 7) << 4))), "r"(zero_w), "r"(zero_w), "r"(zero_w), "r"(zero_w) : "memory");
                                }
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(bn_bar_addr);
                        }
                    }
                }
            }
            int tile = g_e;
            int slot = 0;
            {
                tile = g_e / 2;
                slot = g_e % 2;
            }
            int cls = ((tile < 0) ? 0 : ((tile < 56) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = tile * 128;
            int row_s = (tile - 56) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int epi_stage = 0;
            float red[8];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (epi_stage) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], lane_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_3[0];
                red[1] = _tmem_load_3[1];
                red[2] = _tmem_load_3[2];
                red[3] = _tmem_load_3[3];
                red[4] = _tmem_load_3[4];
                red[5] = _tmem_load_3[5];
                red[6] = _tmem_load_3[6];
                red[7] = _tmem_load_3[7];
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr + (epi_stage) * 8);
                }
            }
            if (slot == 1) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(red_bar_addr), "r"(0));
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mapa_0) : "memory");
                fin = 0;
            } else {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(red_bar_addr);
                    }
                }
                mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
                red[0] = red[0] + land[tid_1 * 4];
                red[1] = red[1] + land[tid_1 * 4 + 1];
                red[2] = red[2] + land[tid_1 * 4 + 2];
                red[3] = red[3] + land[tid_1 * 4 + 3];
                red[4] = red[4] + land[(128 + tid_1) * 4];
                red[5] = red[5] + land[(128 + tid_1) * 4 + 1];
                red[6] = red[6] + land[(128 + tid_1) * 4 + 2];
                red[7] = red[7] + land[(128 + tid_1) * 4 + 3];
            }
            if (fin == 1) {
                {
                    if (num_tokens > 0) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[0]);
                        }
                    }
                    if (num_tokens > 1) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (7168 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[1]);
                        }
                    }
                    if (num_tokens > 2) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (14336 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[2]);
                        }
                    }
                    if (num_tokens > 3) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (21504 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[3]);
                        }
                    }
                    if (num_tokens > 4) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (28672 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[4]);
                        }
                    }
                    if (num_tokens > 5) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (35840 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[5]);
                        }
                    }
                    if (num_tokens > 6) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (43008 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[6]);
                        }
                    }
                    if (num_tokens > 7) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (50176 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[7]);
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
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
            {
                tile_l = g_l / 2;
                rank_l = g_l % 2;
            }
            int d_lo_l = ((rank_l == 0) ? 0 : 6);
            int d_cnt_l = ((rank_l == 0) ? 6 : 6);
            int u_lo_l = ((rank_l == 0) ? 0 : 4);
            int u_cnt_l = ((rank_l == 0) ? 4 : 3);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int _min_0 = ((11) < (u_count_l) ? (11) : (u_count_l));
            int _min_1 = ((3) < (u_count_l) ? (3) : (u_count_l));
            int pro = ((0) ? _min_0 : ((1) ? _min_1 : 0));
            {
                if (elect_sync()) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_R))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_L))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_S))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_2))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_1))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_2))) : "memory");
                }
            }
            {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int slot_1 = 0; slot_1 < pro; slot_1++) {
                        int sa = (int)slot_1;
                        int tile_a = tile_l;
                        int m_a = ((sa < d_cnt_l) ? d_lo_l + sa : 12 + u_lo_l + (sa - d_cnt_l));
                        int cls_a = ((tile_a < 0) ? 0 : ((tile_a < 56) ? 1 : 2));
                        int row_ra = tile_a * 128;
                        int row_la = tile_a * 128;
                        int row_sa = (tile_a - 56) * 64;
                        if (m_a < 12) {
                            int ka2 = m_a;
                            tma_3d_gmem2smem(smem_a_addr + slot_1 * 16384, (&A_2), 0, row_la, ka2, a_full_addr + (slot_1) * 8);
                        } else {
                            int ka1 = k1_off + (m_a - 12);
                            tma_3d_gmem2smem(smem_a_addr + slot_1 * 16384, (&A_L), 0, row_la, ka1, a_full_addr + (slot_1) * 8);
                        }
                    }
                }
                {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                }
            }
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = ((0) ? 1 : 0);
                {
                    mbarrier_wait(issue_bar_addr, 0);
                }
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : 12 + u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 0) ? 0 : ((tile_1 < 56) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = tile_1 * 128;
                    int row_s_1 = (tile_1 - 56) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    if (m < 12) {
                        int kb2 = m;
                        tma_3d_gmem2smem(smem_b_addr + stage * 1024, (&B_2), 0, 0, kb2, a_full_addr + (stage) * 8);
                    } else if (!1) {
                        if (norm_ok == 0) {
                            mbarrier_wait(rows_bar_addr, 0);
                            asm volatile("fence.release.gpu;" ::: "memory");
                            uint32_t _atomic_inc_old_0;
                            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                                : "=r"(_atomic_inc_old_0) : "l"(&counters[0]), "r"(static_cast<uint32_t>(111)) : "memory");
                            unsigned int old = _atomic_inc_old_0;
                            if ((int)old != 111) {
                                #pragma unroll 1
                                for (int _spin = 0; _spin < 4194304; _spin++) {
                                    uint32_t _relaxed_ld_0;
                                    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(counters + 0) : "memory");
                                    unsigned int cnt = _relaxed_ld_0;
                                    if (cnt == 0) {
                                        break;
                                    }
                                }
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                            asm volatile("fence.proxy.async;");
                            norm_ok = 1;
                        }
                        int kb = k1_off + (m - 12);
                        tma_3d_gmem2smem(smem_b_addr + stage * 1024, (&B_1), 0, 0, kb, a_full_addr + (stage) * 8);
                    }
                    if (need_a == 1) {
                        if (m < 12) {
                            int kc2 = m;
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_2), 0, row_l_1, kc2, a_full_addr + (stage) * 8);
                        } else {
                            int kc = k1_off + (m - 12);
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_L), 0, row_l_1, kc, a_full_addr + (stage) * 8);
                        }
                    }
                    {
                        int txb = ((m < 12) ? 17408 : 16384);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, txb);
                    }
                    stage += 1;
                    if (stage == 11) { stage = 0; _phase_a_empty ^= 1; }
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
            {
                tile_2 = g_m / 2;
                rank_m = g_m % 2;
            }
            int u_count_m = ((rank_m == 0) ? 10 : 9);
            int d_cnt_m = ((rank_m == 0) ? 6 : 6);
            unsigned int stage_1 = 0;
            unsigned int epi_stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 0) ? 0 : ((tile_2 < 56) ? 1 : 2));
                mbarrier_wait(acc_empty_addr + (epi_stage_1) * 8, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    if (m_1 == d_cnt_m) {
                        mbarrier_wait(bn_bar_addr, 0);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    {
                        int init_sub = ((1) ? init_flag : 0);
                        if (d_cnt_m > m_1) {
                            int _mma_a_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 64;
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
                    "mov.b32 id, 134349968;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
                        } else {
                            unsigned int ub = (unsigned int)(m_1 - d_cnt_m);
                            int _mma_a_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_4 = (((smem_bn_addr) >> 4) & 0x3FFF) + (ub) * 64;
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
                    "mov.b32 id, 134349968;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
                        }
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 11) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr + (epi_stage_1) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
