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
#include "cake_mla_nvfp4_paged_decode_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_W_OFF 0
#define SMEM_SMEM_W_STAGE_BYTES 8192
#define SMEM_SMEM_W_STRIDE 8192
#define SMEM_TOTAL 8192
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_mla_nvfp4_paged_decode_b2e4c5513e1d7e5618c0(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, __nv_bfloat16* __restrict__ O, float* __restrict__ lse, int* __restrict__ cum_seq_lens_q, int batch, int num_heads, int num_split, float bmm2_scale, float lse_bias, int has_lse)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_w = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_W_OFF);
    const int smem_w_addr = smem + SMEM_SMEM_W_OFF;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int part = 0;
    int row = blockIdx.x * 8 + warp;
    int rows_total = cum_seq_lens_q[batch] * num_heads;
    if (row < rows_total) {
        int stat_base = row * num_split;
        int w_base = warp * 256;
        float m_loc = -CAKE_INF;
        for (int s = lane; s < num_split; s += 32) {
            float m_s = partial_max[stat_base + s];
            float _max_0 = max_noftz(m_loc, m_s);
            m_loc = _max_0;
        }
        float _warp_reduce_0 = m_loc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float max_m = _warp_reduce_0;
        float w_loc = 0.0f;
        for (int s_1 = lane; s_1 < num_split; s_1 += 32) {
            float m_s2 = partial_max[stat_base + s_1];
            float sum_s = partial_sum[stat_base + s_1];
            float w_s = 0.0f;
            if (sum_s > 0.0f) {
                float _exp2_0 = approx_exp2(m_s2 - max_m);
                w_s = _exp2_0 * sum_s;
            }
            smem_w[w_base + s_1] = w_s;
            w_loc = w_loc + w_s;
        }
        float _warp_reduce_1 = w_loc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float sum_w = _warp_reduce_1;
        __syncwarp();
        float inv_sum = 0.0f;
        if (sum_w > 0.0f) {
            float _rcp_0 = approx_rcp(sum_w);
            inv_sum = _rcp_0 * bmm2_scale;
        }
        int d0 = part * 512 + lane * 16;
        int last_split = num_split - 1;
        float pf[64];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (last_split >= j) {
                #pragma unroll
                for (int q = 0; q < 16; q += 8) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (stat_base + j) * 512 + d0 + q);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pf[j * 16 + q + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&pf[j * 16 + q + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
        }
        float acc[16];
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            acc[e] = 0.0f;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 4; j_1++) {
            float w_raw_j = smem_w[w_base + j_1];
            float w_j = ((last_split >= j_1) ? w_raw_j : 0.0f);
            #pragma unroll
            for (int e_1 = 0; e_1 < 16; e_1++) {
                float c_j = w_j * pf[j_1 * 16 + e_1];
                acc[e_1] = acc[e_1] + ((w_j > 0.0f) ? c_j : 0.0f);
            }
        }
        #pragma unroll 4
        for (int k = 4; k < num_split; k++) {
            float w_k = smem_w[w_base + k];
            float _vec_load_0[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + (stat_base + k) * 512 + d0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_1[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + (stat_base + k) * 512 + d0 + 8);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                    }
                }
            }
            #pragma unroll
            for (int e_2 = 0; e_2 < 8; e_2++) {
                float c_e = w_k * _vec_load_0[e_2];
                float safe_e = ((w_k > 0.0f) ? c_e : 0.0f);
                acc[e_2] = acc[e_2] + safe_e;
            }
            #pragma unroll
            for (int e_3 = 0; e_3 < 8; e_3++) {
                float c_e_1 = w_k * _vec_load_1[e_3];
                float safe_e_1 = ((w_k > 0.0f) ? c_e_1 : 0.0f);
                acc[8 + e_3] = acc[8 + e_3] + safe_e_1;
            }
        }
        #pragma unroll
        for (int e_4 = 0; e_4 < 16; e_4++) {
            acc[e_4] = acc[e_4] * inv_sum;
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O))[row * 512 + d0 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(acc[8 + 0], acc[8 + 1]);
            _pk[1] = __floats2bfloat162_rn(acc[8 + 2], acc[8 + 3]);
            _pk[2] = __floats2bfloat162_rn(acc[8 + 4], acc[8 + 5]);
            _pk[3] = __floats2bfloat162_rn(acc[8 + 6], acc[8 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O))[row * 512 + d0 + 8 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        if (has_lse != 0) {
            if (part == 0) {
                if (lane == 0) {
                    float safe_w = ((sum_w > 0.0f) ? sum_w : 1.0f);
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(safe_w));
                    float lse_l2 = max_m + _log2_0 - lse_bias;
                    float lse_v = ((sum_w > 0.0f) ? lse_l2 * 0.6931471805599453f : -CAKE_INF);
                    *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v;
                }
            }
        }
    }
}

} // extern "C"
