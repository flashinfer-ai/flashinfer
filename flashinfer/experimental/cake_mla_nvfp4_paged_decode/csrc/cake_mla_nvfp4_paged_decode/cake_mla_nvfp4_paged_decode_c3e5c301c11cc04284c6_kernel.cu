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
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_RED_OFF 1024
#define SMEM_SMEM_RED_STAGE_BYTES 64
#define SMEM_SMEM_RED_STRIDE 64
#define SMEM_SMEM_ACC_OFF 1088
#define SMEM_SMEM_ACC_STAGE_BYTES 4096
#define SMEM_SMEM_ACC_STRIDE 4096
#define SMEM_TOTAL 5248
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_mla_nvfp4_paged_decode_c3e5c301c11cc04284c6(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, __nv_bfloat16* __restrict__ O, float* __restrict__ lse, int* __restrict__ cum_seq_lens_q, int batch, int num_heads, int num_split, float bmm2_scale, float lse_bias, int has_lse)
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
    float* smem_red = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_RED_OFF);
    const int smem_red_addr = smem + SMEM_SMEM_RED_OFF;
    float* smem_acc = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_ACC_OFF);
    const int smem_acc_addr = smem + SMEM_SMEM_ACC_OFF;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int chunk = blockIdx.y;
    int row = blockIdx.x;
    int rows_total = cum_seq_lens_q[batch] * num_heads;
    if (row < rows_total) {
        int stat_base = row * num_split;
        int last_split = num_split - 1;
        int half = lane >> 4;
        int d0 = chunk * 128 + (lane & 15) * 8;
        unsigned int pfw[20];
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            int s_pf = (j * 8 + warp) * 2 + half;
            if (s_pf <= last_split) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (stat_base + s_pf) * 512 + d0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&pfw[j * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
            }
        }
        int s_idx = tid;
        int s_ld = ((s_idx > last_split) ? last_split : s_idx);
        float m_raw = partial_max[stat_base + s_ld];
        float sum_raw = partial_sum[stat_base + s_ld];
        float m_s = ((sum_raw > 0.0f) ? m_raw : -CAKE_INF);
        m_s = ((s_idx > last_split) ? -CAKE_INF : m_s);
        float _warp_reduce_0 = m_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float m_warp = _warp_reduce_0;
        if (lane == 0) {
            smem_red[warp] = m_warp;
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        float max_m = smem_red[0];
        #pragma unroll
        for (int w = 1; w < 8; w++) {
            float m_w = smem_red[w];
            float _max_0 = max_noftz(max_m, m_w);
            max_m = _max_0;
        }
        float w_s = 0.0f;
        if (m_s > -CAKE_INF) {
            float _exp2_0 = approx_exp2(m_s - max_m);
            w_s = _exp2_0 * sum_raw;
        }
        smem_w[s_idx] = w_s;
        float _warp_reduce_1 = w_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float sum_warp = _warp_reduce_1;
        if (lane == 0) {
            smem_red[8 + warp] = sum_warp;
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        float sum_w = 0.0f;
        #pragma unroll
        for (int w_1 = 0; w_1 < 8; w_1++) {
            float s_w = smem_red[8 + w_1];
            sum_w = sum_w + s_w;
        }
        float inv_sum = 0.0f;
        if (sum_w > 0.0f) {
            float _rcp_0 = approx_rcp(sum_w);
            inv_sum = _rcp_0 * bmm2_scale;
        }
        float acc[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            acc[e] = 0.0f;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 5; j_1++) {
            int s_raw_p = (j_1 * 8 + warp) * 2 + half;
            int s_w_p = ((s_raw_p > last_split) ? last_split : s_raw_p);
            float w_raw_p = smem_w[s_w_p];
            float w_p = ((s_raw_p > last_split) ? 0.0f : w_raw_p);
            #pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                unsigned int wd = pfw[j_1 * 4 + e2];
                float lo_p = __uint_as_float(wd << 16);
                float hi_p = __uint_as_float(wd & 4294901760u);
                float c_lo = w_p * lo_p;
                float c_hi = w_p * hi_p;
                float s_lo = ((w_p > 0.0f) ? c_lo : 0.0f);
                float s_hi = ((w_p > 0.0f) ? c_hi : 0.0f);
                acc[2 * e2] = acc[2 * e2] + s_lo;
                acc[2 * e2 + 1] = acc[2 * e2 + 1] + s_hi;
            }
        }
        int n_iter_w = (num_split + 16 - 1) / 16;
        #pragma unroll 8
        for (int k = 5; k < n_iter_w; k++) {
            int s_raw = (k * 8 + warp) * 2 + half;
            int s_w_idx = ((s_raw > last_split) ? last_split : s_raw);
            float w_raw = smem_w[s_w_idx];
            float w_k = ((s_raw > last_split) ? 0.0f : w_raw);
            float _vec_load_0[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + (stat_base + s_w_idx) * 512 + d0);
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
            #pragma unroll
            for (int e_1 = 0; e_1 < 8; e_1++) {
                float contrib = w_k * _vec_load_0[e_1];
                float safe = ((w_k > 0.0f) ? contrib : 0.0f);
                acc[e_1] = acc[e_1] + safe;
            }
        }
        #pragma unroll
        for (int e_2 = 0; e_2 < 8; e_2++) {
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, acc[e_2], 16);
            float other = _shfl_xor_0;
            acc[e_2] = acc[e_2] + other;
        }
        if (lane < 16) {
            #pragma unroll
            for (int e_3 = 0; e_3 < 8; e_3++) {
                smem_acc[warp * 128 + lane * 8 + e_3] = acc[e_3];
            }
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        if (warp == 0) {
            int d_out = chunk * 128 + lane * 4;
            float out[4];
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                float tot = 0.0f;
                #pragma unroll
                for (int w_2 = 0; w_2 < 8; w_2++) {
                    float a_w = smem_acc[w_2 * 128 + lane * 4 + e_4];
                    tot = tot + a_w;
                }
                out[e_4] = tot * inv_sum;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O))[row * 512 + d_out]) = _pk2;
            }
            if (has_lse != 0) {
                if (chunk == 0) {
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
}

} // extern "C"
