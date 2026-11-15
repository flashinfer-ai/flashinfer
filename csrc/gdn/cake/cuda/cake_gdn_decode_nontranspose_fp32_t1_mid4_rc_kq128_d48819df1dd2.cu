// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_gdn_device_common.cuh"
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
__device__ __forceinline__ int make_warp_uniform_local(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define SMEM_SSTATE_OFF 0
#define SMEM_SSTATE_STAGE_BYTES 18432
#define SMEM_SSTATE_STRIDE 18432
#define SMEM_SQP_OFF 18432
#define SMEM_SQP_STAGE_BYTES 640
#define SMEM_SQP_STRIDE 640
#define SMEM_SKP_OFF 19072
#define SMEM_SKP_STAGE_BYTES 640
#define SMEM_SKP_STRIDE 640
#define SMEM_TOTAL 19840
#define THREADS 256
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#elif H != 4 && H != 16
#error "H takes 4, 16 in this program; another value renders other code"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#elif HV != 8 && HV != 32
#error "HV takes 8, 32 in this program; another value renders other code"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif

extern "C" {

__global__ __launch_bounds__(256) void
kernel_gdn_decode_nontranspose_fp32_t1_mid4_rc_kq128(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, long long q_stride_p0, long long k_stride_p0, long long v_stride_p0, long long a_stride_p0, long long b_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform_local(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform_local(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* sState = reinterpret_cast<float*>(smem_raw + 0);
    const int sState_addr = smem + 0;
    float* sQp = reinterpret_cast<float*>(smem_raw + 18432);
    const int sQp_addr = smem + 18432;
    float* sKp = reinterpret_cast<float*>(smem_raw + 19072);
    const int sKp_addr = smem + 19072;

    // === Task calls (declaration order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 4;
    int split = linear_block - state_head * 4;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int qk_h = h / (HV / H);
    int tile_v_base = split * 32;
    long long q_base = (long long)n * q_stride_p0 + (long long)qk_h * 128;
    long long k_base = (long long)n * k_stride_p0 + (long long)qk_h * 128;
    long long v_base = (long long)n * v_stride_p0 + (long long)h * 128 + (long long)tile_v_base;
    int out_base = (n * HV + h) * 128 + tile_v_base;
    long long state_head_base = (long long)state_head * 128 * 128;
    int lane_local = lane;
    int warp_local = warp;
    int k_local = lane_local / 4;
    int v_local = lane_local - k_local * 4;
    int tile_v_local = warp_local * 4 + v_local;
    int qk_perm_base = (lane_local & 1) * 80 + lane_local / 2;
    float r_q[4];
    float r_k[4];
    float r_h[16];
    float r_store[4];
    float r_sk[16];
    float r_sq[16];
    #pragma unroll
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_seg = copy_iter * 256 + tid;
        int copy_k = copy_seg / 8;
        int copy_v_vec = copy_seg - copy_k * 8;
        int copy_v_base = copy_v_vec * 4;
        int copy_dst = sState_addr + (unsigned int)((copy_k * 36 + copy_v_base) * 4);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(copy_dst), "l"(state + (state_head_base + (long long)(copy_k * 128) + (long long)tile_v_base + (long long)copy_v_base)));
    }
    asm volatile("cp.async.commit_group;");
    if (warp_local == 0) {
        int qk_lane_base = lane_local * 4;
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(q + q_base + (long long)qk_lane_base);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k + k_base + (long long)qk_lane_base);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&r_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float sum_q = 0.0f;
        float sum_k = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            sum_q = __fmaf_rn(r_q[i], r_q[i], sum_q);
            sum_k = __fmaf_rn(r_k[i], r_k[i], sum_k);
        }
        float _warp_reduce_2 = sum_q;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        sum_q = _warp_reduce_2;
        float _warp_reduce_3 = sum_k;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        sum_k = _warp_reduce_3;
        float _rsqrt_4 = rsqrtf(sum_q + 1e-06f);
        float q_norm = cake_f32_mul_f64const_rn(_rsqrt_4, SCALE);
        float k_norm = rsqrtf(sum_k + 1e-06f);
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            sQp[qk_perm_base + i_1 * 20] = r_q[i_1] * q_norm;
            sKp[qk_perm_base + i_1 * 20] = r_k[i_1] * k_norm;
        }
    }
    float decay_lane = 0.0f;
    float beta_lane = 0.0f;
    if (lane_local == 0) {
        float gate_x = (float)a[(long long)n * a_stride_p0 + (long long)h] + dt_bias[h];
        float softplus_x = gate_x;
        if (gate_x <= 20.0f) {
            float _exp_6 = __expf(gate_x);
            float _log2_5;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(1.0f + _exp_6));
            softplus_x = _log2_5 * 0.6931472f;
        }
        float _exp_8 = __expf(A_log[h]);
        float _exp_7 = __expf((-_exp_8) * softplus_x);
        decay_lane = _exp_7;
        float _exp_10 = __expf(-(float)b[(long long)n * b_stride_p0 + (long long)h]);
        float _rcp_9 = approx_rcp(1.0f + _exp_10);
        beta_lane = _rcp_9;
    }
    float decay_val = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
    float beta_val = __shfl_sync(0xFFFFFFFF, beta_lane, 0);
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    #pragma unroll
    for (int kq_iter = 0; kq_iter < 4; kq_iter++) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_sk[kq_iter * 4])), "=r"(*reinterpret_cast<uint32_t*>(&r_sk[(kq_iter * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_sk[(kq_iter * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_sk[(kq_iter * 4) + 3]))
            : "r"(sKp_addr + (unsigned int)((k_local * 20 + kq_iter * 4) * 4)));
    }
    float sum_hk = 0.0f;
    #pragma unroll
    for (int k_iter = 0; k_iter < 16; k_iter++) {
        int k_idx = k_iter * 8 + k_local;
        float h_val = sState[k_idx * 36 + tile_v_local] * decay_val;
        r_h[k_iter] = h_val;
        sum_hk = __fmaf_rn(h_val, r_sk[k_iter], sum_hk);
    }
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 16);
    sum_hk += _shfl_xor_sync_11;
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 8);
    sum_hk += _shfl_xor_sync_12;
    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 4);
    sum_hk += _shfl_xor_sync_13;
    float value_lane = 0.0f;
    if (k_local == 0) {
        value_lane = (float)v[v_base + (long long)tile_v_local];
    }
    float value_val = __shfl_sync(0xFFFFFFFF, value_lane, v_local);
    float delta = (value_val - sum_hk) * beta_val;
    #pragma unroll
    for (int kq_iter_1 = 0; kq_iter_1 < 4; kq_iter_1++) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_sk[kq_iter_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&r_sk[(kq_iter_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_sk[(kq_iter_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_sk[(kq_iter_1 * 4) + 3]))
            : "r"(sKp_addr + (unsigned int)((k_local * 20 + kq_iter_1 * 4) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_sq[kq_iter_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&r_sq[(kq_iter_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_sq[(kq_iter_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_sq[(kq_iter_1 * 4) + 3]))
            : "r"(sQp_addr + (unsigned int)((k_local * 20 + kq_iter_1 * 4) * 4)));
    }
    float sum_hq = 0.0f;
    #pragma unroll
    for (int k_iter_1 = 0; k_iter_1 < 16; k_iter_1++) {
        int k_idx_1 = k_iter_1 * 8 + k_local;
        float h_new = __fmaf_rn(r_sk[k_iter_1], delta, r_h[k_iter_1]);
        r_h[k_iter_1] = h_new;
        sState[k_idx_1 * 36 + tile_v_local] = h_new;
        sum_hq = __fmaf_rn(h_new, r_sq[k_iter_1], sum_hq);
    }
    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 16);
    sum_hq += _shfl_xor_sync_14;
    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 8);
    sum_hq += _shfl_xor_sync_15;
    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 4);
    sum_hq += _shfl_xor_sync_16;
    if (k_local == 0) {
        out[out_base + tile_v_local] = sum_hq;
    }
    __syncthreads();
    #pragma unroll
    for (int copy_iter_1 = 0; copy_iter_1 < 4; copy_iter_1++) {
        int copy_seg_1 = copy_iter_1 * 256 + tid;
        int copy_k_1 = copy_seg_1 / 8;
        int copy_v_vec_1 = copy_seg_1 - copy_k_1 * 8;
        int copy_v_base_1 = copy_v_vec_1 * 4;
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; i_2++) {
            r_store[i_2] = sState[copy_k_1 * 36 + copy_v_base_1 + i_2];
        }
        {
            float4 _v4 = make_float4(r_store[0 + 0], r_store[0 + 1], r_store[0 + 2], r_store[0 + 3]);
            *reinterpret_cast<float4*>(state + state_head_base + (long long)(copy_k_1 * 128) + (long long)tile_v_base + (long long)copy_v_base_1) = _v4;
        }
    }
}

} // extern "C"
// clang-format on
