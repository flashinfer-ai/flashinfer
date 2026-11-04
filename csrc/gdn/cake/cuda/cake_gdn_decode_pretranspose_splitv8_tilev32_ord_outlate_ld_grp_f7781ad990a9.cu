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
#define SMEM_SSTATE_STAGE_BYTES 16384
#define SMEM_SSTATE_STRIDE 16384
#define SMEM_SV_OFF 16384
#define SMEM_SV_STAGE_BYTES 512
#define SMEM_SV_STRIDE 512
#define SMEM_SOUTPUT_OFF 16896
#define SMEM_SOUTPUT_STAGE_BYTES 256
#define SMEM_SOUTPUT_STRIDE 256
#define SMEM_TOTAL 17280
#define THREADS 128
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#elif H != 16
#error "H takes 16 in this program; another value renders other code"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#elif HV != 32
#error "HV takes 32 in this program; another value renders other code"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif

__device__ __forceinline__ float2 mul_f32x2_local(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_splitv8_tilev32_ord_outlate_ld_grp(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_size_p0)
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
    float* sV = reinterpret_cast<float*>(smem_raw + 16384);
    const int sV_addr = smem + 16384;
    __nv_bfloat16* sOutput = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16896);
    const int sOutput_addr = smem + 16896;

    // === Task calls (declaration order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 4;
    int split = linear_block - state_head * 4;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    int qk_base = (n * H + qk_h) * 128;
    int vh_base = (n * HV + h) * 128;
    long long state_pool_size = state_size_p0;
    int read_state_slot = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot >= 0 && state_pool_size > (long long)read_state_slot) ? 1 : 0);
    int safe_read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot : 0);
    int write_state_slot = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot >= 0 && state_pool_size > (long long)write_state_slot) ? 1 : 0);
    int safe_write_state_slot = ((write_state_slot_valid != 0) ? write_state_slot : 0);
    long long read_state_head_base = ((long long)safe_read_state_slot * (long long)HV + (long long)h) * 16384;
    long long write_state_head_base = ((long long)safe_write_state_slot * (long long)HV + (long long)h) * 16384;
    int split_v_base = split * 32;
    int k_start = lane_local * 4;
    float r_q[4];
    float r_k[4];
    float r_v[4];
    float r_h[4];
    float2 r_q_pair0;
    float2 r_q_pair1;
    float2 r_k_pair0;
    float2 r_k_pair1;
    float r_hs[4];
    float r_o[32];
    float r_A_log = A_log[h];
    float r_a = (float)a[n * HV + h];
    float r_dt_bias = dt_bias[h];
    float r_b = (float)b[n * HV + h];
    #pragma unroll
    for (int copy_iter = 0; copy_iter < 8; copy_iter++) {
        int copy_seg = copy_iter * 128 + tid;
        int copy_row = copy_seg / 32;
        int copy_k_vec = copy_seg - copy_row * 32;
        int copy_v_row = split_v_base + copy_row;
        int copy_k_base = copy_k_vec * 4;
        int copy_dst = sState_addr + (unsigned int)((copy_row * 128 + copy_k_base) * 4);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(copy_dst), "l"(state + (read_state_head_base + (long long)(copy_v_row * 128) + (long long)copy_k_base)), "r"((read_state_slot_valid != 0) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
    }
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(q + qk_base + k_start);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(k + qk_base + k_start);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
            (&r_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
    }
    float2 q_raw_pair0 = make_float2(r_q[0], r_q[1]);
    float2 q_raw_pair1 = make_float2(r_q[2], r_q[3]);
    float2 k_raw_pair0 = make_float2(r_k[0], r_k[1]);
    float2 k_raw_pair1 = make_float2(r_k[2], r_k[3]);
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair0, k_raw_pair0, make_float2(0.0f, 0.0f));
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair1, k_raw_pair1, sum_k_pair);
    float sum_q = sum_q_pair.x + sum_q_pair.y;
    float sum_k = sum_k_pair.x + sum_k_pair.y;
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(v + vh_base + k_start);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_v[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
            (&r_v[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
    }
    if (warp_local == 0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            sV[k_start + i] = r_v[i];
        }
    }
    float _warp_reduce_3 = sum_q;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
    sum_q = _warp_reduce_3;
    float _warp_reduce_4 = sum_k;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
    sum_k = _warp_reduce_4;
    float _rsqrt_5 = rsqrtf(sum_q + 1e-06f);
    float q_norm = cake_f32_mul_f64const_rn(_rsqrt_5, SCALE);
    float k_norm = rsqrtf(sum_k + 1e-06f);
    float2 q_norm_pair = make_float2(q_norm, q_norm);
    float2 k_norm_pair = make_float2(k_norm, k_norm);
    r_q_pair0 = mul_f32x2_local(q_raw_pair0, q_norm_pair);
    r_q_pair1 = mul_f32x2_local(q_raw_pair1, q_norm_pair);
    r_k_pair0 = mul_f32x2_local(k_raw_pair0, k_norm_pair);
    r_k_pair1 = mul_f32x2_local(k_raw_pair1, k_norm_pair);
    float decay_lane = 0.0f;
    float beta_lane = 0.0f;
    if (lane_local == 0) {
        float x = r_a + r_dt_bias;
        float softplus_x = x;
        if (x <= 20.0f) {
            float _exp_7 = __expf(x);
            float _log2_6;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(1.0f + _exp_7));
            softplus_x = _log2_6 * 0.6931472f;
        }
        float _exp_9 = __expf(-r_b);
        float _rcp_8 = approx_rcp(1.0f + _exp_9);
        beta_lane = _rcp_8;
        float _exp_10 = __expf(r_A_log);
        float g_log = (-_exp_10) * softplus_x;
        float _exp_11 = __expf(g_log);
        decay_lane = _exp_11;
    }
    float decay_val = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
    float beta_val = __shfl_sync(0xFFFFFFFF, beta_lane, 0);
    __syncthreads();
    float2 decay_pair = make_float2(decay_val, decay_val);
    #pragma unroll
    for (int tile = 0; tile < 1; tile++) {
        int stage = 0;
        int tile_v_base = split_v_base + tile * 32;
        __syncthreads();
        if (tile + 1 < 1) {
            int next_tile_v_base = split_v_base + (tile + 1) * 32;
            int next_stage = 0;
            #pragma unroll
            for (int copy_iter_1 = 0; copy_iter_1 < 8; copy_iter_1++) {
                int copy_seg_1 = copy_iter_1 * 128 + tid;
                int copy_row_1 = copy_seg_1 / 32;
                int copy_k_vec_1 = copy_seg_1 - copy_row_1 * 32;
                int copy_v_row_1 = next_tile_v_base + copy_row_1;
                int copy_k_base_1 = copy_k_vec_1 * 4;
                int copy_dst_1 = sState_addr + (unsigned int)(next_stage * 16384) + (unsigned int)((copy_row_1 * 128 + copy_k_base_1) * 4);
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(copy_dst_1), "l"(state + (read_state_head_base + (long long)(copy_v_row_1 * 128) + (long long)copy_k_base_1)), "r"((read_state_slot_valid != 0) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int row_group = 0; row_group < 32; row_group += 4) {
            if (row_group == 0) {
                asm volatile("cp.async.wait_group 7;");
            }
            if (row_group == 4) {
                asm volatile("cp.async.wait_group 6;");
            }
            if (row_group == 8) {
                asm volatile("cp.async.wait_group 5;");
            }
            if (row_group == 12) {
                asm volatile("cp.async.wait_group 4;");
            }
            if (row_group == 16) {
                asm volatile("cp.async.wait_group 3;");
            }
            if (row_group == 20) {
                asm volatile("cp.async.wait_group 2;");
            }
            if (row_group == 24) {
                asm volatile("cp.async.wait_group 1;");
            }
            if (row_group == 28) {
                asm volatile("cp.async.wait_group 0;");
            }
            int v_row = tile_v_base + row_group + warp_local;
            int local_row = row_group + warp_local;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r_hs[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_hs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_hs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_hs[(0) + 3]))
                : "r"(sState_addr + (unsigned int)((stage * 4096 + local_row * 128 + k_start) * 4)) : "memory");
            float2 h_src_pair0 = make_float2(r_hs[0], r_hs[1]);
            float2 h_src_pair1 = make_float2(r_hs[2], r_hs[3]);
            float2 h_pair0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&h_pair0) : "l"(*(const unsigned long long*)&h_src_pair0), "l"(*(const unsigned long long*)&decay_pair));
            float2 h_pair1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&h_pair1) : "l"(*(const unsigned long long*)&h_src_pair1), "l"(*(const unsigned long long*)&decay_pair));
            float2 sum_hk_pair = fma_f32x2_rn_ftz(h_pair0, r_k_pair0, make_float2(0.0f, 0.0f));
            sum_hk_pair = fma_f32x2_rn_ftz(h_pair1, r_k_pair1, sum_hk_pair);
            float sum_hk = sum_hk_pair.x + sum_hk_pair.y;
            float _warp_reduce_12 = sum_hk;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
            sum_hk = _warp_reduce_12;
            float v_val = sV[v_row];
            float v_new = (v_val - sum_hk) * beta_val;
            float2 v_new_pair = make_float2(v_new, v_new);
            float2 h_new_pair0 = fma_f32x2_rn_ftz(r_k_pair0, v_new_pair, h_pair0);
            float2 h_new_pair1 = fma_f32x2_rn_ftz(r_k_pair1, v_new_pair, h_pair1);
            r_h[0] = h_new_pair0.x;
            r_h[1] = h_new_pair0.y;
            r_h[2] = h_new_pair1.x;
            r_h[3] = h_new_pair1.y;
            float2 sum_hq_pair = fma_f32x2_rn_ftz(h_new_pair0, r_q_pair0, make_float2(0.0f, 0.0f));
            sum_hq_pair = fma_f32x2_rn_ftz(h_new_pair1, r_q_pair1, sum_hq_pair);
            float sum_hq = sum_hq_pair.x + sum_hq_pair.y;
            if (read_state_slot_valid != 0) {
                if (write_state_slot_valid != 0) {
                    {
                        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                        *reinterpret_cast<float4*>((state + (write_state_head_base + (long long)(v_row * 128))) + k_start) = _v4;
                    }
                }
            }
            float _warp_reduce_13 = sum_hq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_13 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset);
            sum_hq = _warp_reduce_13;
            r_o[row_group] = ((read_state_slot_valid != 0) ? sum_hq : 0.0f);
        }
        if (lane_local == 0) {
            #pragma unroll
            for (int out_rg = 0; out_rg < 32; out_rg += 4) {
                out[vh_base + tile_v_base + out_rg + warp_local] = r_o[out_rg];
            }
        }
    }
}

} // extern "C"
// clang-format on
