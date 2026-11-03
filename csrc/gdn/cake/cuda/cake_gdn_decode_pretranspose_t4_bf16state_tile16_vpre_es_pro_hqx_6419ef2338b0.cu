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
#define SMEM_SQ_OFF 0
#define SMEM_SQ_STAGE_BYTES 4352
#define SMEM_SQ_STRIDE 4352
#define SMEM_SK_OFF 4352
#define SMEM_SK_STAGE_BYTES 4352
#define SMEM_SK_STRIDE 4352
#define SMEM_SSCALAR_OFF 8704
#define SMEM_SSCALAR_STAGE_BYTES 64
#define SMEM_SSCALAR_STRIDE 64
#define SMEM_SV_OFF 8800
#define SMEM_SV_STAGE_BYTES 512
#define SMEM_SV_STRIDE 512
#define SMEM_TOTAL 9344
#define THREADS 128
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#elif H != 4
#error "H takes 4 in this program; another value renders other code"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#elif HV != 8
#error "HV takes 8 in this program; another value renders other code"
#endif
#define T_STEPS 4
#ifndef INTERMEDIATE_BATCH_STRIDE
#error "INTERMEDIATE_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#elif INTERMEDIATE_BATCH_STRIDE != 524288
#error "INTERMEDIATE_BATCH_STRIDE takes 524288 in this program; another value renders other code"
#endif
#ifndef INTERMEDIATE_TOKEN_STRIDE
#error "INTERMEDIATE_TOKEN_STRIDE is a downstream specialization of this program; define it on the compile line"
#elif INTERMEDIATE_TOKEN_STRIDE != 131072
#error "INTERMEDIATE_TOKEN_STRIDE takes 131072 in this program; another value renders other code"
#endif
#ifndef STRIDED_INPUTS
#error "STRIDED_INPUTS is a downstream specialization of this program; define it on the compile line"
#elif STRIDED_INPUTS != 1
#error "STRIDED_INPUTS takes 1 in this program; another value renders other code"
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
kernel_gdn_decode_pretranspose_t4_bf16state_tile16_vpre_es_pro_hqx(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_size_p0, long long state_stride_p0, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2)
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
    float* sQ = reinterpret_cast<float*>(smem_raw + 0);
    const int sQ_addr = smem + 0;
    float* sK = reinterpret_cast<float*>(smem_raw + 4352);
    const int sK_addr = smem + 4352;
    float* sScalar = reinterpret_cast<float*>(smem_raw + 8704);
    const int sScalar_addr = smem + 8704;
    float* sV = reinterpret_cast<float*>(smem_raw + 8800);
    const int sV_addr = smem + 8800;

    // === Task calls (declaration order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 8;
    int i_v_tile = linear_block - state_head * 8;
    int v_tile_base = i_v_tile * 16;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot_raw : 0);
    long long state_slot_stride = state_stride_p0;
    long long read_state_head_base = (long long)read_state_slot * state_slot_stride + (long long)h * 16384;
    int k_start = lane_local * 4;
    int qk_smem_col = k_start;
    int v_row_a = v_tile_base + warp_local * 4;
    int v_row_b = v_row_a + 1;
    int v_row_c = v_row_a + 2;
    int v_row_d = v_row_a + 3;
    float r_q[4];
    float r_k[4];
    float r_h_a[4];
    float r_h_b[4];
    float r_h_c[4];
    float r_h_d[4];
    float r_h[4];
    float r_v[4];
    float r_o[4];
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_a * 128) + (long long)k_start);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_h_a[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&r_h_a[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_b * 128) + (long long)k_start);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_h_b[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
            (&r_h_b[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
    }
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_c * 128) + (long long)k_start);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_h_c[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
            (&r_h_c[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
    }
    {
        uint2 _vld_3;
        _vld_3 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_d * 128) + (long long)k_start);
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_h_d[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
            (&r_h_d[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
    }
    float2 h_a_pair0 = make_float2(r_h_a[0], r_h_a[1]);
    float2 h_a_pair1 = make_float2(r_h_a[2], r_h_a[3]);
    float2 h_b_pair0 = make_float2(r_h_b[0], r_h_b[1]);
    float2 h_b_pair1 = make_float2(r_h_b[2], r_h_b[3]);
    float2 h_c_pair0 = make_float2(r_h_c[0], r_h_c[1]);
    float2 h_c_pair1 = make_float2(r_h_c[2], r_h_c[3]);
    float2 h_d_pair0 = make_float2(r_h_d[0], r_h_d[1]);
    float2 h_d_pair1 = make_float2(r_h_d[2], r_h_d[3]);
    int v_tok = ((lane_local < T_STEPS) ? lane_local : 0);
    {
        long long v_pre_base = (long long)n * v_stride_p0 + (long long)v_tok * v_stride_p1 + (long long)h * v_stride_p2;
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(v + v_pre_base + (long long)v_row_a);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_v[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&r_v[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
    }
    if (lane_local < T_STEPS) {
        sV[(warp_local * 8 + v_tok) * 4] = r_v[0];
        sV[(warp_local * 8 + v_tok) * 4 + 1] = r_v[1];
        sV[(warp_local * 8 + v_tok) * 4 + 2] = r_v[2];
        sV[(warp_local * 8 + v_tok) * 4 + 3] = r_v[3];
    }
    int t_pre0_raw = warp_local;
    int t_pre0 = t_pre0_raw;
    int t_pre1_raw = warp_local + 4;
    int t_pre1 = t_pre1_raw;
    {
        t_pre1 = ((t_pre1_raw < T_STEPS) ? t_pre1_raw : 0);
    }
    float r_q0[4];
    float r_k0[4];
    float r_q1[4];
    float r_k1[4];
    float a_raw0 = 0.0f;
    float b_raw0 = 0.0f;
    float a_raw1 = 0.0f;
    float b_raw1 = 0.0f;
    {
        long long q_base0 = (long long)n * q_stride_p0 + (long long)t_pre0 * q_stride_p1 + (long long)qk_h * q_stride_p2;
        long long k_base0 = (long long)n * k_stride_p0 + (long long)t_pre0 * k_stride_p1 + (long long)qk_h * k_stride_p2;
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(q + q_base0 + (long long)k_start);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_q0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&r_q0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(k + k_base0 + (long long)k_start);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_k0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                (&r_k0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
            }
        }
        long long a_index0 = (long long)n * a_stride_p0 + (long long)t_pre0 * a_stride_p1 + (long long)h * a_stride_p2;
        long long b_index0 = (long long)n * b_stride_p0 + (long long)t_pre0 * b_stride_p1 + (long long)h * b_stride_p2;
        a_raw0 = (float)a[a_index0];
        b_raw0 = (float)b[b_index0];
    }
    float dt_h = dt_bias[h];
    float alog_h = A_log[h];
    float2 q_raw_pair0 = make_float2(r_q0[0], r_q0[1]);
    float2 q_raw_pair1 = make_float2(r_q0[2], r_q0[3]);
    float2 k_raw_pair0 = make_float2(r_k0[0], r_k0[1]);
    float2 k_raw_pair1 = make_float2(r_k0[2], r_k0[3]);
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair0, k_raw_pair0, make_float2(0.0f, 0.0f));
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair1, k_raw_pair1, sum_k_pair);
    float sum_q = sum_q_pair.x + sum_q_pair.y;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_q += __shfl_xor_sync(0xFFFFFFFF, sum_q, offset);
    float sum_k = sum_k_pair.x + sum_k_pair.y;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_k += __shfl_xor_sync(0xFFFFFFFF, sum_k, offset);
    float _rsqrt_7 = rsqrtf(sum_q + 1e-06f);
    float q_norm = cake_f32_mul_f64const_rn(_rsqrt_7, SCALE);
    float k_norm = rsqrtf(sum_k + 1e-06f);
    float2 _f32x2_mul_b_8 = make_float2(q_norm, q_norm);
    float2 q_pair0;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair0) : "l"(*(const unsigned long long*)&q_raw_pair0), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
    float2 _f32x2_mul_b_9 = make_float2(q_norm, q_norm);
    float2 q_pair1;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair1) : "l"(*(const unsigned long long*)&q_raw_pair1), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
    float2 _f32x2_mul_b_10 = make_float2(k_norm, k_norm);
    float2 k_pair0;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k_pair0) : "l"(*(const unsigned long long*)&k_raw_pair0), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
    float2 _f32x2_mul_b_11 = make_float2(k_norm, k_norm);
    float2 k_pair1;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k_pair1) : "l"(*(const unsigned long long*)&k_raw_pair1), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
    sQ[t_pre0_raw * 136 + qk_smem_col] = q_pair0.x;
    sQ[t_pre0_raw * 136 + qk_smem_col + 1] = q_pair0.y;
    sQ[t_pre0_raw * 136 + qk_smem_col + 2] = q_pair1.x;
    sQ[t_pre0_raw * 136 + qk_smem_col + 3] = q_pair1.y;
    sK[t_pre0_raw * 136 + qk_smem_col] = k_pair0.x;
    sK[t_pre0_raw * 136 + qk_smem_col + 1] = k_pair0.y;
    sK[t_pre0_raw * 136 + qk_smem_col + 2] = k_pair1.x;
    sK[t_pre0_raw * 136 + qk_smem_col + 3] = k_pair1.y;
    float x = a_raw0 + dt_h;
    float _exp_12 = __expf(-b_raw0);
    float beta_scalar = approx_rcp(1.0f + _exp_12);
    float _exp_14 = __expf(x);
    float _log2_13;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_13) : "f"(1.0f + _exp_14));
    float softplus_val = _log2_13 * 0.6931472f;
    float softplus_x = ((x <= 20.0f) ? softplus_val : x);
    float _exp_15 = __expf(alog_h);
    float decay_scalar = __expf((-_exp_15) * softplus_x);
    if (lane_local == 0) {
        sScalar[t_pre0_raw * 2] = decay_scalar;
        sScalar[t_pre0_raw * 2 + 1] = beta_scalar;
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < T_STEPS; t++) {
        int q_smem_addr = sQ_addr + (unsigned int)((t * 136 + qk_smem_col) * 4);
        int k_smem_addr = sK_addr + (unsigned int)((t * 136 + qk_smem_col) * 4);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_q[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 3]))
            : "r"(q_smem_addr));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 3]))
            : "r"(k_smem_addr));
        float2 _f2_0 = make_float2(r_q[0], r_q[1]);
        float2 _f2_1 = make_float2(r_q[2], r_q[3]);
        float2 _f2_2 = make_float2(r_k[0], r_k[1]);
        float2 _f2_3 = make_float2(r_k[2], r_k[3]);
        float decay_val = sScalar[t * 2];
        float beta_val = sScalar[t * 2 + 1];
        float2 decay_pair = make_float2(decay_val, decay_val);
        h_a_pair0 = mul_f32x2_local(h_a_pair0, decay_pair);
        h_a_pair1 = mul_f32x2_local(h_a_pair1, decay_pair);
        h_b_pair0 = mul_f32x2_local(h_b_pair0, decay_pair);
        h_b_pair1 = mul_f32x2_local(h_b_pair1, decay_pair);
        h_c_pair0 = mul_f32x2_local(h_c_pair0, decay_pair);
        h_c_pair1 = mul_f32x2_local(h_c_pair1, decay_pair);
        h_d_pair0 = mul_f32x2_local(h_d_pair0, decay_pair);
        h_d_pair1 = mul_f32x2_local(h_d_pair1, decay_pair);
        float2 sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair0, _f2_2, make_float2(0.0f, 0.0f));
        sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair1, _f2_3, sum_hk_a_pair);
        float2 sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair0, _f2_2, make_float2(0.0f, 0.0f));
        sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair1, _f2_3, sum_hk_b_pair);
        float2 sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair0, _f2_2, make_float2(0.0f, 0.0f));
        sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair1, _f2_3, sum_hk_c_pair);
        float2 sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair0, _f2_2, make_float2(0.0f, 0.0f));
        sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair1, _f2_3, sum_hk_d_pair);
        float sum_hk_a = sum_hk_a_pair.x + sum_hk_a_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_hk_a += __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, offset);
        float sum_hk_b = sum_hk_b_pair.x + sum_hk_b_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_hk_b += __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, offset);
        float sum_hk_c = sum_hk_c_pair.x + sum_hk_c_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_hk_c += __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, offset);
        float sum_hk_d = sum_hk_d_pair.x + sum_hk_d_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_hk_d += __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, offset);
        r_v[0] = sV[(warp_local * 8 + t) * 4];
        r_v[1] = sV[(warp_local * 8 + t) * 4 + 1];
        r_v[2] = sV[(warp_local * 8 + t) * 4 + 2];
        r_v[3] = sV[(warp_local * 8 + t) * 4 + 3];
        float v_new_a = (r_v[0] - sum_hk_a) * beta_val;
        float v_new_b = (r_v[1] - sum_hk_b) * beta_val;
        float v_new_c = (r_v[2] - sum_hk_c) * beta_val;
        float v_new_d = (r_v[3] - sum_hk_d) * beta_val;
        h_a_pair0 = fma_f32x2_rn_ftz(_f2_2, make_float2(v_new_a, v_new_a), h_a_pair0);
        h_a_pair1 = fma_f32x2_rn_ftz(_f2_3, make_float2(v_new_a, v_new_a), h_a_pair1);
        h_b_pair0 = fma_f32x2_rn_ftz(_f2_2, make_float2(v_new_b, v_new_b), h_b_pair0);
        h_b_pair1 = fma_f32x2_rn_ftz(_f2_3, make_float2(v_new_b, v_new_b), h_b_pair1);
        h_c_pair0 = fma_f32x2_rn_ftz(_f2_2, make_float2(v_new_c, v_new_c), h_c_pair0);
        h_c_pair1 = fma_f32x2_rn_ftz(_f2_3, make_float2(v_new_c, v_new_c), h_c_pair1);
        h_d_pair0 = fma_f32x2_rn_ftz(_f2_2, make_float2(v_new_d, v_new_d), h_d_pair0);
        h_d_pair1 = fma_f32x2_rn_ftz(_f2_3, make_float2(v_new_d, v_new_d), h_d_pair1);
        if (read_state_slot_valid != 0) {
            long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
            r_h[0] = h_a_pair0.x;
            r_h[1] = h_a_pair0.y;
            r_h[2] = h_a_pair1.x;
            r_h[3] = h_a_pair1.y;
            unsigned int _packed_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(intermediate_state + (cache_head_base + (long long)(v_row_a * 128) + (long long)k_start))[0] = reinterpret_cast<int2*>(_packed_0)[0];
            r_h[0] = h_b_pair0.x;
            r_h[1] = h_b_pair0.y;
            r_h[2] = h_b_pair1.x;
            r_h[3] = h_b_pair1.y;
            unsigned int _packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(intermediate_state + (cache_head_base + (long long)(v_row_b * 128) + (long long)k_start))[0] = reinterpret_cast<int2*>(_packed_1)[0];
            r_h[0] = h_c_pair0.x;
            r_h[1] = h_c_pair0.y;
            r_h[2] = h_c_pair1.x;
            r_h[3] = h_c_pair1.y;
            unsigned int _packed_2[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(intermediate_state + (cache_head_base + (long long)(v_row_c * 128) + (long long)k_start))[0] = reinterpret_cast<int2*>(_packed_2)[0];
            r_h[0] = h_d_pair0.x;
            r_h[1] = h_d_pair0.y;
            r_h[2] = h_d_pair1.x;
            r_h[3] = h_d_pair1.y;
            unsigned int _packed_3[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(intermediate_state + (cache_head_base + (long long)(v_row_d * 128) + (long long)k_start))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        }
        float2 sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair0, _f2_0, make_float2(0.0f, 0.0f));
        sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair1, _f2_1, sum_hq_a_pair);
        float2 sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair0, _f2_0, make_float2(0.0f, 0.0f));
        sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair1, _f2_1, sum_hq_b_pair);
        float2 sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair0, _f2_0, make_float2(0.0f, 0.0f));
        sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair1, _f2_1, sum_hq_c_pair);
        float2 sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair0, _f2_0, make_float2(0.0f, 0.0f));
        sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair1, _f2_1, sum_hq_d_pair);
        float hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
        float hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
        float hq_c = sum_hq_c_pair.x + sum_hq_c_pair.y;
        float hq_d = sum_hq_d_pair.x + sum_hq_d_pair.y;
        int hqx_lane_bit8 = lane_local & 8;
        float hqx_send0 = ((lane_local < 16) ? hq_c : hq_a);
        float hqx_send1 = ((lane_local < 16) ? hq_d : hq_b);
        float hqx_keep0 = ((lane_local < 16) ? hq_a : hq_c);
        float hqx_keep1 = ((lane_local < 16) ? hq_b : hq_d);
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, hqx_send0, 16);
        hqx_keep0 = hqx_keep0 + _shfl_xor_sync_16;
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, hqx_send1, 16);
        hqx_keep1 = hqx_keep1 + _shfl_xor_sync_17;
        float hqx_send = ((hqx_lane_bit8 == 0) ? hqx_keep1 : hqx_keep0);
        float hqx_keep = ((hqx_lane_bit8 == 0) ? hqx_keep0 : hqx_keep1);
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, hqx_send, 8);
        hqx_keep = hqx_keep + _shfl_xor_sync_18;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, hqx_keep, 4);
        hqx_keep = hqx_keep + _shfl_xor_sync_19;
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, hqx_keep, 2);
        hqx_keep = hqx_keep + _shfl_xor_sync_20;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, hqx_keep, 1);
        hqx_keep = hqx_keep + _shfl_xor_sync_21;
        float sum_hq_a = hqx_keep;
        float sum_hq_b = __shfl_xor_sync(0xFFFFFFFF, hqx_keep, 8);
        float sum_hq_c = __shfl_xor_sync(0xFFFFFFFF, hqx_keep, 16);
        float sum_hq_d = __shfl_xor_sync(0xFFFFFFFF, hqx_keep, 24);
        if (lane_local == 0) {
            if (read_state_slot_valid != 0) {
                r_o[0] = sum_hq_a;
                r_o[1] = sum_hq_b;
                r_o[2] = sum_hq_c;
                r_o[3] = sum_hq_d;
                unsigned int _packed_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_o[_lp*2 + 0], r_o[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_o[_lp*2 + 2], r_o[_lp*2+1 + 2]));
                    _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(out + (((n * T_STEPS + t) * HV + h) * 128 + v_row_a))[0] = reinterpret_cast<int2*>(_packed_4)[0];
            }
        }
    }
}

} // extern "C"
// clang-format on
