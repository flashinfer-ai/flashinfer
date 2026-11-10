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
#define SMEM_SQ_STAGE_BYTES 6144
#define SMEM_SQ_STRIDE 6144
#define SMEM_SK_OFF 6144
#define SMEM_SK_STAGE_BYTES 6144
#define SMEM_SK_STRIDE 6144
#define SMEM_SSCALAR_OFF 12288
#define SMEM_SSCALAR_STAGE_BYTES 64
#define SMEM_SSCALAR_STRIDE 64
#define SMEM_SV_OFF 12384
#define SMEM_SV_STAGE_BYTES 4096
#define SMEM_SV_STRIDE 4096
#define SMEM_TOTAL 16512
#define THREADS 128
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#elif H != 16
#error "H takes 16 in this program; another value renders other code"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#elif HV != 64
#error "HV takes 64 in this program; another value renders other code"
#endif
#ifndef T_STEPS
#error "T_STEPS is a downstream specialization of this program; define it on the compile line"
#elif T_STEPS != 2
#error "T_STEPS takes 2 in this program; another value renders other code"
#endif
#ifndef TILE_V_WIDE
#error "TILE_V_WIDE is a downstream specialization of this program; define it on the compile line"
#elif TILE_V_WIDE != 64
#error "TILE_V_WIDE takes 64 in this program; another value renders other code"
#endif
#ifndef UPDATE_STATE
#error "UPDATE_STATE is a downstream specialization of this program; define it on the compile line"
#elif UPDATE_STATE != 1
#error "UPDATE_STATE takes 1 in this program; another value renders other code"
#endif
#ifndef CACHE_INTERMEDIATE_STATES
#error "CACHE_INTERMEDIATE_STATES is a downstream specialization of this program; define it on the compile line"
#elif CACHE_INTERMEDIATE_STATES != 0
#error "CACHE_INTERMEDIATE_STATES takes 0 in this program; another value renders other code"
#endif
#ifndef INTERMEDIATE_BATCH_STRIDE
#error "INTERMEDIATE_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#elif INTERMEDIATE_BATCH_STRIDE != 16384
#error "INTERMEDIATE_BATCH_STRIDE takes 16384 in this program; another value renders other code"
#endif
#ifndef INTERMEDIATE_TOKEN_STRIDE
#error "INTERMEDIATE_TOKEN_STRIDE is a downstream specialization of this program; define it on the compile line"
#elif INTERMEDIATE_TOKEN_STRIDE != 16384
#error "INTERMEDIATE_TOKEN_STRIDE takes 16384 in this program; another value renders other code"
#endif
#ifndef STRIDED_INPUTS
#error "STRIDED_INPUTS is a downstream specialization of this program; define it on the compile line"
#elif STRIDED_INPUTS != 1
#error "STRIDED_INPUTS takes 1 in this program; another value renders other code"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif
#define num_v_tiles (128 / TILE_V_WIDE)
#define rows_per_group (TILE_V_WIDE / 8)
#define iters_per_group (rows_per_group / 4)

__device__ __forceinline__ float2 mul_f32x2_local(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_vpre_xchg_pro_w64(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_size_p0, long long state_stride_p0, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2)
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
    float* sK = reinterpret_cast<float*>(smem_raw + 6144);
    const int sK_addr = smem + 6144;
    float* sScalar = reinterpret_cast<float*>(smem_raw + 12288);
    const int sScalar_addr = smem + 12288;
    float* sV = reinterpret_cast<float*>(smem_raw + 12384);
    const int sV_addr = smem + 12384;

    // === Task calls (declaration order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / num_v_tiles;
    int i_v_tile = linear_block - state_head * num_v_tiles;
    int v_tile_base = i_v_tile * TILE_V_WIDE;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int group_idx = tid / 16;
    int lane_in_group = tid - group_idx * 16;
    int lane_hi8 = ((lane_in_group >= 8) ? 1 : 0);
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot_raw : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
    int write_state_slot = ((write_state_slot_valid != 0) ? write_state_slot_raw : 0);
    long long state_slot_stride = state_stride_p0;
    long long read_state_head_base = (long long)read_state_slot * state_slot_stride + (long long)h * 16384;
    long long write_state_head_base = (long long)write_state_slot * state_slot_stride + (long long)h * 16384;
    int k_start = lane_in_group * 8;
    int qk_smem_col = lane_in_group * 12;
    float r_q[8];
    float r_k[8];
    float r_h[8];
    float r_o[4];
    float r_v[4];
    float r_hp0[8];
    float r_hp1[8];
    float r_hp2[8];
    float r_hp3[8];
    float decay_scalar = 0.0f;
    float beta_scalar = 0.0f;
    int t_pre0_raw = warp_local;
    int t_pre0 = t_pre0_raw;
    {
        t_pre0 = ((t_pre0_raw < T_STEPS) ? t_pre0_raw : 0);
    }
    float a_raw0 = 0.0f;
    float b_raw0 = 0.0f;
    {
        long long q_base0 = (long long)n * q_stride_p0 + (long long)t_pre0 * q_stride_p1 + (long long)qk_h * q_stride_p2;
        long long k_base0 = (long long)n * k_stride_p0 + (long long)t_pre0 * k_stride_p1 + (long long)qk_h * k_stride_p2;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + q_base0 + (long long)k_start);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&r_q[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&r_q[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k + k_base0 + (long long)k_start);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&r_k[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&r_k[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        long long a_index0 = (long long)n * a_stride_p0 + (long long)t_pre0 * a_stride_p1 + (long long)h * a_stride_p2;
        long long b_index0 = (long long)n * b_stride_p0 + (long long)t_pre0 * b_stride_p1 + (long long)h * b_stride_p2;
        a_raw0 = (float)a[a_index0];
        b_raw0 = (float)b[b_index0];
    }
    float dt_h = dt_bias[h];
    float alog_h = A_log[h];
    int t_pre1_raw = warp_local + 4;
    int t_pre1 = t_pre1_raw;
    float r_q1[8];
    float r_k1[8];
    float a_raw1 = 0.0f;
    float b_raw1 = 0.0f;
    int v_tok = ((lane_in_group < T_STEPS) ? lane_in_group : 0);
    int v_row_pre = v_tile_base + group_idx * rows_per_group;
    float r_v_pre0[4];
    float r_v_pre1[4];
    {
        long long v_pre_base = (long long)n * v_stride_p0 + (long long)v_tok * v_stride_p1 + (long long)h * v_stride_p2;
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(v + v_pre_base + (long long)v_row_pre);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_v_pre0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&r_v_pre0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(v + v_pre_base + (long long)v_row_pre + 4);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_v_pre1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&r_v_pre1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
    }
    if (iters_per_group == 1) {
        int v_row_hp = v_tile_base + group_idx * rows_per_group;
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_hp * 128) + (long long)k_start);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&r_hp0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&r_hp0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)((v_row_hp + 1) * 128) + (long long)k_start);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&r_hp1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&r_hp1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)((v_row_hp + 2) * 128) + (long long)k_start);
            uint4 _vld_6[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_6[_blk] = _vptr_6[_blk];
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&r_hp2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&r_hp2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)((v_row_hp + 3) * 128) + (long long)k_start);
            uint4 _vld_7[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_7[_blk] = _vptr_7[_blk];
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&r_hp3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&r_hp3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
        }
    }
    float2 q_raw_pair0 = make_float2(r_q[0], r_q[1]);
    float2 q_raw_pair1 = make_float2(r_q[2], r_q[3]);
    float2 q_raw_pair2 = make_float2(r_q[4], r_q[5]);
    float2 q_raw_pair3 = make_float2(r_q[6], r_q[7]);
    float2 k_raw_pair0 = make_float2(r_k[0], r_k[1]);
    float2 k_raw_pair1 = make_float2(r_k[2], r_k[3]);
    float2 k_raw_pair2 = make_float2(r_k[4], r_k[5]);
    float2 k_raw_pair3 = make_float2(r_k[6], r_k[7]);
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair2, q_raw_pair2, sum_q_pair);
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair3, q_raw_pair3, sum_q_pair);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair0, k_raw_pair0, make_float2(0.0f, 0.0f));
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair1, k_raw_pair1, sum_k_pair);
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair2, k_raw_pair2, sum_k_pair);
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair3, k_raw_pair3, sum_k_pair);
    float sum_q = sum_q_pair.x + sum_q_pair.y;
    float sum_k = sum_k_pair.x + sum_k_pair.y;
    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 8);
    sum_q = sum_q + _shfl_xor_sync_8;
    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 8);
    sum_k = sum_k + _shfl_xor_sync_9;
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 4);
    sum_q = sum_q + _shfl_xor_sync_10;
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 4);
    sum_k = sum_k + _shfl_xor_sync_11;
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 2);
    sum_q = sum_q + _shfl_xor_sync_12;
    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 2);
    sum_k = sum_k + _shfl_xor_sync_13;
    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 1);
    sum_q = sum_q + _shfl_xor_sync_14;
    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 1);
    sum_k = sum_k + _shfl_xor_sync_15;
    float _rsqrt_16 = rsqrtf(sum_q + 1e-06f);
    float q_norm = cake_f32_mul_f64const_rn(_rsqrt_16, SCALE);
    float k_norm = rsqrtf(sum_k + 1e-06f);
    float2 q_norm_pair = make_float2(q_norm, q_norm);
    float2 k_norm_pair = make_float2(k_norm, k_norm);
    float2 q_pair0;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair0) : "l"(*(const unsigned long long*)&q_raw_pair0), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair1;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair1) : "l"(*(const unsigned long long*)&q_raw_pair1), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair2;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair2) : "l"(*(const unsigned long long*)&q_raw_pair2), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair3;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair3) : "l"(*(const unsigned long long*)&q_raw_pair3), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 k_pair0;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k_pair0) : "l"(*(const unsigned long long*)&k_raw_pair0), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair1;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k_pair1) : "l"(*(const unsigned long long*)&k_raw_pair1), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair2;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k_pair2) : "l"(*(const unsigned long long*)&k_raw_pair2), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair3;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k_pair3) : "l"(*(const unsigned long long*)&k_raw_pair3), "l"(*(const unsigned long long*)&k_norm_pair));
    if (lane_local < 16) {
        sQ[t_pre0_raw * 192 + qk_smem_col] = q_pair0.x;
        sQ[t_pre0_raw * 192 + qk_smem_col + 1] = q_pair0.y;
        sQ[t_pre0_raw * 192 + qk_smem_col + 2] = q_pair1.x;
        sQ[t_pre0_raw * 192 + qk_smem_col + 3] = q_pair1.y;
        sQ[t_pre0_raw * 192 + qk_smem_col + 4] = q_pair2.x;
        sQ[t_pre0_raw * 192 + qk_smem_col + 5] = q_pair2.y;
        sQ[t_pre0_raw * 192 + qk_smem_col + 6] = q_pair3.x;
        sQ[t_pre0_raw * 192 + qk_smem_col + 7] = q_pair3.y;
        sK[t_pre0_raw * 192 + qk_smem_col] = k_pair0.x;
        sK[t_pre0_raw * 192 + qk_smem_col + 1] = k_pair0.y;
        sK[t_pre0_raw * 192 + qk_smem_col + 2] = k_pair1.x;
        sK[t_pre0_raw * 192 + qk_smem_col + 3] = k_pair1.y;
        sK[t_pre0_raw * 192 + qk_smem_col + 4] = k_pair2.x;
        sK[t_pre0_raw * 192 + qk_smem_col + 5] = k_pair2.y;
        sK[t_pre0_raw * 192 + qk_smem_col + 6] = k_pair3.x;
        sK[t_pre0_raw * 192 + qk_smem_col + 7] = k_pair3.y;
    }
    float x = a_raw0 + dt_h;
    float _exp_18 = __expf(-b_raw0);
    float _rcp_17 = approx_rcp(1.0f + _exp_18);
    beta_scalar = _rcp_17;
    float _exp_20 = __expf(x);
    float _log2_19;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_19) : "f"(1.0f + _exp_20));
    float softplus_val = _log2_19 * 0.6931472f;
    float softplus_x = ((x <= 20.0f) ? softplus_val : x);
    float _exp_21 = __expf(alog_h);
    float g_log = (-_exp_21) * softplus_x;
    float _exp_22 = __expf(g_log);
    decay_scalar = _exp_22;
    if (lane_local == 0) {
        sScalar[t_pre0_raw * 2] = decay_scalar;
        sScalar[t_pre0_raw * 2 + 1] = beta_scalar;
    }
    if (lane_in_group < T_STEPS) {
        int sv_pre = (group_idx * 8 + v_tok) * 16;
        sV[sv_pre] = r_v_pre0[0];
        sV[sv_pre + 1] = r_v_pre0[1];
        sV[sv_pre + 2] = r_v_pre0[2];
        sV[sv_pre + 3] = r_v_pre0[3];
        sV[sv_pre + 4] = r_v_pre1[0];
        sV[sv_pre + 4 + 1] = r_v_pre1[1];
        sV[sv_pre + 4 + 2] = r_v_pre1[2];
        sV[sv_pre + 4 + 3] = r_v_pre1[3];
    }
    {
        __syncthreads();
    }
    #pragma unroll 1
    for (int iter_idx = 0; iter_idx < iters_per_group; iter_idx++) {
        int v_row_a = v_tile_base + group_idx * rows_per_group + iter_idx * 4;
        int v_row_b = v_row_a + 1;
        int v_row_c = v_row_a + 2;
        int v_row_d = v_row_a + 3;
        float2 h_a_pair0 = make_float2(0.0f, 0.0f);
        float2 h_a_pair1 = make_float2(0.0f, 0.0f);
        float2 h_a_pair2 = make_float2(0.0f, 0.0f);
        float2 h_a_pair3 = make_float2(0.0f, 0.0f);
        float2 h_b_pair0 = make_float2(0.0f, 0.0f);
        float2 h_b_pair1 = make_float2(0.0f, 0.0f);
        float2 h_b_pair2 = make_float2(0.0f, 0.0f);
        float2 h_b_pair3 = make_float2(0.0f, 0.0f);
        float2 h_c_pair0 = make_float2(0.0f, 0.0f);
        float2 h_c_pair1 = make_float2(0.0f, 0.0f);
        float2 h_c_pair2 = make_float2(0.0f, 0.0f);
        float2 h_c_pair3 = make_float2(0.0f, 0.0f);
        float2 h_d_pair0 = make_float2(0.0f, 0.0f);
        float2 h_d_pair1 = make_float2(0.0f, 0.0f);
        float2 h_d_pair2 = make_float2(0.0f, 0.0f);
        float2 h_d_pair3 = make_float2(0.0f, 0.0f);
        if (iters_per_group == 1) {
            h_a_pair0 = make_float2(r_hp0[0], r_hp0[1]);
            h_a_pair1 = make_float2(r_hp0[2], r_hp0[3]);
            h_a_pair2 = make_float2(r_hp0[4], r_hp0[5]);
            h_a_pair3 = make_float2(r_hp0[6], r_hp0[7]);
            h_b_pair0 = make_float2(r_hp1[0], r_hp1[1]);
            h_b_pair1 = make_float2(r_hp1[2], r_hp1[3]);
            h_b_pair2 = make_float2(r_hp1[4], r_hp1[5]);
            h_b_pair3 = make_float2(r_hp1[6], r_hp1[7]);
            h_c_pair0 = make_float2(r_hp2[0], r_hp2[1]);
            h_c_pair1 = make_float2(r_hp2[2], r_hp2[3]);
            h_c_pair2 = make_float2(r_hp2[4], r_hp2[5]);
            h_c_pair3 = make_float2(r_hp2[6], r_hp2[7]);
            h_d_pair0 = make_float2(r_hp3[0], r_hp3[1]);
            h_d_pair1 = make_float2(r_hp3[2], r_hp3[3]);
            h_d_pair2 = make_float2(r_hp3[4], r_hp3[5]);
            h_d_pair3 = make_float2(r_hp3[6], r_hp3[7]);
        } else {
            {
                const uint4* _vptr_23 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_a * 128) + (long long)k_start);
                uint4 _vld_23[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_23[_blk] = _vptr_23[_blk];
                    uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&r_h[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
                        (&r_h[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
                    }
                }
            }
            h_a_pair0 = make_float2(r_h[0], r_h[1]);
            h_a_pair1 = make_float2(r_h[2], r_h[3]);
            h_a_pair2 = make_float2(r_h[4], r_h[5]);
            h_a_pair3 = make_float2(r_h[6], r_h[7]);
            {
                const uint4* _vptr_24 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_b * 128) + (long long)k_start);
                uint4 _vld_24[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_24[_blk] = _vptr_24[_blk];
                    uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&r_h[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                        (&r_h[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                    }
                }
            }
            h_b_pair0 = make_float2(r_h[0], r_h[1]);
            h_b_pair1 = make_float2(r_h[2], r_h[3]);
            h_b_pair2 = make_float2(r_h[4], r_h[5]);
            h_b_pair3 = make_float2(r_h[6], r_h[7]);
            {
                const uint4* _vptr_25 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_c * 128) + (long long)k_start);
                uint4 _vld_25[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_25[_blk] = _vptr_25[_blk];
                    uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&r_h[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                        (&r_h[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                    }
                }
            }
            h_c_pair0 = make_float2(r_h[0], r_h[1]);
            h_c_pair1 = make_float2(r_h[2], r_h[3]);
            h_c_pair2 = make_float2(r_h[4], r_h[5]);
            h_c_pair3 = make_float2(r_h[6], r_h[7]);
            {
                const uint4* _vptr_26 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_d * 128) + (long long)k_start);
                uint4 _vld_26[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_26[_blk] = _vptr_26[_blk];
                    uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&r_h[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
                        (&r_h[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
                    }
                }
            }
            h_d_pair0 = make_float2(r_h[0], r_h[1]);
            h_d_pair1 = make_float2(r_h[2], r_h[3]);
            h_d_pair2 = make_float2(r_h[4], r_h[5]);
            h_d_pair3 = make_float2(r_h[6], r_h[7]);
        }
        #pragma unroll
        for (int t = 0; t < T_STEPS; t++) {
            int q_smem_addr = sQ_addr + (unsigned int)((t * 192 + qk_smem_col) * 4);
            int k_smem_addr = sK_addr + (unsigned int)((t * 192 + qk_smem_col) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r_q[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 3]))
                : "r"(q_smem_addr));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r_q[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(4) + 3]))
                : "r"(q_smem_addr + 16));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 3]))
                : "r"(k_smem_addr));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r_k[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(4) + 3]))
                : "r"(k_smem_addr + 16));
            float2 q_pair0_0 = make_float2(r_q[0], r_q[1]);
            float2 q_pair1_1 = make_float2(r_q[2], r_q[3]);
            float2 q_pair2_2 = make_float2(r_q[4], r_q[5]);
            float2 q_pair3_3 = make_float2(r_q[6], r_q[7]);
            float2 k_pair0_4 = make_float2(r_k[0], r_k[1]);
            float2 k_pair1_5 = make_float2(r_k[2], r_k[3]);
            float2 k_pair2_6 = make_float2(r_k[4], r_k[5]);
            float2 k_pair3_7 = make_float2(r_k[6], r_k[7]);
            float decay_val = decay_scalar;
            {
                decay_val = sScalar[t * 2];
            }
            float2 decay_pair = make_float2(decay_val, decay_val);
            h_a_pair0 = mul_f32x2_local(h_a_pair0, decay_pair);
            h_a_pair1 = mul_f32x2_local(h_a_pair1, decay_pair);
            h_a_pair2 = mul_f32x2_local(h_a_pair2, decay_pair);
            h_a_pair3 = mul_f32x2_local(h_a_pair3, decay_pair);
            h_b_pair0 = mul_f32x2_local(h_b_pair0, decay_pair);
            h_b_pair1 = mul_f32x2_local(h_b_pair1, decay_pair);
            h_b_pair2 = mul_f32x2_local(h_b_pair2, decay_pair);
            h_b_pair3 = mul_f32x2_local(h_b_pair3, decay_pair);
            h_c_pair0 = mul_f32x2_local(h_c_pair0, decay_pair);
            h_c_pair1 = mul_f32x2_local(h_c_pair1, decay_pair);
            h_c_pair2 = mul_f32x2_local(h_c_pair2, decay_pair);
            h_c_pair3 = mul_f32x2_local(h_c_pair3, decay_pair);
            h_d_pair0 = mul_f32x2_local(h_d_pair0, decay_pair);
            h_d_pair1 = mul_f32x2_local(h_d_pair1, decay_pair);
            h_d_pair2 = mul_f32x2_local(h_d_pair2, decay_pair);
            h_d_pair3 = mul_f32x2_local(h_d_pair3, decay_pair);
            float2 sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair0, k_pair0_4, make_float2(0.0f, 0.0f));
            sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair1, k_pair1_5, sum_hk_a_pair);
            sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair2, k_pair2_6, sum_hk_a_pair);
            sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair3, k_pair3_7, sum_hk_a_pair);
            float2 sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair0, k_pair0_4, make_float2(0.0f, 0.0f));
            sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair1, k_pair1_5, sum_hk_b_pair);
            sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair2, k_pair2_6, sum_hk_b_pair);
            sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair3, k_pair3_7, sum_hk_b_pair);
            float2 sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair0, k_pair0_4, make_float2(0.0f, 0.0f));
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair1, k_pair1_5, sum_hk_c_pair);
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair2, k_pair2_6, sum_hk_c_pair);
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair3, k_pair3_7, sum_hk_c_pair);
            float2 sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair0, k_pair0_4, make_float2(0.0f, 0.0f));
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair1, k_pair1_5, sum_hk_d_pair);
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair2, k_pair2_6, sum_hk_d_pair);
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair3, k_pair3_7, sum_hk_d_pair);
            float sum_hk_a = sum_hk_a_pair.x + sum_hk_a_pair.y;
            float sum_hk_b = sum_hk_b_pair.x + sum_hk_b_pair.y;
            float sum_hk_c = sum_hk_c_pair.x + sum_hk_c_pair.y;
            float sum_hk_d = sum_hk_d_pair.x + sum_hk_d_pair.y;
            float xk_send0 = ((lane_hi8 == 0) ? sum_hk_c : sum_hk_a);
            float xk_send1 = ((lane_hi8 == 0) ? sum_hk_d : sum_hk_b);
            float xk_keep0 = ((lane_hi8 == 0) ? sum_hk_a : sum_hk_c);
            float xk_keep1 = ((lane_hi8 == 0) ? sum_hk_b : sum_hk_d);
            float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, xk_send0, 8);
            xk_keep0 = xk_keep0 + _shfl_xor_sync_27;
            float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, xk_send1, 8);
            xk_keep1 = xk_keep1 + _shfl_xor_sync_28;
            float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, xk_keep0, 4);
            xk_keep0 = xk_keep0 + _shfl_xor_sync_29;
            float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, xk_keep1, 4);
            xk_keep1 = xk_keep1 + _shfl_xor_sync_30;
            float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, xk_keep0, 2);
            xk_keep0 = xk_keep0 + _shfl_xor_sync_31;
            float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, xk_keep1, 2);
            xk_keep1 = xk_keep1 + _shfl_xor_sync_32;
            float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, xk_keep0, 1);
            xk_keep0 = xk_keep0 + _shfl_xor_sync_33;
            float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, xk_keep1, 1);
            xk_keep1 = xk_keep1 + _shfl_xor_sync_34;
            float xk_other0 = __shfl_xor_sync(0xFFFFFFFF, xk_keep0, 8);
            float xk_other1 = __shfl_xor_sync(0xFFFFFFFF, xk_keep1, 8);
            sum_hk_a = ((lane_hi8 == 0) ? xk_keep0 : xk_other0);
            sum_hk_b = ((lane_hi8 == 0) ? xk_keep1 : xk_other1);
            sum_hk_c = ((lane_hi8 == 0) ? xk_other0 : xk_keep0);
            sum_hk_d = ((lane_hi8 == 0) ? xk_other1 : xk_keep1);
            float beta_val = beta_scalar;
            {
                beta_val = sScalar[t * 2 + 1];
            }
            float v_new_a = 0.0f;
            float v_new_b = 0.0f;
            float v_new_c = 0.0f;
            float v_new_d = 0.0f;
            {
                int sv_base = (group_idx * 8 + t) * 16 + iter_idx * 4;
                r_v[0] = sV[sv_base];
                r_v[1] = sV[sv_base + 1];
                r_v[2] = sV[sv_base + 2];
                r_v[3] = sV[sv_base + 3];
                v_new_a = (r_v[0] - sum_hk_a) * beta_val;
                v_new_b = (r_v[1] - sum_hk_b) * beta_val;
                v_new_c = (r_v[2] - sum_hk_c) * beta_val;
                v_new_d = (r_v[3] - sum_hk_d) * beta_val;
            }
            int out_base = ((n * T_STEPS + t) * HV + h) * 128;
            float2 v_new_a_pair = make_float2(v_new_a, v_new_a);
            float2 v_new_b_pair = make_float2(v_new_b, v_new_b);
            float2 v_new_c_pair = make_float2(v_new_c, v_new_c);
            float2 v_new_d_pair = make_float2(v_new_d, v_new_d);
            h_a_pair0 = fma_f32x2_rn_ftz(k_pair0_4, v_new_a_pair, h_a_pair0);
            h_a_pair1 = fma_f32x2_rn_ftz(k_pair1_5, v_new_a_pair, h_a_pair1);
            h_a_pair2 = fma_f32x2_rn_ftz(k_pair2_6, v_new_a_pair, h_a_pair2);
            h_a_pair3 = fma_f32x2_rn_ftz(k_pair3_7, v_new_a_pair, h_a_pair3);
            h_b_pair0 = fma_f32x2_rn_ftz(k_pair0_4, v_new_b_pair, h_b_pair0);
            h_b_pair1 = fma_f32x2_rn_ftz(k_pair1_5, v_new_b_pair, h_b_pair1);
            h_b_pair2 = fma_f32x2_rn_ftz(k_pair2_6, v_new_b_pair, h_b_pair2);
            h_b_pair3 = fma_f32x2_rn_ftz(k_pair3_7, v_new_b_pair, h_b_pair3);
            h_c_pair0 = fma_f32x2_rn_ftz(k_pair0_4, v_new_c_pair, h_c_pair0);
            h_c_pair1 = fma_f32x2_rn_ftz(k_pair1_5, v_new_c_pair, h_c_pair1);
            h_c_pair2 = fma_f32x2_rn_ftz(k_pair2_6, v_new_c_pair, h_c_pair2);
            h_c_pair3 = fma_f32x2_rn_ftz(k_pair3_7, v_new_c_pair, h_c_pair3);
            h_d_pair0 = fma_f32x2_rn_ftz(k_pair0_4, v_new_d_pair, h_d_pair0);
            h_d_pair1 = fma_f32x2_rn_ftz(k_pair1_5, v_new_d_pair, h_d_pair1);
            h_d_pair2 = fma_f32x2_rn_ftz(k_pair2_6, v_new_d_pair, h_d_pair2);
            h_d_pair3 = fma_f32x2_rn_ftz(k_pair3_7, v_new_d_pair, h_d_pair3);
            float2 sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair0, q_pair0_0, make_float2(0.0f, 0.0f));
            sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair1, q_pair1_1, sum_hq_a_pair);
            sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair2, q_pair2_2, sum_hq_a_pair);
            sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair3, q_pair3_3, sum_hq_a_pair);
            float2 sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair0, q_pair0_0, make_float2(0.0f, 0.0f));
            sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair1, q_pair1_1, sum_hq_b_pair);
            sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair2, q_pair2_2, sum_hq_b_pair);
            sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair3, q_pair3_3, sum_hq_b_pair);
            float2 sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair0, q_pair0_0, make_float2(0.0f, 0.0f));
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair1, q_pair1_1, sum_hq_c_pair);
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair2, q_pair2_2, sum_hq_c_pair);
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair3, q_pair3_3, sum_hq_c_pair);
            float2 sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair0, q_pair0_0, make_float2(0.0f, 0.0f));
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair1, q_pair1_1, sum_hq_d_pair);
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair2, q_pair2_2, sum_hq_d_pair);
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair3, q_pair3_3, sum_hq_d_pair);
            float sum_hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
            float sum_hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
            float sum_hq_c = sum_hq_c_pair.x + sum_hq_c_pair.y;
            float sum_hq_d = sum_hq_d_pair.x + sum_hq_d_pair.y;
            float xq_send0 = ((lane_hi8 == 0) ? sum_hq_c : sum_hq_a);
            float xq_send1 = ((lane_hi8 == 0) ? sum_hq_d : sum_hq_b);
            float xq_keep0 = ((lane_hi8 == 0) ? sum_hq_a : sum_hq_c);
            float xq_keep1 = ((lane_hi8 == 0) ? sum_hq_b : sum_hq_d);
            float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, xq_send0, 8);
            xq_keep0 = xq_keep0 + _shfl_xor_sync_35;
            float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, xq_send1, 8);
            xq_keep1 = xq_keep1 + _shfl_xor_sync_36;
            float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, xq_keep0, 4);
            xq_keep0 = xq_keep0 + _shfl_xor_sync_37;
            float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, xq_keep1, 4);
            xq_keep1 = xq_keep1 + _shfl_xor_sync_38;
            float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, xq_keep0, 2);
            xq_keep0 = xq_keep0 + _shfl_xor_sync_39;
            float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, xq_keep1, 2);
            xq_keep1 = xq_keep1 + _shfl_xor_sync_40;
            float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, xq_keep0, 1);
            xq_keep0 = xq_keep0 + _shfl_xor_sync_41;
            float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, xq_keep1, 1);
            xq_keep1 = xq_keep1 + _shfl_xor_sync_42;
            float xq_other0 = __shfl_xor_sync(0xFFFFFFFF, xq_keep0, 8);
            float xq_other1 = __shfl_xor_sync(0xFFFFFFFF, xq_keep1, 8);
            sum_hq_a = ((lane_hi8 == 0) ? xq_keep0 : xq_other0);
            sum_hq_b = ((lane_hi8 == 0) ? xq_keep1 : xq_other1);
            sum_hq_c = ((lane_hi8 == 0) ? xq_other0 : xq_keep0);
            sum_hq_d = ((lane_hi8 == 0) ? xq_other1 : xq_keep1);
            if (lane_in_group == 0) {
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
                    reinterpret_cast<int2*>(out + (out_base + v_row_a))[0] = reinterpret_cast<int2*>(_packed_4)[0];
                }
            }
        }
        {
            r_h[0] = h_a_pair0.x;
            r_h[1] = h_a_pair0.y;
            r_h[2] = h_a_pair1.x;
            r_h[3] = h_a_pair1.y;
            r_h[4] = h_a_pair2.x;
            r_h[5] = h_a_pair2.y;
            r_h[6] = h_a_pair3.x;
            r_h[7] = h_a_pair3.y;
            if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
                unsigned int _packed_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 4], r_h[_lp*2+1 + 4]));
                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 6], r_h[_lp*2+1 + 6]));
                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(state + (write_state_head_base + (long long)(v_row_a * 128) + (long long)k_start))[0] = reinterpret_cast<int4*>(_packed_5)[0];
            }
            r_h[0] = h_b_pair0.x;
            r_h[1] = h_b_pair0.y;
            r_h[2] = h_b_pair1.x;
            r_h[3] = h_b_pair1.y;
            r_h[4] = h_b_pair2.x;
            r_h[5] = h_b_pair2.y;
            r_h[6] = h_b_pair3.x;
            r_h[7] = h_b_pair3.y;
            if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
                unsigned int _packed_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                    _packed_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                    _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 4], r_h[_lp*2+1 + 4]));
                    _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 6], r_h[_lp*2+1 + 6]));
                    _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(state + (write_state_head_base + (long long)(v_row_b * 128) + (long long)k_start))[0] = reinterpret_cast<int4*>(_packed_6)[0];
            }
            r_h[0] = h_c_pair0.x;
            r_h[1] = h_c_pair0.y;
            r_h[2] = h_c_pair1.x;
            r_h[3] = h_c_pair1.y;
            r_h[4] = h_c_pair2.x;
            r_h[5] = h_c_pair2.y;
            r_h[6] = h_c_pair3.x;
            r_h[7] = h_c_pair3.y;
            if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
                unsigned int _packed_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                    _packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                    _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 4], r_h[_lp*2+1 + 4]));
                    _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 6], r_h[_lp*2+1 + 6]));
                    _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(state + (write_state_head_base + (long long)(v_row_c * 128) + (long long)k_start))[0] = reinterpret_cast<int4*>(_packed_7)[0];
            }
            r_h[0] = h_d_pair0.x;
            r_h[1] = h_d_pair0.y;
            r_h[2] = h_d_pair1.x;
            r_h[3] = h_d_pair1.y;
            r_h[4] = h_d_pair2.x;
            r_h[5] = h_d_pair2.y;
            r_h[6] = h_d_pair3.x;
            r_h[7] = h_d_pair3.y;
            if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
                unsigned int _packed_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 0], r_h[_lp*2+1 + 0]));
                    _packed_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 2], r_h[_lp*2+1 + 2]));
                    _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 4], r_h[_lp*2+1 + 4]));
                    _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_h[_lp*2 + 6], r_h[_lp*2+1 + 6]));
                    _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(state + (write_state_head_base + (long long)(v_row_d * 128) + (long long)k_start))[0] = reinterpret_cast<int4*>(_packed_8)[0];
            }
        }
    }
}

} // extern "C"
// clang-format on
