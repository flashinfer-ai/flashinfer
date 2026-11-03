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
#define NUM_PANEL_PIPE_STAGES 3
#define SMEM_SSTATE_OFF 1024
#define SMEM_SSTATE_STAGE_BYTES 49152
#define SMEM_SSTATE_STRIDE 49152
#define SMEM_SQ_OFF 50176
#define SMEM_SQ_STAGE_BYTES 512
#define SMEM_SQ_STRIDE 512
#define SMEM_SK_OFF 50688
#define SMEM_SK_STAGE_BYTES 512
#define SMEM_SK_STRIDE 512
#define SMEM_TOTAL 51328
#define THREADS 256
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

__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_gdn_decode_nontranspose_fp32_t1_hqsplit_tma3(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, const __grid_constant__ CUtensorMap state_tma, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, long long q_stride_p0, long long k_stride_p0, long long v_stride_p0, long long a_stride_p0, long long b_stride_p0)
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

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* sState = reinterpret_cast<float*>(smem_raw + 1024);
    const int sState_addr = smem + 1024;
    float* sQ = reinterpret_cast<float*>(smem_raw + 50176);
    const int sQ_addr = smem + 50176;
    float* sK = reinterpret_cast<float*>(smem_raw + 50688);
    const int sK_addr = smem + 50688;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'panel_pipe' ---
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int state_head = blockIdx.x;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int qk_h = h / (HV / H);
    long long q_base = (long long)n * q_stride_p0 + (long long)qk_h * 128;
    long long k_base = (long long)n * k_stride_p0 + (long long)qk_h * 128;
    long long v_base = (long long)n * v_stride_p0 + (long long)h * 128;
    int out_base = (n * HV + h) * 128;
    int lane_local = lane;
    int warp_local = warp;
    int k_local = lane_local / 4;
    int v_local = lane_local - k_local * 4;
    int tile_v_local = warp_local * 4 + v_local;
    int swz_col = (warp_local ^ k_local) * 4 + v_local;
    float r_q[4];
    float r_k[4];
    float r_h[16];
    if (warp == 0) {
        if (elect_sync()) {
            #pragma unroll
            for (int panel = 0; panel < 2; panel++) {
                mbarrier_arrive_expect_tx(full_addr + (panel) * 8, 16384);
                tma_4d_gmem2smem(sState_addr + (unsigned int)(panel * 16384), (&state_tma), panel * 32, 0, h, n, full_addr + (panel) * 8);
            }
        }
    }
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
            sQ[qk_lane_base + i_1] = r_q[i_1] * q_norm;
            sK[qk_lane_base + i_1] = r_k[i_1] * k_norm;
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
    __syncthreads();
    int cs = 0;
    unsigned int _phase_full = 0;
    #pragma unroll
    for (int tile = 0; tile < 4; tile++) {
        unsigned int stage_base = sState_addr + (unsigned int)(cs * 16384);
        int stage_elems = cs * 4096;
        int tile_v_base = tile * 32;
        if (tile + 2 < 4) {
            int ahead_stage = (tile + 2) % 3;
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive_expect_tx(full_addr + (((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2)) * 8, 16384);
                    tma_4d_gmem2smem(sState_addr + (unsigned int)(ahead_stage * 16384), (&state_tma), (tile + 2) * 32, 0, h, n, full_addr + (((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2)) * 8);
                }
            }
        }
        mbarrier_wait(full_addr + (cs) * 8, _phase_full);
        float sum_hk = 0.0f;
        #pragma unroll
        for (int k_iter = 0; k_iter < 16; k_iter++) {
            int k_idx = k_iter * 8 + k_local;
            float h_val = sState[stage_elems + k_idx * 32 + swz_col] * decay_val;
            r_h[k_iter] = h_val;
            sum_hk = __fmaf_rn(h_val, sK[k_idx], sum_hk);
        }
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 16);
        sum_hk += _shfl_xor_sync_11;
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 8);
        sum_hk += _shfl_xor_sync_12;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 4);
        sum_hk += _shfl_xor_sync_13;
        float value_lane = 0.0f;
        if (k_local == 0) {
            value_lane = (float)v[v_base + (long long)tile_v_base + (long long)tile_v_local];
        }
        float value_val = __shfl_sync(0xFFFFFFFF, value_lane, v_local);
        float delta = (value_val - sum_hk) * beta_val;
        float sum_hq = 0.0f;
        #pragma unroll
        for (int k_iter_1 = 0; k_iter_1 < 16; k_iter_1++) {
            int k_idx_1 = k_iter_1 * 8 + k_local;
            float h_new = __fmaf_rn(sK[k_idx_1], delta, r_h[k_iter_1]);
            r_h[k_iter_1] = h_new;
            sum_hq = __fmaf_rn(h_new, sQ[k_idx_1], sum_hq);
        }
        #pragma unroll
        for (int k_iter_2 = 0; k_iter_2 < 16; k_iter_2++) {
            int k_idx_2 = k_iter_2 * 8 + k_local;
            sState[stage_elems + k_idx_2 * 32 + swz_col] = r_h[k_iter_2];
        }
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 16);
        sum_hq += _shfl_xor_sync_14;
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 8);
        sum_hq += _shfl_xor_sync_15;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 4);
        sum_hq += _shfl_xor_sync_16;
        if (k_local == 0) {
            out[out_base + tile_v_base + tile_v_local] = sum_hq;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            if (elect_sync()) {
                tma_store_4d((&state_tma), tile * 32, 0, h, n, stage_base);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
        cs += 1;
        if (cs == 3) { cs = 0; _phase_full ^= 1; }
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
// clang-format on
