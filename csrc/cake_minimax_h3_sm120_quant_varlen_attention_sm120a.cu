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
// clang-format off
// MiniMax-H3 FP8 (E4M3) non-causal packed-varlen self-attention for SM120 (GB202: RTX 5090 /
// RTX PRO 6000 Blackwell), generated from the Cake kernel schedules.  One operator call = four
// launches over packed THD tensors q, k, v [T, H, 128] BF16 with int32 cu_seqlens:
//   1. kv_stats:           per (segment Q tile, head) K channel sums and V channel maxima,
//   2. kv_stats_finalize:  per (segment, head) K mean and V channel scale (amax / 448),
//   3. quantize_fp8:       Q rows (per-token scale), segment-centred K blocks (per-128-key-block
//                          scale), transposed / in-chunk-permuted V^T tile (per-channel scale),
//   4. attention_fp8:      persistent 256-thread CTA per SM, 8 warps x 16 query rows, 128-key K/V
//                          tiles through a two-stage TMA ring, FA3-style ping-pong between the two
//                          warp groups, mma.sync m16n8k32 kind::f8f6f4 for QK^T and PV, FP32 online
//                          softmax with E4M3 probabilities (2^8 exponent bias), BF16 output rows.
// Device code: TMA, ldmatrix, mma.sync kind::f8f6f4, mbarrier pipelines, named barriers.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

namespace h3_varlen_kv_stats_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_RED_SUM_OFF 0
#define SMEM_RED_SUM_STAGE_BYTES 8192
#define SMEM_RED_SUM_STRIDE 8192
#define SMEM_RED_MAX_OFF 8192
#define SMEM_RED_MAX_STAGE_BYTES 8192
#define SMEM_RED_MAX_STRIDE 8192
#define SMEM_TOTAL 16384
#define THREADS 256

#include <math_constants.h>


__global__ __launch_bounds__(256, 2) void
kernel_minimax_h3_sm120_varlen_kv_stats(__nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, float* __restrict__ partials, int num_tiles, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* red_sum = reinterpret_cast<float*>(smem_raw + 0);
    const int red_sum_addr = smem + 0;
    float* red_max = reinterpret_cast<float*>(smem_raw + 8192);
    const int red_max_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    #pragma unroll 1
    for (int work = bid; work < num_tiles * num_heads; work += num_bids) {
        int tile = work / num_heads;
        int head = work % num_heads;
        int seg = tile_table[2 * tile];
        int tile_in_seg = tile_table[2 * tile + 1];
        int sb = seg_begin[seg];
        int row_begin = sb + tile_in_seg * 128;
        int seg_end = sb + seg_len[seg];
        int row_end = ((seg_end > row_begin + 128) ? row_begin + 128 : seg_end);
        float ksum[8];
        float vmax[8];
        for (int e = 0; e < 8; e++) {
            ksum[e] = 0.0f;
            vmax[e] = 0.0f;
        }
        #pragma unroll 1
        for (int row = row_begin + rgrp; row < row_end; row += 16) {
            int base = (row * num_heads + head) * 128 + col;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(K + base + 0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
            float _vec_load_1[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(V + base + 0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            for (int e_1 = 0; e_1 < 8; e_1++) {
                float kf = _vec_load_0[e_1];
                float vf = _vec_load_1[e_1];
                ksum[e_1] = ksum[e_1] + kf;
                float _fabs_0 = fabsf(vf);
                float _fmax_0 = fmaxf(vmax[e_1], _fabs_0);
                vmax[e_1] = _fmax_0;
            }
        }
        for (int e_2 = 0; e_2 < 8; e_2++) {
            red_sum[rgrp * 128 + col + e_2] = ksum[e_2];
            red_max[rgrp * 128 + col + e_2] = vmax[e_2];
        }
        __syncthreads();
        if (tid < 128) {
            float total = 0.0f;
            float vmx = 0.0f;
            for (int g = 0; g < 16; g++) {
                total = total + red_sum[g * 128 + tid];
                float _fmax_1 = fmaxf(vmx, red_max[g * 128 + tid]);
                vmx = _fmax_1;
            }
            *(reinterpret_cast<float*>(partials + (work * 256 + tid)) + (0)) = total;
            *(reinterpret_cast<float*>(partials + (work * 256 + 128 + tid)) + (0)) = vmx;
        }
        __syncthreads();
    }
}

}  // namespace h3_varlen_kv_stats_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef SMEM_RED_MAX_OFF
#undef SMEM_RED_MAX_STAGE_BYTES
#undef SMEM_RED_MAX_STRIDE
#undef SMEM_RED_SUM_OFF
#undef SMEM_RED_SUM_STAGE_BYTES
#undef SMEM_RED_SUM_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_varlen_kv_stats_finalize_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>


__global__ __launch_bounds__(256, 4) void
kernel_minimax_h3_sm120_varlen_kv_stats_finalize(float* __restrict__ partials, int* __restrict__ seg_tile_begin, int* __restrict__ seg_len, float* __restrict__ mean_k, float* __restrict__ v_scale, int num_segments, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int chan = tid & 127;
    int is_max = tid >> 7;
    #pragma unroll 1
    for (int work = bid; work < num_segments * num_heads; work += num_bids) {
        int seg = work / num_heads;
        int head = work % num_heads;
        int t0 = seg_tile_begin[seg];
        int t1 = seg_tile_begin[seg + 1];
        int length = seg_len[seg];
        float acc = 0.0f;
        #pragma unroll 1
        for (int tile = t0; tile < t1; tile++) {
            float value = partials[(tile * num_heads + head) * 256 + is_max * 128 + chan];
            float _fmax_0 = fmaxf(acc, value);
            acc = ((is_max == 0) ? acc + value : _fmax_0);
        }
        if (length > 0) {
            if (is_max == 0) {
                float _fdiv_rn_0 = __fdiv_rn(acc, (float)length);
                *(reinterpret_cast<float*>(mean_k + (work * 128 + chan)) + (0)) = _fdiv_rn_0;
            } else {
                float _fmax_1 = fmaxf(acc, 1e-12f);
                float _fdiv_rn_1 = __fdiv_rn(_fmax_1, 448.0f);
                *(reinterpret_cast<float*>(v_scale + (work * 128 + chan)) + (0)) = _fdiv_rn_1;
            }
        }
    }
}

}  // namespace h3_varlen_kv_stats_finalize_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef THREADS

namespace h3_varlen_quantize_fp8_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_AMAX_PART_OFF 0
#define SMEM_AMAX_PART_STAGE_BYTES 32
#define SMEM_AMAX_PART_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 256

#include <math_constants.h>


__global__ __launch_bounds__(256, 2) void
kernel_minimax_h3_sm120_varlen_quantize_fp8(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ cu_seqlens, float* __restrict__ mean_k, float* __restrict__ v_scale, unsigned int* __restrict__ q8, unsigned int* __restrict__ k8, unsigned int* __restrict__ vt8, float* __restrict__ q_scale, float* __restrict__ k_scale, int total_tokens, int total_padded, int num_segments, int num_heads, int num_kblocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* amax_part = reinterpret_cast<float*>(smem_raw + 0);
    const int amax_part_addr = smem + 0;

    // === Task calls (dependency order) ===
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    #pragma unroll 1
    for (int work = bid; work < num_kblocks * num_heads; work += num_bids) {
        int blk = work / num_heads;
        int head = work % num_heads;
        int tok_base = blk * 128;
        float kvals[64];
        float kmax = 0.0f;
        for (int ps = 0; ps < 4; ps++) {
            int tok = tok_base + ps * 32 + trow;
            int valid = ((tok < total_tokens) ? 1 : 0);
            int tok_c = ((tok < total_tokens) ? tok : total_tokens - 1);
            int lo = 0;
            int hi = num_segments - 1;
            #pragma unroll 1
            for (int step = 0; step < 17; step++) {
                int mid = lo + hi + 1 >> 1;
                if (tok_c >= cu_seqlens[mid]) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            int seg = lo;
            int base = (tok_c * num_heads + head) * 128 + col16;
            float _vec_load_0[16];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(Q + base + 0);
                uint4 _vld_0[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
            float qamax = 0.0f;
            for (int e = 0; e < 16; e++) {
                float qf = _vec_load_0[e];
                float _fabs_0 = fabsf(qf);
                float _fmax_0 = fmaxf(qamax, _fabs_0);
                qamax = _fmax_0;
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, qamax, 4);
            float _fmax_1 = fmaxf(qamax, _shfl_xor_0);
            qamax = _fmax_1;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, qamax, 2);
            float _fmax_2 = fmaxf(qamax, _shfl_xor_1);
            qamax = _fmax_2;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, qamax, 1);
            float _fmax_3 = fmaxf(qamax, _shfl_xor_2);
            qamax = _fmax_3;
            float _fmax_4 = fmaxf(qamax, 1e-12f);
            qamax = _fmax_4;
            float _fdiv_rn_0 = __fdiv_rn(qamax, 448.0f);
            float qs = _fdiv_rn_0;
            float _fdiv_rn_1 = __fdiv_rn(448.0f, qamax);
            float qinv = _fdiv_rn_1;
            float qn[16];
            for (int e_1 = 0; e_1 < 16; e_1++) {
                float qf2 = _vec_load_0[e_1];
                qn[e_1] = qf2 * qinv;
            }
            unsigned int qw[4];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(qn[0]), "f"(qn[1]),
                                       "f"(qn[2]), "f"(qn[3]));
                qw[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(qn[4]), "f"(qn[5]),
                                       "f"(qn[6]), "f"(qn[7]));
                qw[1] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(qn[8]), "f"(qn[9]),
                                       "f"(qn[10]), "f"(qn[11]));
                qw[2] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(qn[12]), "f"(qn[13]),
                                       "f"(qn[14]), "f"(qn[15]));
                qw[3] = _packed;
            }
            if (valid == 1) {
                reinterpret_cast<int4*>(q8 + (base >> 2))[0] = reinterpret_cast<int4*>(qw)[0];
                if (sub == 0) {
                    *(reinterpret_cast<float*>(q_scale + (tok * num_heads + head)) + (0)) = qs;
                }
            }
            float _vec_load_1[16];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(K + base + 0);
                uint4 _vld_1[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            int mbase = (seg * num_heads + head) * 128 + col16;
            for (int quad = 0; quad < 4; quad++) {
                float _vec_load_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad * 4) + 0);
                    _vec_load_2[0 + 0] = _v4.x;
                    _vec_load_2[0 + 1] = _v4.y;
                    _vec_load_2[0 + 2] = _v4.z;
                    _vec_load_2[0 + 3] = _v4.w;
                }
                for (int e_2 = 0; e_2 < 4; e_2++) {
                    float kf = _vec_load_1[quad * 4 + e_2];
                    float kc = kf - _vec_load_2[e_2];
                    kvals[ps * 16 + quad * 4 + e_2] = ((valid == 1) ? kc : 0.0f);
                    float _fabs_1 = fabsf(kvals[ps * 16 + quad * 4 + e_2]);
                    float _fmax_5 = fmaxf(kmax, _fabs_1);
                    kmax = _fmax_5;
                }
            }
        }
        for (int stage = 0; stage < 5; stage++) {
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, kmax, 16 >> stage);
            float _fmax_6 = fmaxf(kmax, _shfl_xor_3);
            kmax = _fmax_6;
        }
        if (lane == 0) {
            amax_part[warp] = kmax;
        }
        __syncthreads();
        kmax = amax_part[lane & 7];
        for (int stage_1 = 0; stage_1 < 3; stage_1++) {
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, kmax, 4 >> stage_1);
            float _fmax_7 = fmaxf(kmax, _shfl_xor_4);
            kmax = _fmax_7;
        }
        float _fmax_8 = fmaxf(kmax, 1e-12f);
        kmax = _fmax_8;
        float _fdiv_rn_2 = __fdiv_rn(448.0f, kmax);
        float kinv = _fdiv_rn_2;
        if (tid == 0) {
            float _fdiv_rn_3 = __fdiv_rn(kmax, 448.0f);
            *(reinterpret_cast<float*>(k_scale + (head * num_kblocks + blk)) + (0)) = _fdiv_rn_3;
        }
        for (int ps_1 = 0; ps_1 < 4; ps_1++) {
            int tok2 = tok_base + ps_1 * 32 + trow;
            float kn[16];
            for (int e_3 = 0; e_3 < 16; e_3++) {
                kn[e_3] = kvals[ps_1 * 16 + e_3] * kinv;
            }
            unsigned int kw[4];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(kn[0]), "f"(kn[1]),
                                       "f"(kn[2]), "f"(kn[3]));
                kw[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(kn[4]), "f"(kn[5]),
                                       "f"(kn[6]), "f"(kn[7]));
                kw[1] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(kn[8]), "f"(kn[9]),
                                       "f"(kn[10]), "f"(kn[11]));
                kw[2] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(kn[12]), "f"(kn[13]),
                                       "f"(kn[14]), "f"(kn[15]));
                kw[3] = _packed;
            }
            if (tok2 < total_tokens) {
                reinterpret_cast<int4*>(k8 + ((tok2 * num_heads + head) * 128 + col16 >> 2))[0] = reinterpret_cast<int4*>(kw)[0];
            }
        }
        for (int pass2 = 0; pass2 < 1; pass2++) {
            int task = tid;
            int cg = task >> 5;
            int chunk = (task & 31) >> 2;
            int t8 = task & 3;
            float vq[64];
            for (int i = 0; i < 4; i++) {
                int key_local = chunk * 16 + (i >> 1) * 8 + 2 * t8 + (i & 1);
                int vtok = tok_base + key_local;
                int vvalid = ((vtok < total_tokens) ? 1 : 0);
                int vtok_c = ((vtok < total_tokens) ? vtok : total_tokens - 1);
                int vlo = 0;
                int vhi = num_segments - 1;
                #pragma unroll 1
                for (int step_1 = 0; step_1 < 17; step_1++) {
                    int vmid = vlo + vhi + 1 >> 1;
                    if (vtok_c >= cu_seqlens[vmid]) {
                        vlo = vmid;
                    } else {
                        vhi = vmid - 1;
                    }
                }
                int vseg = vlo;
                float _vec_load_3[16];
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(V + ((vtok_c * num_heads + head) * 128 + cg * 16) + 0);
                    uint4 _vld_3[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                int sbase = (vseg * num_heads + head) * 128 + cg * 16;
                for (int quad_1 = 0; quad_1 < 4; quad_1++) {
                    float _vec_load_4[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (sbase + quad_1 * 4) + 0);
                        _vec_load_4[0 + 0] = _v4.x;
                        _vec_load_4[0 + 1] = _v4.y;
                        _vec_load_4[0 + 2] = _v4.z;
                        _vec_load_4[0 + 3] = _v4.w;
                    }
                    for (int e_4 = 0; e_4 < 4; e_4++) {
                        float vf = _vec_load_3[quad_1 * 4 + e_4];
                        float _fdiv_rn_4 = __fdiv_rn(vf, _vec_load_4[e_4]);
                        vq[i * 16 + quad_1 * 4 + e_4] = ((vvalid == 1) ? _fdiv_rn_4 : 0.0f);
                    }
                }
            }
            int vt_word = (head * 128 + cg * 16) * total_padded + tok_base + chunk * 16 + 4 * t8 >> 2;
            for (int e_5 = 0; e_5 < 16; e_5++) {
                float four[4];
                for (int i_1 = 0; i_1 < 4; i_1++) {
                    four[i_1] = vq[i_1 * 16 + e_5];
                }
                unsigned int word[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(four[0]), "f"(four[1]),
                                           "f"(four[2]), "f"(four[3]));
                    word[0] = _packed;
                }
                *(reinterpret_cast<unsigned int*>(vt8 + (vt_word + e_5 * (total_padded >> 2))) + (0)) = word[0];
            }
        }
        __syncthreads();
    }
}

}  // namespace h3_varlen_quantize_fp8_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef SMEM_AMAX_PART_OFF
#undef SMEM_AMAX_PART_STAGE_BYTES
#undef SMEM_AMAX_PART_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_varlen_attention_fp8_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_KV_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 16384
#define SMEM_Q_STRIDE 16384
#define SMEM_K_STAGE_OFF 17408
#define SMEM_K_STAGE_STAGE_BYTES 16384
#define SMEM_K_STAGE_STRIDE 16384
#define SMEM_V_STAGE_OFF 50176
#define SMEM_V_STAGE_STAGE_BYTES 16384
#define SMEM_V_STAGE_STRIDE 16384
#define SMEM_TOTAL 82944
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(256, 1) void
kernel_minimax_h3_sm120_varlen_attention_fp8(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap V_map, __nv_bfloat16* __restrict__ O, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 40)
    #define v_empty_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int K_stage_addr = smem + 17408;
    uint8_t* V_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int V_stage_addr = smem + 50176;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv' ---
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            // v_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    if (warp >= 4) {
        asm volatile("barrier.arrive 1, 256;" ::: "memory");
    }
    unsigned int _phase_q_full_0 = 0;
    #pragma unroll 1
    for (int slot = bid; slot < total_units; slot += num_bids) {
        int seg = unit_table[2 * slot];
        int packed = unit_table[2 * slot + 1];
        int head = packed >> 16;
        int q_tile = packed & 65535;
        int seg_begin_v = seg_begin[seg];
        int seg_end_v = seg_begin_v + seg_len[seg];
        int q_row0 = seg_begin_v + q_tile * 128;
        int kb_first = seg_begin_v / 128;
        int kb_last = (seg_end_v - 1) / 128;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(q_row0), "r"(head),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(K_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"(kb_first * 128), "r"(head),
                       "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"(kb_first * 128), "r"(0), "r"(head),
                       "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
        if (kb_last >= kb_first + 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(V_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&V_map)), "r"((kb_first + 1) * 128), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int row0_c = ((row0 < seg_end_v) ? row0 : seg_end_v - 1);
        int row1_c = ((row1 < seg_end_v) ? row1 : seg_end_v - 1);
        float qs0 = q_scale[row0_c * num_heads + head];
        float qs1 = q_scale[row1_c * num_heads + head];
        float c_row0 = qs0 * softmax_scale_log2;
        float c_row1 = qs1 * softmax_scale_log2;
        unsigned int q_frags[16];
        unsigned int k_frag[4];
        unsigned int v_frag[4];
        float s_acc[64];
        unsigned int p_frag[16];
        float p_tmp[8];
        float o_acc[64];
        float o_tmp[4];
        float m_state[2];
        float l_state[2];
        o_acc[0] = 0.0f;
        o_acc[1] = 0.0f;
        o_acc[2] = 0.0f;
        o_acc[3] = 0.0f;
        o_acc[4] = 0.0f;
        o_acc[5] = 0.0f;
        o_acc[6] = 0.0f;
        o_acc[7] = 0.0f;
        o_acc[8] = 0.0f;
        o_acc[9] = 0.0f;
        o_acc[10] = 0.0f;
        o_acc[11] = 0.0f;
        o_acc[12] = 0.0f;
        o_acc[13] = 0.0f;
        o_acc[14] = 0.0f;
        o_acc[15] = 0.0f;
        o_acc[16] = 0.0f;
        o_acc[17] = 0.0f;
        o_acc[18] = 0.0f;
        o_acc[19] = 0.0f;
        o_acc[20] = 0.0f;
        o_acc[21] = 0.0f;
        o_acc[22] = 0.0f;
        o_acc[23] = 0.0f;
        o_acc[24] = 0.0f;
        o_acc[25] = 0.0f;
        o_acc[26] = 0.0f;
        o_acc[27] = 0.0f;
        o_acc[28] = 0.0f;
        o_acc[29] = 0.0f;
        o_acc[30] = 0.0f;
        o_acc[31] = 0.0f;
        o_acc[32] = 0.0f;
        o_acc[33] = 0.0f;
        o_acc[34] = 0.0f;
        o_acc[35] = 0.0f;
        o_acc[36] = 0.0f;
        o_acc[37] = 0.0f;
        o_acc[38] = 0.0f;
        o_acc[39] = 0.0f;
        o_acc[40] = 0.0f;
        o_acc[41] = 0.0f;
        o_acc[42] = 0.0f;
        o_acc[43] = 0.0f;
        o_acc[44] = 0.0f;
        o_acc[45] = 0.0f;
        o_acc[46] = 0.0f;
        o_acc[47] = 0.0f;
        o_acc[48] = 0.0f;
        o_acc[49] = 0.0f;
        o_acc[50] = 0.0f;
        o_acc[51] = 0.0f;
        o_acc[52] = 0.0f;
        o_acc[53] = 0.0f;
        o_acc[54] = 0.0f;
        o_acc[55] = 0.0f;
        o_acc[56] = 0.0f;
        o_acc[57] = 0.0f;
        o_acc[58] = 0.0f;
        o_acc[59] = 0.0f;
        o_acc[60] = 0.0f;
        o_acc[61] = 0.0f;
        o_acc[62] = 0.0f;
        o_acc[63] = 0.0f;
        m_state[0] = -H3_VARLEN_INF;
        m_state[1] = -H3_VARLEN_INF;
        l_state[0] = 0.0f;
        l_state[1] = 0.0f;
        mbarrier_wait(q_full_addr, _phase_q_full_0);
        _phase_q_full_0 ^= 1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[8]), "=r"(q_frags[9]), "=r"(q_frags[10]), "=r"(q_frags[11])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[12]), "=r"(q_frags[13]), "=r"(q_frags[14]), "=r"(q_frags[15])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[0]), "=f"(s_acc[1]), "=f"(s_acc[2]), "=f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[4]), "=f"(s_acc[(4) + 1]), "=f"(s_acc[(4) + 2]), "=f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[8]), "=f"(s_acc[(8) + 1]), "=f"(s_acc[(8) + 2]), "=f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[12]), "=f"(s_acc[(12) + 1]), "=f"(s_acc[(12) + 2]), "=f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[16]), "=f"(s_acc[(16) + 1]), "=f"(s_acc[(16) + 2]), "=f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[20]), "=f"(s_acc[(20) + 1]), "=f"(s_acc[(20) + 2]), "=f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[24]), "=f"(s_acc[(24) + 1]), "=f"(s_acc[(24) + 2]), "=f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[28]), "=f"(s_acc[(28) + 1]), "=f"(s_acc[(28) + 2]), "=f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[32]), "=f"(s_acc[(32) + 1]), "=f"(s_acc[(32) + 2]), "=f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[36]), "=f"(s_acc[(36) + 1]), "=f"(s_acc[(36) + 2]), "=f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[40]), "=f"(s_acc[(40) + 1]), "=f"(s_acc[(40) + 2]), "=f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[44]), "=f"(s_acc[(44) + 1]), "=f"(s_acc[(44) + 2]), "=f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[48]), "=f"(s_acc[(48) + 1]), "=f"(s_acc[(48) + 2]), "=f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[52]), "=f"(s_acc[(52) + 1]), "=f"(s_acc[(52) + 2]), "=f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[56]), "=f"(s_acc[(56) + 1]), "=f"(s_acc[(56) + 2]), "=f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(s_acc[60]), "=f"(s_acc[(60) + 1]), "=f"(s_acc[(60) + 2]), "=f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_first == kb || kb_last == kb) {
                s_acc[0] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[0]);
                s_acc[1] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[1]);
                s_acc[2] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[2]);
                s_acc[3] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[3]);
                s_acc[4] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[4]);
                s_acc[5] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[5]);
                s_acc[6] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[6]);
                s_acc[7] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[7]);
                s_acc[8] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[8]);
                s_acc[9] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[9]);
                s_acc[10] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[10]);
                s_acc[11] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[11]);
                s_acc[12] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[12]);
                s_acc[13] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[13]);
                s_acc[14] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[14]);
                s_acc[15] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[15]);
                s_acc[16] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[16]);
                s_acc[17] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[17]);
                s_acc[18] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[18]);
                s_acc[19] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[19]);
                s_acc[20] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[20]);
                s_acc[21] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[21]);
                s_acc[22] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[22]);
                s_acc[23] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[23]);
                s_acc[24] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[24]);
                s_acc[25] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[25]);
                s_acc[26] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[26]);
                s_acc[27] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[27]);
                s_acc[28] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[28]);
                s_acc[29] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[29]);
                s_acc[30] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[30]);
                s_acc[31] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[31]);
                s_acc[32] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[32]);
                s_acc[33] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[33]);
                s_acc[34] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[34]);
                s_acc[35] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[35]);
                s_acc[36] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[36]);
                s_acc[37] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[37]);
                s_acc[38] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[38]);
                s_acc[39] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[39]);
                s_acc[40] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[40]);
                s_acc[41] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[41]);
                s_acc[42] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[42]);
                s_acc[43] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[43]);
                s_acc[44] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[44]);
                s_acc[45] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[45]);
                s_acc[46] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[46]);
                s_acc[47] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[47]);
                s_acc[48] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[48]);
                s_acc[49] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[49]);
                s_acc[50] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[50]);
                s_acc[51] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[51]);
                s_acc[52] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[52]);
                s_acc[53] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[53]);
                s_acc[54] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[54]);
                s_acc[55] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[55]);
                s_acc[56] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[56]);
                s_acc[57] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[57]);
                s_acc[58] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[58]);
                s_acc[59] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[59]);
                s_acc[60] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[60]);
                s_acc[61] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[61]);
                s_acc[62] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[62]);
                s_acc[63] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[63]);
            }
            float _fmax_0 = fmaxf(s_acc[0], s_acc[1]);
            float _fmax_1 = fmaxf(s_acc[2], s_acc[3]);
            float _fmax_2 = fmaxf(s_acc[4], s_acc[5]);
            float _fmax_3 = fmaxf(_fmax_0, _fmax_2);
            float _fmax_4 = fmaxf(s_acc[6], s_acc[7]);
            float _fmax_5 = fmaxf(_fmax_1, _fmax_4);
            float _fmax_6 = fmaxf(s_acc[8], s_acc[9]);
            float _fmax_7 = fmaxf(_fmax_3, _fmax_6);
            float _fmax_8 = fmaxf(s_acc[10], s_acc[11]);
            float _fmax_9 = fmaxf(_fmax_5, _fmax_8);
            float _fmax_10 = fmaxf(s_acc[12], s_acc[13]);
            float _fmax_11 = fmaxf(_fmax_7, _fmax_10);
            float _fmax_12 = fmaxf(s_acc[14], s_acc[15]);
            float _fmax_13 = fmaxf(_fmax_9, _fmax_12);
            float _fmax_14 = fmaxf(s_acc[16], s_acc[17]);
            float _fmax_15 = fmaxf(_fmax_11, _fmax_14);
            float _fmax_16 = fmaxf(s_acc[18], s_acc[19]);
            float _fmax_17 = fmaxf(_fmax_13, _fmax_16);
            float _fmax_18 = fmaxf(s_acc[20], s_acc[21]);
            float _fmax_19 = fmaxf(_fmax_15, _fmax_18);
            float _fmax_20 = fmaxf(s_acc[22], s_acc[23]);
            float _fmax_21 = fmaxf(_fmax_17, _fmax_20);
            float _fmax_22 = fmaxf(s_acc[24], s_acc[25]);
            float _fmax_23 = fmaxf(_fmax_19, _fmax_22);
            float _fmax_24 = fmaxf(s_acc[26], s_acc[27]);
            float _fmax_25 = fmaxf(_fmax_21, _fmax_24);
            float _fmax_26 = fmaxf(s_acc[28], s_acc[29]);
            float _fmax_27 = fmaxf(_fmax_23, _fmax_26);
            float _fmax_28 = fmaxf(s_acc[30], s_acc[31]);
            float _fmax_29 = fmaxf(_fmax_25, _fmax_28);
            float _fmax_30 = fmaxf(s_acc[32], s_acc[33]);
            float _fmax_31 = fmaxf(_fmax_27, _fmax_30);
            float _fmax_32 = fmaxf(s_acc[34], s_acc[35]);
            float _fmax_33 = fmaxf(_fmax_29, _fmax_32);
            float _fmax_34 = fmaxf(s_acc[36], s_acc[37]);
            float _fmax_35 = fmaxf(_fmax_31, _fmax_34);
            float _fmax_36 = fmaxf(s_acc[38], s_acc[39]);
            float _fmax_37 = fmaxf(_fmax_33, _fmax_36);
            float _fmax_38 = fmaxf(s_acc[40], s_acc[41]);
            float _fmax_39 = fmaxf(_fmax_35, _fmax_38);
            float _fmax_40 = fmaxf(s_acc[42], s_acc[43]);
            float _fmax_41 = fmaxf(_fmax_37, _fmax_40);
            float _fmax_42 = fmaxf(s_acc[44], s_acc[45]);
            float _fmax_43 = fmaxf(_fmax_39, _fmax_42);
            float _fmax_44 = fmaxf(s_acc[46], s_acc[47]);
            float _fmax_45 = fmaxf(_fmax_41, _fmax_44);
            float _fmax_46 = fmaxf(s_acc[48], s_acc[49]);
            float _fmax_47 = fmaxf(_fmax_43, _fmax_46);
            float _fmax_48 = fmaxf(s_acc[50], s_acc[51]);
            float _fmax_49 = fmaxf(_fmax_45, _fmax_48);
            float _fmax_50 = fmaxf(s_acc[52], s_acc[53]);
            float _fmax_51 = fmaxf(_fmax_47, _fmax_50);
            float _fmax_52 = fmaxf(s_acc[54], s_acc[55]);
            float _fmax_53 = fmaxf(_fmax_49, _fmax_52);
            float _fmax_54 = fmaxf(s_acc[56], s_acc[57]);
            float _fmax_55 = fmaxf(_fmax_51, _fmax_54);
            float _fmax_56 = fmaxf(s_acc[58], s_acc[59]);
            float _fmax_57 = fmaxf(_fmax_53, _fmax_56);
            float _fmax_58 = fmaxf(s_acc[60], s_acc[61]);
            float _fmax_59 = fmaxf(_fmax_55, _fmax_58);
            float _fmax_60 = fmaxf(s_acc[62], s_acc[63]);
            float _fmax_61 = fmaxf(_fmax_57, _fmax_60);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _fmax_59, 1);
            float _fmax_62 = fmaxf(_fmax_59, _shfl_xor_0);
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fmax_62, 2);
            float _fmax_63 = fmaxf(_fmax_62, _shfl_xor_1);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _fmax_61, 1);
            float _fmax_64 = fmaxf(_fmax_61, _shfl_xor_2);
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, _fmax_64, 2);
            float _fmax_65 = fmaxf(_fmax_64, _shfl_xor_3);
            float _fmax_66 = fmaxf(m_state[0], _fmax_63 * (c_row0 * k_scale[head * num_kblocks + kb]));
            float _fmax_67 = fmaxf(m_state[1], _fmax_65 * (c_row1 * k_scale[head * num_kblocks + kb]));
            float _exp2_0 = approx_exp2(m_state[0] - _fmax_66);
            float _exp2_1 = approx_exp2(m_state[1] - _fmax_67);
            m_state[0] = _fmax_66;
            m_state[1] = _fmax_67;
            {
                float2 _pair_scale_even2_0 = make_float2(_exp2_0, _exp2_0);
                float2 _pair_scale_odd2_0 = make_float2(_exp2_1, _exp2_1);
                float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&o_acc[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                #else
                o_acc[0] *= _exp2_0;
                o_acc[1] *= _exp2_0;
                o_acc[2] *= _exp2_1;
                o_acc[3] *= _exp2_1;
                o_acc[4] *= _exp2_0;
                o_acc[5] *= _exp2_0;
                o_acc[6] *= _exp2_1;
                o_acc[7] *= _exp2_1;
                o_acc[8] *= _exp2_0;
                o_acc[9] *= _exp2_0;
                o_acc[10] *= _exp2_1;
                o_acc[11] *= _exp2_1;
                o_acc[12] *= _exp2_0;
                o_acc[13] *= _exp2_0;
                o_acc[14] *= _exp2_1;
                o_acc[15] *= _exp2_1;
                o_acc[16] *= _exp2_0;
                o_acc[17] *= _exp2_0;
                o_acc[18] *= _exp2_1;
                o_acc[19] *= _exp2_1;
                o_acc[20] *= _exp2_0;
                o_acc[21] *= _exp2_0;
                o_acc[22] *= _exp2_1;
                o_acc[23] *= _exp2_1;
                o_acc[24] *= _exp2_0;
                o_acc[25] *= _exp2_0;
                o_acc[26] *= _exp2_1;
                o_acc[27] *= _exp2_1;
                o_acc[28] *= _exp2_0;
                o_acc[29] *= _exp2_0;
                o_acc[30] *= _exp2_1;
                o_acc[31] *= _exp2_1;
                o_acc[32] *= _exp2_0;
                o_acc[33] *= _exp2_0;
                o_acc[34] *= _exp2_1;
                o_acc[35] *= _exp2_1;
                o_acc[36] *= _exp2_0;
                o_acc[37] *= _exp2_0;
                o_acc[38] *= _exp2_1;
                o_acc[39] *= _exp2_1;
                o_acc[40] *= _exp2_0;
                o_acc[41] *= _exp2_0;
                o_acc[42] *= _exp2_1;
                o_acc[43] *= _exp2_1;
                o_acc[44] *= _exp2_0;
                o_acc[45] *= _exp2_0;
                o_acc[46] *= _exp2_1;
                o_acc[47] *= _exp2_1;
                o_acc[48] *= _exp2_0;
                o_acc[49] *= _exp2_0;
                o_acc[50] *= _exp2_1;
                o_acc[51] *= _exp2_1;
                o_acc[52] *= _exp2_0;
                o_acc[53] *= _exp2_0;
                o_acc[54] *= _exp2_1;
                o_acc[55] *= _exp2_1;
                o_acc[56] *= _exp2_0;
                o_acc[57] *= _exp2_0;
                o_acc[58] *= _exp2_1;
                o_acc[59] *= _exp2_1;
                o_acc[60] *= _exp2_0;
                o_acc[61] *= _exp2_0;
                o_acc[62] *= _exp2_1;
                o_acc[63] *= _exp2_1;
                #endif
            }
            float _fma_0 = __fmaf_rn(s_acc[0], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_2 = approx_exp2(_fma_0);
            s_acc[0] = _exp2_2;
            float _fma_1 = __fmaf_rn(s_acc[1], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_3 = approx_exp2(_fma_1);
            s_acc[1] = _exp2_3;
            float _fma_2 = __fmaf_rn(s_acc[2], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_4 = approx_exp2(_fma_2);
            s_acc[2] = _exp2_4;
            float _fma_3 = __fmaf_rn(s_acc[3], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_5 = approx_exp2(_fma_3);
            s_acc[3] = _exp2_5;
            float _fma_4 = __fmaf_rn(s_acc[4], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_6 = approx_exp2(_fma_4);
            s_acc[4] = _exp2_6;
            float _fma_5 = __fmaf_rn(s_acc[5], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_7 = approx_exp2(_fma_5);
            s_acc[5] = _exp2_7;
            float _fma_6 = __fmaf_rn(s_acc[6], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_8 = approx_exp2(_fma_6);
            s_acc[6] = _exp2_8;
            float _fma_7 = __fmaf_rn(s_acc[7], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_9 = approx_exp2(_fma_7);
            s_acc[7] = _exp2_9;
            float _fma_8 = __fmaf_rn(s_acc[8], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_10 = approx_exp2(_fma_8);
            s_acc[8] = _exp2_10;
            float _fma_9 = __fmaf_rn(s_acc[9], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_11 = approx_exp2(_fma_9);
            s_acc[9] = _exp2_11;
            float _fma_10 = __fmaf_rn(s_acc[10], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_12 = approx_exp2(_fma_10);
            s_acc[10] = _exp2_12;
            float _fma_11 = __fmaf_rn(s_acc[11], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_13 = approx_exp2(_fma_11);
            s_acc[11] = _exp2_13;
            float _fma_12 = __fmaf_rn(s_acc[12], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_14 = approx_exp2(_fma_12);
            s_acc[12] = _exp2_14;
            float _fma_13 = __fmaf_rn(s_acc[13], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_15 = approx_exp2(_fma_13);
            s_acc[13] = _exp2_15;
            float _fma_14 = __fmaf_rn(s_acc[14], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_16 = approx_exp2(_fma_14);
            s_acc[14] = _exp2_16;
            float _fma_15 = __fmaf_rn(s_acc[15], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_17 = approx_exp2(_fma_15);
            s_acc[15] = _exp2_17;
            float _fma_16 = __fmaf_rn(s_acc[16], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_18 = approx_exp2(_fma_16);
            s_acc[16] = _exp2_18;
            float _fma_17 = __fmaf_rn(s_acc[17], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_19 = approx_exp2(_fma_17);
            s_acc[17] = _exp2_19;
            float _fma_18 = __fmaf_rn(s_acc[18], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_20 = approx_exp2(_fma_18);
            s_acc[18] = _exp2_20;
            float _fma_19 = __fmaf_rn(s_acc[19], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_21 = approx_exp2(_fma_19);
            s_acc[19] = _exp2_21;
            float _fma_20 = __fmaf_rn(s_acc[20], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_22 = approx_exp2(_fma_20);
            s_acc[20] = _exp2_22;
            float _fma_21 = __fmaf_rn(s_acc[21], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_23 = approx_exp2(_fma_21);
            s_acc[21] = _exp2_23;
            float _fma_22 = __fmaf_rn(s_acc[22], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_24 = approx_exp2(_fma_22);
            s_acc[22] = _exp2_24;
            float _fma_23 = __fmaf_rn(s_acc[23], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_25 = approx_exp2(_fma_23);
            s_acc[23] = _exp2_25;
            float _fma_24 = __fmaf_rn(s_acc[24], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_26 = approx_exp2(_fma_24);
            s_acc[24] = _exp2_26;
            float _fma_25 = __fmaf_rn(s_acc[25], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_27 = approx_exp2(_fma_25);
            s_acc[25] = _exp2_27;
            float _fma_26 = __fmaf_rn(s_acc[26], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_28 = approx_exp2(_fma_26);
            s_acc[26] = _exp2_28;
            float _fma_27 = __fmaf_rn(s_acc[27], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_29 = approx_exp2(_fma_27);
            s_acc[27] = _exp2_29;
            float _fma_28 = __fmaf_rn(s_acc[28], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_30 = approx_exp2(_fma_28);
            s_acc[28] = _exp2_30;
            float _fma_29 = __fmaf_rn(s_acc[29], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_31 = approx_exp2(_fma_29);
            s_acc[29] = _exp2_31;
            float _fma_30 = __fmaf_rn(s_acc[30], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_32 = approx_exp2(_fma_30);
            s_acc[30] = _exp2_32;
            float _fma_31 = __fmaf_rn(s_acc[31], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_33 = approx_exp2(_fma_31);
            s_acc[31] = _exp2_33;
            float _fma_32 = __fmaf_rn(s_acc[32], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_34 = approx_exp2(_fma_32);
            s_acc[32] = _exp2_34;
            float _fma_33 = __fmaf_rn(s_acc[33], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_35 = approx_exp2(_fma_33);
            s_acc[33] = _exp2_35;
            float _fma_34 = __fmaf_rn(s_acc[34], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_36 = approx_exp2(_fma_34);
            s_acc[34] = _exp2_36;
            float _fma_35 = __fmaf_rn(s_acc[35], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_37 = approx_exp2(_fma_35);
            s_acc[35] = _exp2_37;
            float _fma_36 = __fmaf_rn(s_acc[36], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_38 = approx_exp2(_fma_36);
            s_acc[36] = _exp2_38;
            float _fma_37 = __fmaf_rn(s_acc[37], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_39 = approx_exp2(_fma_37);
            s_acc[37] = _exp2_39;
            float _fma_38 = __fmaf_rn(s_acc[38], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_40 = approx_exp2(_fma_38);
            s_acc[38] = _exp2_40;
            float _fma_39 = __fmaf_rn(s_acc[39], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_41 = approx_exp2(_fma_39);
            s_acc[39] = _exp2_41;
            float _fma_40 = __fmaf_rn(s_acc[40], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_42 = approx_exp2(_fma_40);
            s_acc[40] = _exp2_42;
            float _fma_41 = __fmaf_rn(s_acc[41], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_43 = approx_exp2(_fma_41);
            s_acc[41] = _exp2_43;
            float _fma_42 = __fmaf_rn(s_acc[42], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_44 = approx_exp2(_fma_42);
            s_acc[42] = _exp2_44;
            float _fma_43 = __fmaf_rn(s_acc[43], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_45 = approx_exp2(_fma_43);
            s_acc[43] = _exp2_45;
            float _fma_44 = __fmaf_rn(s_acc[44], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_46 = approx_exp2(_fma_44);
            s_acc[44] = _exp2_46;
            float _fma_45 = __fmaf_rn(s_acc[45], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_47 = approx_exp2(_fma_45);
            s_acc[45] = _exp2_47;
            float _fma_46 = __fmaf_rn(s_acc[46], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_48 = approx_exp2(_fma_46);
            s_acc[46] = _exp2_48;
            float _fma_47 = __fmaf_rn(s_acc[47], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_49 = approx_exp2(_fma_47);
            s_acc[47] = _exp2_49;
            float _fma_48 = __fmaf_rn(s_acc[48], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_50 = approx_exp2(_fma_48);
            s_acc[48] = _exp2_50;
            float _fma_49 = __fmaf_rn(s_acc[49], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_51 = approx_exp2(_fma_49);
            s_acc[49] = _exp2_51;
            float _fma_50 = __fmaf_rn(s_acc[50], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_52 = approx_exp2(_fma_50);
            s_acc[50] = _exp2_52;
            float _fma_51 = __fmaf_rn(s_acc[51], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_53 = approx_exp2(_fma_51);
            s_acc[51] = _exp2_53;
            float _fma_52 = __fmaf_rn(s_acc[52], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_54 = approx_exp2(_fma_52);
            s_acc[52] = _exp2_54;
            float _fma_53 = __fmaf_rn(s_acc[53], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_55 = approx_exp2(_fma_53);
            s_acc[53] = _exp2_55;
            float _fma_54 = __fmaf_rn(s_acc[54], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_56 = approx_exp2(_fma_54);
            s_acc[54] = _exp2_56;
            float _fma_55 = __fmaf_rn(s_acc[55], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_57 = approx_exp2(_fma_55);
            s_acc[55] = _exp2_57;
            float _fma_56 = __fmaf_rn(s_acc[56], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_58 = approx_exp2(_fma_56);
            s_acc[56] = _exp2_58;
            float _fma_57 = __fmaf_rn(s_acc[57], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_59 = approx_exp2(_fma_57);
            s_acc[57] = _exp2_59;
            float _fma_58 = __fmaf_rn(s_acc[58], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_60 = approx_exp2(_fma_58);
            s_acc[58] = _exp2_60;
            float _fma_59 = __fmaf_rn(s_acc[59], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_61 = approx_exp2(_fma_59);
            s_acc[59] = _exp2_61;
            float _fma_60 = __fmaf_rn(s_acc[60], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_62 = approx_exp2(_fma_60);
            s_acc[60] = _exp2_62;
            float _fma_61 = __fmaf_rn(s_acc[61], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_66);
            float _exp2_63 = approx_exp2(_fma_61);
            s_acc[61] = _exp2_63;
            float _fma_62 = __fmaf_rn(s_acc[62], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_64 = approx_exp2(_fma_62);
            s_acc[62] = _exp2_64;
            float _fma_63 = __fmaf_rn(s_acc[63], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _fmax_67);
            float _exp2_65 = approx_exp2(_fma_63);
            s_acc[63] = _exp2_65;
            float _fma_64 = __fmaf_rn(l_state[0], _exp2_0, _exp2_2 + _exp2_3 + _exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19 + _exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35 + _exp2_38 + _exp2_39 + _exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55 + _exp2_58 + _exp2_59 + _exp2_62 + _exp2_63);
            l_state[0] = _fma_64;
            float _fma_65 = __fmaf_rn(l_state[1], _exp2_1, _exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17 + _exp2_20 + _exp2_21 + _exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37 + _exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53 + _exp2_56 + _exp2_57 + _exp2_60 + _exp2_61 + _exp2_64 + _exp2_65);
            l_state[1] = _fma_65;
            p_tmp[0] = s_acc[0];
            p_tmp[1] = s_acc[1];
            p_tmp[2] = s_acc[4];
            p_tmp[3] = s_acc[5];
            p_tmp[4] = s_acc[2];
            p_tmp[5] = s_acc[3];
            p_tmp[6] = s_acc[6];
            p_tmp[7] = s_acc[7];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[1] = _packed;
            }
            p_tmp[0] = s_acc[8];
            p_tmp[1] = s_acc[9];
            p_tmp[2] = s_acc[12];
            p_tmp[3] = s_acc[13];
            p_tmp[4] = s_acc[10];
            p_tmp[5] = s_acc[11];
            p_tmp[6] = s_acc[14];
            p_tmp[7] = s_acc[15];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[2] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[3] = _packed;
            }
            p_tmp[0] = s_acc[16];
            p_tmp[1] = s_acc[17];
            p_tmp[2] = s_acc[20];
            p_tmp[3] = s_acc[21];
            p_tmp[4] = s_acc[18];
            p_tmp[5] = s_acc[19];
            p_tmp[6] = s_acc[22];
            p_tmp[7] = s_acc[23];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[4] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[5] = _packed;
            }
            p_tmp[0] = s_acc[24];
            p_tmp[1] = s_acc[25];
            p_tmp[2] = s_acc[28];
            p_tmp[3] = s_acc[29];
            p_tmp[4] = s_acc[26];
            p_tmp[5] = s_acc[27];
            p_tmp[6] = s_acc[30];
            p_tmp[7] = s_acc[31];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[6] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[7] = _packed;
            }
            p_tmp[0] = s_acc[32];
            p_tmp[1] = s_acc[33];
            p_tmp[2] = s_acc[36];
            p_tmp[3] = s_acc[37];
            p_tmp[4] = s_acc[34];
            p_tmp[5] = s_acc[35];
            p_tmp[6] = s_acc[38];
            p_tmp[7] = s_acc[39];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[8] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[9] = _packed;
            }
            p_tmp[0] = s_acc[40];
            p_tmp[1] = s_acc[41];
            p_tmp[2] = s_acc[44];
            p_tmp[3] = s_acc[45];
            p_tmp[4] = s_acc[42];
            p_tmp[5] = s_acc[43];
            p_tmp[6] = s_acc[46];
            p_tmp[7] = s_acc[47];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[10] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[11] = _packed;
            }
            p_tmp[0] = s_acc[48];
            p_tmp[1] = s_acc[49];
            p_tmp[2] = s_acc[52];
            p_tmp[3] = s_acc[53];
            p_tmp[4] = s_acc[50];
            p_tmp[5] = s_acc[51];
            p_tmp[6] = s_acc[54];
            p_tmp[7] = s_acc[55];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[12] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[13] = _packed;
            }
            p_tmp[0] = s_acc[56];
            p_tmp[1] = s_acc[57];
            p_tmp[2] = s_acc[60];
            p_tmp[3] = s_acc[61];
            p_tmp[4] = s_acc[58];
            p_tmp[5] = s_acc[59];
            p_tmp[6] = s_acc[62];
            p_tmp[7] = s_acc[63];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[14] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[15] = _packed;
            }
            mbarrier_wait(v_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            }
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, kv_phase);
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"((kb + 2) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"((kb + 2) * 128), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_7;
        float _rcp_0 = approx_rcp(l0);
        float inv0 = _rcp_0;
        float _rcp_1 = approx_rcp(l1);
        float inv1 = _rcp_1;
        float _vec_load_0[2];
        {
            float2 _v2_1 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + (lane & 3) * 2) + 0);
            _vec_load_0[0] = _v2_1.x;
            _vec_load_0[0 + 1] = _v2_1.y;
        }
        o_tmp[0] = o_acc[0] * inv0 * _vec_load_0[0];
        o_tmp[1] = o_acc[1] * inv0 * _vec_load_0[1];
        o_tmp[2] = o_acc[2] * inv1 * _vec_load_0[0];
        o_tmp[3] = o_acc[3] * inv1 * _vec_load_0[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2)))[0]) = _pk;
            }
        }
        float _vec_load_1[2];
        {
            float2 _v2_2 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 8 + (lane & 3) * 2) + 0);
            _vec_load_1[0] = _v2_2.x;
            _vec_load_1[0 + 1] = _v2_2.y;
        }
        o_tmp[0] = o_acc[4] * inv0 * _vec_load_1[0];
        o_tmp[1] = o_acc[5] * inv0 * _vec_load_1[1];
        o_tmp[2] = o_acc[6] * inv1 * _vec_load_1[0];
        o_tmp[3] = o_acc[7] * inv1 * _vec_load_1[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 8)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 8)))[0]) = _pk;
            }
        }
        float _vec_load_2[2];
        {
            float2 _v2_3 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 16 + (lane & 3) * 2) + 0);
            _vec_load_2[0] = _v2_3.x;
            _vec_load_2[0 + 1] = _v2_3.y;
        }
        o_tmp[0] = o_acc[8] * inv0 * _vec_load_2[0];
        o_tmp[1] = o_acc[9] * inv0 * _vec_load_2[1];
        o_tmp[2] = o_acc[10] * inv1 * _vec_load_2[0];
        o_tmp[3] = o_acc[11] * inv1 * _vec_load_2[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 16)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 16)))[0]) = _pk;
            }
        }
        float _vec_load_3[2];
        {
            float2 _v2_4 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 24 + (lane & 3) * 2) + 0);
            _vec_load_3[0] = _v2_4.x;
            _vec_load_3[0 + 1] = _v2_4.y;
        }
        o_tmp[0] = o_acc[12] * inv0 * _vec_load_3[0];
        o_tmp[1] = o_acc[13] * inv0 * _vec_load_3[1];
        o_tmp[2] = o_acc[14] * inv1 * _vec_load_3[0];
        o_tmp[3] = o_acc[15] * inv1 * _vec_load_3[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 24)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 24)))[0]) = _pk;
            }
        }
        float _vec_load_4[2];
        {
            float2 _v2_5 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 32 + (lane & 3) * 2) + 0);
            _vec_load_4[0] = _v2_5.x;
            _vec_load_4[0 + 1] = _v2_5.y;
        }
        o_tmp[0] = o_acc[16] * inv0 * _vec_load_4[0];
        o_tmp[1] = o_acc[17] * inv0 * _vec_load_4[1];
        o_tmp[2] = o_acc[18] * inv1 * _vec_load_4[0];
        o_tmp[3] = o_acc[19] * inv1 * _vec_load_4[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 32)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 32)))[0]) = _pk;
            }
        }
        float _vec_load_5[2];
        {
            float2 _v2_6 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 40 + (lane & 3) * 2) + 0);
            _vec_load_5[0] = _v2_6.x;
            _vec_load_5[0 + 1] = _v2_6.y;
        }
        o_tmp[0] = o_acc[20] * inv0 * _vec_load_5[0];
        o_tmp[1] = o_acc[21] * inv0 * _vec_load_5[1];
        o_tmp[2] = o_acc[22] * inv1 * _vec_load_5[0];
        o_tmp[3] = o_acc[23] * inv1 * _vec_load_5[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 40)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 40)))[0]) = _pk;
            }
        }
        float _vec_load_6[2];
        {
            float2 _v2_7 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 48 + (lane & 3) * 2) + 0);
            _vec_load_6[0] = _v2_7.x;
            _vec_load_6[0 + 1] = _v2_7.y;
        }
        o_tmp[0] = o_acc[24] * inv0 * _vec_load_6[0];
        o_tmp[1] = o_acc[25] * inv0 * _vec_load_6[1];
        o_tmp[2] = o_acc[26] * inv1 * _vec_load_6[0];
        o_tmp[3] = o_acc[27] * inv1 * _vec_load_6[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 48)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 48)))[0]) = _pk;
            }
        }
        float _vec_load_7[2];
        {
            float2 _v2_8 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 56 + (lane & 3) * 2) + 0);
            _vec_load_7[0] = _v2_8.x;
            _vec_load_7[0 + 1] = _v2_8.y;
        }
        o_tmp[0] = o_acc[28] * inv0 * _vec_load_7[0];
        o_tmp[1] = o_acc[29] * inv0 * _vec_load_7[1];
        o_tmp[2] = o_acc[30] * inv1 * _vec_load_7[0];
        o_tmp[3] = o_acc[31] * inv1 * _vec_load_7[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 56)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 56)))[0]) = _pk;
            }
        }
        float _vec_load_8[2];
        {
            float2 _v2_9 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 64 + (lane & 3) * 2) + 0);
            _vec_load_8[0] = _v2_9.x;
            _vec_load_8[0 + 1] = _v2_9.y;
        }
        o_tmp[0] = o_acc[32] * inv0 * _vec_load_8[0];
        o_tmp[1] = o_acc[33] * inv0 * _vec_load_8[1];
        o_tmp[2] = o_acc[34] * inv1 * _vec_load_8[0];
        o_tmp[3] = o_acc[35] * inv1 * _vec_load_8[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 64)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 64)))[0]) = _pk;
            }
        }
        float _vec_load_9[2];
        {
            float2 _v2_10 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 72 + (lane & 3) * 2) + 0);
            _vec_load_9[0] = _v2_10.x;
            _vec_load_9[0 + 1] = _v2_10.y;
        }
        o_tmp[0] = o_acc[36] * inv0 * _vec_load_9[0];
        o_tmp[1] = o_acc[37] * inv0 * _vec_load_9[1];
        o_tmp[2] = o_acc[38] * inv1 * _vec_load_9[0];
        o_tmp[3] = o_acc[39] * inv1 * _vec_load_9[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 72)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 72)))[0]) = _pk;
            }
        }
        float _vec_load_10[2];
        {
            float2 _v2_11 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 80 + (lane & 3) * 2) + 0);
            _vec_load_10[0] = _v2_11.x;
            _vec_load_10[0 + 1] = _v2_11.y;
        }
        o_tmp[0] = o_acc[40] * inv0 * _vec_load_10[0];
        o_tmp[1] = o_acc[41] * inv0 * _vec_load_10[1];
        o_tmp[2] = o_acc[42] * inv1 * _vec_load_10[0];
        o_tmp[3] = o_acc[43] * inv1 * _vec_load_10[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 80)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 80)))[0]) = _pk;
            }
        }
        float _vec_load_11[2];
        {
            float2 _v2_12 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 88 + (lane & 3) * 2) + 0);
            _vec_load_11[0] = _v2_12.x;
            _vec_load_11[0 + 1] = _v2_12.y;
        }
        o_tmp[0] = o_acc[44] * inv0 * _vec_load_11[0];
        o_tmp[1] = o_acc[45] * inv0 * _vec_load_11[1];
        o_tmp[2] = o_acc[46] * inv1 * _vec_load_11[0];
        o_tmp[3] = o_acc[47] * inv1 * _vec_load_11[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 88)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 88)))[0]) = _pk;
            }
        }
        float _vec_load_12[2];
        {
            float2 _v2_13 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 96 + (lane & 3) * 2) + 0);
            _vec_load_12[0] = _v2_13.x;
            _vec_load_12[0 + 1] = _v2_13.y;
        }
        o_tmp[0] = o_acc[48] * inv0 * _vec_load_12[0];
        o_tmp[1] = o_acc[49] * inv0 * _vec_load_12[1];
        o_tmp[2] = o_acc[50] * inv1 * _vec_load_12[0];
        o_tmp[3] = o_acc[51] * inv1 * _vec_load_12[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 96)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 96)))[0]) = _pk;
            }
        }
        float _vec_load_13[2];
        {
            float2 _v2_14 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 104 + (lane & 3) * 2) + 0);
            _vec_load_13[0] = _v2_14.x;
            _vec_load_13[0 + 1] = _v2_14.y;
        }
        o_tmp[0] = o_acc[52] * inv0 * _vec_load_13[0];
        o_tmp[1] = o_acc[53] * inv0 * _vec_load_13[1];
        o_tmp[2] = o_acc[54] * inv1 * _vec_load_13[0];
        o_tmp[3] = o_acc[55] * inv1 * _vec_load_13[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 104)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 104)))[0]) = _pk;
            }
        }
        float _vec_load_14[2];
        {
            float2 _v2_15 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 112 + (lane & 3) * 2) + 0);
            _vec_load_14[0] = _v2_15.x;
            _vec_load_14[0 + 1] = _v2_15.y;
        }
        o_tmp[0] = o_acc[56] * inv0 * _vec_load_14[0];
        o_tmp[1] = o_acc[57] * inv0 * _vec_load_14[1];
        o_tmp[2] = o_acc[58] * inv1 * _vec_load_14[0];
        o_tmp[3] = o_acc[59] * inv1 * _vec_load_14[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 112)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 112)))[0]) = _pk;
            }
        }
        float _vec_load_15[2];
        {
            float2 _v2_16 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 120 + (lane & 3) * 2) + 0);
            _vec_load_15[0] = _v2_16.x;
            _vec_load_15[0 + 1] = _v2_16.y;
        }
        o_tmp[0] = o_acc[60] * inv0 * _vec_load_15[0];
        o_tmp[1] = o_acc[61] * inv0 * _vec_load_15[1];
        o_tmp[2] = o_acc[62] * inv1 * _vec_load_15[0];
        o_tmp[3] = o_acc[63] * inv1 * _vec_load_15[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 120)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 120)))[0]) = _pk;
            }
        }
        __syncthreads();
    }

    // Cleanup
    __syncthreads();
}

}  // namespace h3_varlen_attention_fp8_sm120a
#undef H3_VARLEN_INF
#undef NUM_KV_STAGES
#undef SMEM_K_STAGE_OFF
#undef SMEM_K_STAGE_STAGE_BYTES
#undef SMEM_K_STAGE_STRIDE
#undef SMEM_Q_OFF
#undef SMEM_Q_STAGE_BYTES
#undef SMEM_Q_STRIDE
#undef SMEM_TOTAL
#undef SMEM_V_STAGE_OFF
#undef SMEM_V_STAGE_STAGE_BYTES
#undef SMEM_V_STAGE_STRIDE
#undef THREADS
#undef k_empty_addr
#undef k_full_addr
#undef q_full_addr
#undef v_empty_addr
#undef v_full_addr

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlockM = 128;
constexpr int64_t kBlockN = 128;
constexpr int64_t kMaxHeads = 32768;
constexpr int kStatsThreads = 256;
constexpr int kFinalizeThreads = 256;
constexpr int kQuantThreads = 256;
constexpr int kAttentionThreads = 256;
constexpr int kStatsSmemBytes = 16384;
constexpr int kQuantSmemBytes = 128;
constexpr int kAttentionSmemBytes = 82944;  // Q tile + two-stage K/V^T ring + mbarriers
// Statistics partials: per (Q tile, head) 128 K channel sums followed by 128 V channel maxima.
constexpr int64_t kPartialFloats = 2 * kHeadDim;

struct DeviceInfo {
  int num_sms;
};

DeviceInfo ConfigureKernels() {
  static std::mutex mutex;
  static std::vector<std::pair<int, DeviceInfo>> configured;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : configured) {
    if (entry.first == device) return entry.second;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 12, RuntimeError)
      << "MiniMax-H3 SM120 FP8 varlen attention requires compute capability 12.x (GB202)";
  status = cudaFuncSetAttribute(h3_varlen_attention_fp8_sm120a::kernel_minimax_h3_sm120_varlen_attention_fp8,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kAttentionSmemBytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
  DeviceInfo info{properties.multiProcessorCount};
  configured.emplace_back(device, info);
  return info;
}

void CheckDevice(const TensorView& tensor, const char* name, DLDevice device) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as q";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

void CheckThd(const TensorView& tensor, const char* name, int64_t tokens, int64_t heads, DLDevice device) {
  CheckDevice(tensor, name, device);
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dl_bfloat16), ValueError)
      << name << " must be bfloat16";
  TVM_FFI_CHECK(tensor.ndim() == 3 && tensor.size(0) == tokens && tensor.size(1) == heads &&
                    tensor.size(2) == kHeadDim,
                ValueError)
      << name << " must have shape [tokens, heads, 128]";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}

void CheckFlat(const TensorView& tensor, const char* name, DLDataType dtype, const char* dtype_name,
               int64_t min_numel, DLDevice device) {
  CheckDevice(tensor, name, device);
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " must be " << dtype_name;
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.IsContiguous() && tensor.size(0) >= min_numel, ValueError)
      << name << " must be a contiguous 1-D " << dtype_name << " tensor with at least " << min_numel
      << " elements";
}

// One head's [128 channels x box_rows tokens] E4M3 tile of a [tokens, heads, 128] byte tensor,
// addressed as the 3-D view (128 channels, tokens, heads) with a 128-byte swizzle.  The token box
// may run past the tensor: TMA zero-fills those rows, the kernel masks those keys / never stores
// those query rows.
CUtensorMap EncodeRowsTile(const TensorView& rows, int64_t tokens, int64_t heads, uint32_t box_rows,
                           const char* name) {
  uint64_t global_dim[3] = {static_cast<uint64_t>(kHeadDim), static_cast<uint64_t>(tokens),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(heads * kHeadDim), static_cast<uint64_t>(kHeadDim)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(kHeadDim), box_rows, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, rows.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

// One head's [128 channel rows x 128 keys] tile of the transposed V^T byte tensor [heads, 128,
// padded_tokens]: 3-D view (padded tokens, channels, heads), 128-byte swizzle.
CUtensorMap EncodeTransposedTile(const TensorView& vt, int64_t padded_tokens, int64_t heads, const char* name) {
  uint64_t global_dim[3] = {static_cast<uint64_t>(padded_tokens), static_cast<uint64_t>(kHeadDim),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(padded_tokens),
                                static_cast<uint64_t>(padded_tokens * kHeadDim)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(kBlockN), static_cast<uint32_t>(kHeadDim), 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, vt.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

}  // namespace

// out[a:b, h] = softmax(q[a:b, h] k[a:b, h]^T * softmax_scale) v[a:b, h] for every segment [a, b) of
// cu_seqlens and every head h, with E4M3 Q / K / P / V operands and an FP32 softmax.
//   q, k, v, out: contiguous BF16 [tokens, heads, 128]; cu_seqlens: int32 [segments + 1] on the device.
//   Plan (built on the host from cu_seqlens): seg_begin / seg_len int32 [segments], seg_tile_begin int32
//   [segments + 1], tile_table int32 [2 * num_tiles] (segment, tile), unit_table int32 [2 * num_units]
//   (segment, head << 16 | q_tile) in persistent-grid slot order.
//   Workspaces: q8, k8 uint8 [>= tokens * heads * 128], vt8 uint8 [>= heads * 128 * padded_tokens],
//   q_scale f32 [>= tokens * heads], k_scale f32 [>= heads * num_kblocks], v_scale and mean_k f32
//   [>= segments * heads * 128], partials f32 [>= num_tiles * heads * 256].
void minimax_h3_sm120_varlen_attention_fp8(TensorView q, TensorView k, TensorView v, TensorView cu_seqlens,
                                           TensorView out, TensorView seg_begin, TensorView seg_len,
                                           TensorView seg_tile_begin, TensorView tile_table, TensorView unit_table,
                                           TensorView q8, TensorView k8, TensorView vt8, TensorView q_scale,
                                           TensorView k_scale, TensorView v_scale, TensorView mean_k,
                                           TensorView partials, int64_t num_segments, int64_t num_tiles,
                                           int64_t num_units, int64_t attention_grid, double softmax_scale) {
  TVM_FFI_CHECK(q.ndim() == 3, ValueError) << "q must be [tokens, heads, 128]";
  const int64_t tokens = q.size(0);
  const int64_t heads = q.size(1);
  TVM_FFI_CHECK(tokens >= 0 && heads >= 1 && heads < kMaxHeads, ValueError)
      << "q must be [tokens, heads, 128] with 1 <= heads < " << kMaxHeads;
  const DLDevice device = q.device();
  CheckThd(q, "q", tokens, heads, device);
  CheckThd(k, "k", tokens, heads, device);
  CheckThd(v, "v", tokens, heads, device);
  CheckThd(out, "out", tokens, heads, device);
  TVM_FFI_CHECK(num_segments >= 1 && num_tiles >= 0 && num_units >= 0 && attention_grid >= 1, ValueError)
      << "invalid segment plan";
  const int64_t num_kblocks = std::max<int64_t>(1, (tokens + kBlockN - 1) / kBlockN);
  const int64_t padded_tokens = num_kblocks * kBlockN;
  CheckFlat(cu_seqlens, "cu_seqlens", dl_int32, "int32", num_segments + 1, device);
  CheckFlat(seg_begin, "seg_begin", dl_int32, "int32", std::max<int64_t>(1, num_segments), device);
  CheckFlat(seg_len, "seg_len", dl_int32, "int32", std::max<int64_t>(1, num_segments), device);
  CheckFlat(seg_tile_begin, "seg_tile_begin", dl_int32, "int32", num_segments + 1, device);
  CheckFlat(tile_table, "tile_table", dl_int32, "int32", std::max<int64_t>(2, 2 * num_tiles), device);
  CheckFlat(unit_table, "unit_table", dl_int32, "int32", std::max<int64_t>(2, 2 * num_units), device);
  CheckFlat(q8, "q8", dl_uint8, "uint8", tokens * heads * kHeadDim, device);
  CheckFlat(k8, "k8", dl_uint8, "uint8", tokens * heads * kHeadDim, device);
  CheckFlat(vt8, "vt8", dl_uint8, "uint8", heads * kHeadDim * padded_tokens, device);
  CheckFlat(q_scale, "q_scale", dl_float32, "float32", tokens * heads, device);
  CheckFlat(k_scale, "k_scale", dl_float32, "float32", heads * num_kblocks, device);
  CheckFlat(v_scale, "v_scale", dl_float32, "float32", num_segments * heads * kHeadDim, device);
  CheckFlat(mean_k, "mean_k", dl_float32, "float32", num_segments * heads * kHeadDim, device);
  CheckFlat(partials, "partials", dl_float32, "float32", std::max<int64_t>(1, num_tiles * heads * kPartialFloats), device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const DeviceInfo info = ConfigureKernels();
  if (tokens == 0 || num_units == 0) return;  // no rows to write (all segments empty)
  const int cta_cap = 4 * info.num_sms;
  const int heads_i = static_cast<int>(heads);

  if (num_tiles > 0) {
    const int stats_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(num_tiles * heads, cta_cap)));
    h3_varlen_kv_stats_sm120a::kernel_minimax_h3_sm120_varlen_kv_stats<<<stats_grid, kStatsThreads, kStatsSmemBytes,
                                                                       stream>>>(
        static_cast<__nv_bfloat16*>(k.data_ptr()), static_cast<__nv_bfloat16*>(v.data_ptr()),
        static_cast<int*>(tile_table.data_ptr()), static_cast<int*>(seg_begin.data_ptr()),
        static_cast<int*>(seg_len.data_ptr()), static_cast<float*>(partials.data_ptr()), static_cast<int>(num_tiles),
        heads_i);
    const int finalize_grid =
        static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(num_segments * heads, cta_cap)));
    h3_varlen_kv_stats_finalize_sm120a::kernel_minimax_h3_sm120_varlen_kv_stats_finalize<<<finalize_grid, kFinalizeThreads,
                                                                                         0, stream>>>(
        static_cast<float*>(partials.data_ptr()), static_cast<int*>(seg_tile_begin.data_ptr()),
        static_cast<int*>(seg_len.data_ptr()), static_cast<float*>(mean_k.data_ptr()),
        static_cast<float*>(v_scale.data_ptr()), static_cast<int>(num_segments), heads_i);
  }
  const int quant_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(num_kblocks * heads, cta_cap)));
  h3_varlen_quantize_fp8_sm120a::kernel_minimax_h3_sm120_varlen_quantize_fp8<<<quant_grid, kQuantThreads, kQuantSmemBytes,
                                                                             stream>>>(
      static_cast<__nv_bfloat16*>(q.data_ptr()), static_cast<__nv_bfloat16*>(k.data_ptr()),
      static_cast<__nv_bfloat16*>(v.data_ptr()), static_cast<int*>(cu_seqlens.data_ptr()),
      static_cast<float*>(mean_k.data_ptr()), static_cast<float*>(v_scale.data_ptr()),
      static_cast<unsigned int*>(q8.data_ptr()), static_cast<unsigned int*>(k8.data_ptr()),
      static_cast<unsigned int*>(vt8.data_ptr()), static_cast<float*>(q_scale.data_ptr()),
      static_cast<float*>(k_scale.data_ptr()), static_cast<int>(tokens), static_cast<int>(padded_tokens),
      static_cast<int>(num_segments), heads_i, static_cast<int>(num_kblocks));

  const CUtensorMap q_map = EncodeRowsTile(q8, tokens, heads, static_cast<uint32_t>(kBlockM), "q8");
  const CUtensorMap k_map = EncodeRowsTile(k8, tokens, heads, static_cast<uint32_t>(kBlockN), "k8");
  const CUtensorMap v_map = EncodeTransposedTile(vt8, padded_tokens, heads, "vt8");
  const int grid = static_cast<int>(std::min<int64_t>(attention_grid, std::max<int64_t>(1, num_units)));
  const float softmax_scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  h3_varlen_attention_fp8_sm120a::kernel_minimax_h3_sm120_varlen_attention_fp8<<<grid, kAttentionThreads,
                                                                                 kAttentionSmemBytes, stream>>>(
      q_map, k_map, v_map, static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<float*>(q_scale.data_ptr()),
      static_cast<float*>(k_scale.data_ptr()), static_cast<float*>(v_scale.data_ptr()),
      static_cast<int*>(seg_begin.data_ptr()), static_cast<int*>(seg_len.data_ptr()),
      static_cast<int*>(unit_table.data_ptr()), static_cast<int>(num_units), heads_i, static_cast<int>(num_kblocks),
      softmax_scale_log2);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 SM120 FP8 varlen attention launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_varlen_attention_fp8, minimax_h3_sm120_varlen_attention_fp8);
// clang-format on
