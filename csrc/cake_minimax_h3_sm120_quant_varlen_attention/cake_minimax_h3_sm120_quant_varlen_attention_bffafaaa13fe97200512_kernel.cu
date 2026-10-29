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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_1_STAGES 1
#define SMEM_RED_SUM_OFF 0
#define SMEM_RED_SUM_STAGE_BYTES 16384
#define SMEM_RED_SUM_STRIDE 16384
#define SMEM_RED_MAX_OFF 16384
#define SMEM_RED_MAX_STAGE_BYTES 16384
#define SMEM_RED_MAX_STRIDE 16384
#define SMEM_FLAG_OFF 32768
#define SMEM_FLAG_STAGE_BYTES 32
#define SMEM_FLAG_STRIDE 32
#define SMEM_AMAX_PART_OFF 32800
#define SMEM_AMAX_PART_STAGE_BYTES 64
#define SMEM_AMAX_PART_STRIDE 64
#define SMEM_ITEM_CELL_OFF 0
#define SMEM_ITEM_CELL_STAGE_BYTES 32
#define SMEM_ITEM_CELL_STRIDE 32
#define SMEM_TOTAL 33024
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_quant_varlen_attention_bffafaaa13fe97200512(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ seg_tile_begin, int* __restrict__ cu_seqlens, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ v_scale, unsigned int* __restrict__ q8, unsigned int* __restrict__ k8, unsigned int* __restrict__ vt8, float* __restrict__ q_scale, float* __restrict__ k_scale, unsigned int* __restrict__ hctr, int num_tiles, int total_tokens, int total_padded, int num_segments, int num_heads, int num_kblocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[32];
    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* red_sum = reinterpret_cast<float*>(smem_raw + 0);
    const int red_sum_addr = smem + 0;
    float* red_max = reinterpret_cast<float*>(smem_raw + 16384);
    const int red_max_addr = smem + 16384;
    int* flag = reinterpret_cast<int*>(smem_raw + 32768);
    const int flag_addr = smem + 32768;
    float* amax_part = reinterpret_cast<float*>(smem_raw + 32800);
    const int amax_part_addr = smem + 32800;
    int* item_cell = reinterpret_cast<int*>(smem_static_raw + 0);
    const int item_cell_addr = (int)(unsigned long long)__cvta_generic_to_shared(item_cell);

    // ---- Role: wk0 ----
    if (warp <= 7) {
        { // wk0_main
            int wtid = tid;
            int col = (wtid & 15) * 8;
            int rgrp = wtid >> 4;
            int sub = wtid & 7;
            int trow = wtid >> 3;
            int col16 = sub * 16;
            int _count_leading_zeros_0 = __clz((unsigned int)(num_segments - 1));
            int nsteps = 32 - _count_leading_zeros_0;
            int per = num_tiles + num_kblocks;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int nxt_r[2];
            if (wtid == 0) {
                unsigned int _atomic_fetch_add_1 = atomicAdd(&hctr[4 * num_heads + 4], 1);
                item_cell[0] = (int)_atomic_fetch_add_1;
            }
            asm volatile("bar.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int it = 0; it < per * num_heads; it++) {
                int w = item_cell[0 + (it & 1)];
                if (w >= per * num_heads) {
                    break;
                }
                if (wtid == 0) {
                    unsigned int _atomic_fetch_add_2 = atomicAdd(&hctr[4 * num_heads + 4], 1);
                    nxt_r[0] = _atomic_fetch_add_2;
                }
                int w2 = ((w >= num_tiles) ? w - num_tiles : 0);
                int s = w2 / per;
                int r = w2 - s * per;
                int last_s = ((s + 1 >= num_heads) ? 1 : 0);
                int is_q = ((w >= num_tiles && (last_s == 1 || r >= num_tiles)) ? 1 : 0);
                int head = ((w < num_tiles) ? 0 : ((is_q == 1) ? s : s + 1));
                int ri = ((w < num_tiles) ? w : ((last_s == 1) ? r : ((is_q == 1) ? r - num_tiles : r)));
                if (is_q == 0) {
                    int s_work = ri * num_heads + head;
                    int s_seg = tile_table[2 * ri];
                    int s_tile_in_seg = tile_table[2 * ri + 1];
                    int s_sb = seg_begin[s_seg];
                    int s_length = seg_len[s_seg];
                    int s_t0 = seg_tile_begin[s_seg];
                    int s_t1 = seg_tile_begin[s_seg + 1];
                    int s_row_begin = s_sb + s_tile_in_seg * 128;
                    int s_seg_end = s_sb + s_length;
                    int s_row_end = ((s_seg_end > s_row_begin + 128) ? s_row_begin + 128 : s_seg_end);
                    float ksum[8];
                    float vmax[8];
                    for (int e = 0; e < 8; e++) {
                        ksum[e] = 0.0f;
                        vmax[e] = 0.0f;
                    }
                    #pragma unroll 1
                    for (int row = s_row_begin + rgrp; row < s_row_end; row += 16) {
                        int s_base = (row * num_heads + head) * 128 + col;
                        float _vec_load_s0[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(K + s_base + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_s0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_s0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_s1[8];
                        {
                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(V + s_base + 0);
                            uint4 _vld_4[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_4[_blk] = _vptr_4[_blk];
                                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_s1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                    (&_vec_load_s1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        for (int e_1 = 0; e_1 < 8; e_1++) {
                            float s_kf = _vec_load_s0[e_1];
                            float s_vf = _vec_load_s1[e_1];
                            ksum[e_1] = ksum[e_1] + s_kf;
                            vmax[e_1] = fmaxf(vmax[e_1], fabsf(s_vf));
                        }
                    }
                    for (int e_2 = 0; e_2 < 8; e_2++) {
                        red_sum[0 + rgrp * 128 + col + e_2] = ksum[e_2];
                        red_max[0 + rgrp * 128 + col + e_2] = vmax[e_2];
                    }
                    asm volatile("bar.sync 1, 256;" ::: "memory");
                    int single = ((s_t1 - s_t0 == 1) ? 1 : 0);
                    if (wtid < 128) {
                        float total = 0.0f;
                        float vmx = 0.0f;
                        for (int g = 0; g < 16; g++) {
                            total = total + red_sum[0 + g * 128 + wtid];
                            vmx = fmaxf(vmx, red_max[0 + g * 128 + wtid]);
                        }
                        if (single == 1) {
                            float msum = 0.0f;
                            float mmax = 0.0f;
                            msum = msum + total;
                            mmax = fmaxf(mmax, vmx);
                            float _fdiv_5 = __fdiv_rn(msum, (float)s_length);
                            *(reinterpret_cast<float*>(mean_k + ((s_seg * num_heads + head) * 128 + wtid)) + (0)) = _fdiv_5;
                            float _fdiv_6 = __fdiv_rn(fmaxf(mmax, 1e-12f), 448.0f);
                            *(reinterpret_cast<float*>(v_scale + ((s_seg * num_heads + head) * 128 + wtid)) + (0)) = _fdiv_6;
                        } else {
                            *(reinterpret_cast<float*>(partials + (s_work * 256 + wtid)) + (0)) = total;
                            *(reinterpret_cast<float*>(partials + (s_work * 256 + 128 + wtid)) + (0)) = vmx;
                        }
                    }
                    if (single == 0) {
                        asm volatile("bar.sync 1, 256;" ::: "memory");
                        if (wtid == 0) {
                            unsigned int old;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(old) : "l"(&counters[s_seg * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
                            int last = (((int)old == s_t1 - s_t0 - 1) ? 1 : 0);
                            flag[0] = last;
                            if (last == 1) {
                                *(reinterpret_cast<unsigned int*>(counters + (s_seg * num_heads + head)) + (0)) = (unsigned int)0;
                            }
                        }
                        asm volatile("bar.sync 1, 256;" ::: "memory");
                        if (flag[0] == 1) {
                            int chan = wtid & 127;
                            int is_max = wtid >> 7;
                            float acc[1];
                            float fold_v[8];
                            acc[0] = 0.0f;
                            #pragma unroll 1
                            for (int tt = s_t0; tt < s_t1; tt += 8) {
                                for (int j = 0; j < 8; j++) {
                                    int ttj = ((s_t1 > tt + j) ? tt + j : s_t1 - 1);
                                    fold_v[j] = partials[(ttj * num_heads + head) * 256 + is_max * 128 + chan];
                                }
                                for (int j_1 = 0; j_1 < 8; j_1++) {
                                    if (s_t1 > tt + j_1) {
                                        acc[0] = ((is_max == 0) ? acc[0] + fold_v[j_1] : fmaxf(acc[0], fold_v[j_1]));
                                    }
                                }
                            }
                            if (is_max == 0) {
                                float _fdiv_7 = __fdiv_rn(acc[0], (float)s_length);
                                *(reinterpret_cast<float*>(mean_k + ((s_seg * num_heads + head) * 128 + chan)) + (0)) = _fdiv_7;
                            } else {
                                float _fdiv_8 = __fdiv_rn(fmaxf(acc[0], 1e-12f), 448.0f);
                                *(reinterpret_cast<float*>(v_scale + ((s_seg * num_heads + head) * 128 + chan)) + (0)) = _fdiv_8;
                            }
                        }
                    }
                    asm volatile("bar.sync 1, 256;" ::: "memory");
                    if (wtid == 0) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                } else {
                    if (wtid == 0) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(num_tiles)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("bar.sync 1, 256;" ::: "memory");
                    int tok_base = ri * 128;
                    float kvals[64];
                    float kmax = 0.0f;
                    for (int ps = 0; ps < 4; ps++) {
                        int tok = tok_base + ps * 32 + trow;
                        int valid = ((tok < total_tokens) ? 1 : 0);
                        int tok_c = ((tok < total_tokens) ? tok : total_tokens - 1);
                        int base = (tok_c * num_heads + head) * 128 + col16;
                        float _vec_load_0[16];
                        {
                            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(Q + base + 0);
                            uint4 _vld_9[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vld_9[_blk].x), "=r"(_vld_9[_blk].y), "=r"(_vld_9[_blk].z), "=r"(_vld_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_1[16];
                        {
                            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(K + base + 0);
                            uint4 _vld_10[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vld_10[_blk].x), "=r"(_vld_10[_blk].y), "=r"(_vld_10[_blk].z), "=r"(_vld_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        int lo = 0;
                        int hi = num_segments - 1;
                        #pragma unroll 1
                        for (int step = 0; step < nsteps; step++) {
                            int mid = lo + hi + 1 >> 1;
                            if (tok_c >= cu_seqlens[mid]) {
                                lo = mid;
                            } else {
                                hi = mid - 1;
                            }
                        }
                        int seg = lo;
                        float qamax = 0.0f;
                        for (int e_3 = 0; e_3 < 16; e_3++) {
                            float qf = _vec_load_0[e_3];
                            qamax = fmaxf(qamax, fabsf(qf));
                        }
                        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, qamax, 4);
                        qamax = fmaxf(qamax, _shfl_xor_sync_11);
                        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, qamax, 2);
                        qamax = fmaxf(qamax, _shfl_xor_sync_12);
                        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, qamax, 1);
                        qamax = fmaxf(qamax, _shfl_xor_sync_13);
                        qamax = fmaxf(qamax, 1e-12f);
                        float qs = __fdiv_rn(qamax, 448.0f);
                        float qinv = __fdiv_rn(448.0f, qamax);
                        float qn[16];
                        for (int e_4 = 0; e_4 < 16; e_4++) {
                            float qf2 = _vec_load_0[e_4];
                            qn[e_4] = qf2 * qinv;
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
                            for (int e_5 = 0; e_5 < 4; e_5++) {
                                float kf = _vec_load_1[quad * 4 + e_5];
                                float kc = kf - _vec_load_2[e_5];
                                kvals[ps * 16 + quad * 4 + e_5] = ((valid == 1) ? kc : 0.0f);
                                kmax = fmaxf(kmax, fabsf(kvals[ps * 16 + quad * 4 + e_5]));
                            }
                        }
                    }
                    for (int stage = 0; stage < 5; stage++) {
                        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, kmax, 16 >> stage);
                        kmax = fmaxf(kmax, _shfl_xor_sync_15);
                    }
                    if (lane == 0) {
                        amax_part[0 + (warp & 7)] = kmax;
                    }
                    asm volatile("bar.sync 1, 256;" ::: "memory");
                    kmax = amax_part[0 + (lane & 7)];
                    for (int stage_1 = 0; stage_1 < 3; stage_1++) {
                        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, kmax, 4 >> stage_1);
                        kmax = fmaxf(kmax, _shfl_xor_sync_16);
                    }
                    kmax = fmaxf(kmax, 1e-12f);
                    float kinv = __fdiv_rn(448.0f, kmax);
                    if (wtid == 0) {
                        float _fdiv_17 = __fdiv_rn(kmax, 448.0f);
                        *(reinterpret_cast<float*>(k_scale + (head * num_kblocks + ri)) + (0)) = _fdiv_17;
                    }
                    for (int ps_1 = 0; ps_1 < 4; ps_1++) {
                        int tok2 = tok_base + ps_1 * 32 + trow;
                        float kn[16];
                        for (int e_6 = 0; e_6 < 16; e_6++) {
                            kn[e_6] = kvals[ps_1 * 16 + e_6] * kinv;
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
                    int task = wtid;
                    int cg = task >> 5;
                    int chunk = (task & 31) >> 2;
                    int t8 = task & 3;
                    int c0 = cg * 16;
                    float vq[64];
                    for (int i = 0; i < 4; i++) {
                        int key_local = chunk * 16 + (i >> 1) * 8 + 2 * t8 + (i & 1);
                        int vtok = tok_base + key_local;
                        int vvalid = ((vtok < total_tokens) ? 1 : 0);
                        int vtok_c = ((vtok < total_tokens) ? vtok : total_tokens - 1);
                        float _vec_load_3[16];
                        {
                            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(V + ((vtok_c * num_heads + head) * 128 + c0) + 0);
                            uint4 _vld_18[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vld_18[_blk].x), "=r"(_vld_18[_blk].y), "=r"(_vld_18[_blk].z), "=r"(_vld_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        int vlo = 0;
                        int vhi = num_segments - 1;
                        #pragma unroll 1
                        for (int step_1 = 0; step_1 < nsteps; step_1++) {
                            int vmid = vlo + vhi + 1 >> 1;
                            if (vtok_c >= cu_seqlens[vmid]) {
                                vlo = vmid;
                            } else {
                                vhi = vmid - 1;
                            }
                        }
                        int vseg = vlo;
                        int sbase = (vseg * num_heads + head) * 128 + c0;
                        for (int quad_1 = 0; quad_1 < 4; quad_1++) {
                            float _vec_load_4[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (sbase + quad_1 * 4) + 0);
                                _vec_load_4[0 + 0] = _v4.x;
                                _vec_load_4[0 + 1] = _v4.y;
                                _vec_load_4[0 + 2] = _v4.z;
                                _vec_load_4[0 + 3] = _v4.w;
                            }
                            for (int e_7 = 0; e_7 < 4; e_7++) {
                                float vf = _vec_load_3[quad_1 * 4 + e_7];
                                float _fdiv_20 = __fdiv_rn(vf, _vec_load_4[e_7]);
                                vq[i * 16 + quad_1 * 4 + e_7] = ((vvalid == 1) ? _fdiv_20 : 0.0f);
                            }
                        }
                    }
                    int vt_word = (head * 128 + c0) * total_padded + tok_base + chunk * 16 + 4 * t8 >> 2;
                    for (int e_8 = 0; e_8 < 16; e_8++) {
                        float four[4];
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            four[i_1] = vq[i_1 * 16 + e_8];
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
                        *(reinterpret_cast<unsigned int*>(vt8 + (vt_word + e_8 * (total_padded >> 2))) + (0)) = word[0];
                    }
                }
                if (wtid == 0) {
                    item_cell[0 + (it + 1 & 1)] = (int)nxt_r[0];
                }
                asm volatile("bar.sync 1, 256;" ::: "memory");
                if (wtid == 0) {
                    if (is_q == 1) {
                        unsigned int qold;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(qold) : "l"(&hctr[2 * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
                        if ((int)qold == num_kblocks - 1) {
                            *(reinterpret_cast<unsigned int*>(hctr + (num_heads + head)) + (0)) = (unsigned int)0;
                            *(reinterpret_cast<unsigned int*>(hctr + (2 * num_heads + head)) + (0)) = (unsigned int)0;
                        }
                    }
                    unsigned int hold;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(hold) : "l"(&hctr[head]), "r"(static_cast<uint32_t>(1)) : "memory");
                    if ((int)hold == per - 1) {
                        *(reinterpret_cast<unsigned int*>(hctr + head) + (0)) = (unsigned int)0;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 3, 512;" ::: "memory");
            if (tid == 0) {
                unsigned int eold;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(eold) : "l"(&hctr[4 * num_heads + 5]), "r"(static_cast<uint32_t>(1)) : "memory");
                if ((int)eold == num_bids - 1) {
                    *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads + 4)) + (0)) = (unsigned int)0;
                    *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads + 5)) + (0)) = (unsigned int)0;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (4 * num_heads + 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
        }
    }
    // ---- Role: wk1 ----
    if (warp >= 8 && warp <= 15) {
        { // wk1_main
            int wtid_1 = tid - 256;
            int col_1 = (wtid_1 & 15) * 8;
            int rgrp_1 = wtid_1 >> 4;
            int sub_1 = wtid_1 & 7;
            int trow_1 = wtid_1 >> 3;
            int col16_1 = sub_1 * 16;
            int _count_leading_zeros_0_1 = __clz((unsigned int)(num_segments - 1));
            int nsteps_1 = 32 - _count_leading_zeros_0_1;
            int per_1 = num_tiles + num_kblocks;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int nxt_r_1[2];
            if (wtid_1 == 0) {
                unsigned int _atomic_fetch_add_1_1 = atomicAdd(&hctr[4 * num_heads + 4], 1);
                item_cell[4] = (int)_atomic_fetch_add_1_1;
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            #pragma unroll 1
            for (int it_1 = 0; it_1 < per_1 * num_heads; it_1++) {
                int w_1 = item_cell[4 + (it_1 & 1)];
                if (w_1 >= per_1 * num_heads) {
                    break;
                }
                if (wtid_1 == 0) {
                    unsigned int _atomic_fetch_add_2_1 = atomicAdd(&hctr[4 * num_heads + 4], 1);
                    nxt_r_1[0] = _atomic_fetch_add_2_1;
                }
                int w2_1 = ((w_1 >= num_tiles) ? w_1 - num_tiles : 0);
                int s_1 = w2_1 / per_1;
                int r_1 = w2_1 - s_1 * per_1;
                int last_s_1 = ((s_1 + 1 >= num_heads) ? 1 : 0);
                int is_q_1 = ((w_1 >= num_tiles && (last_s_1 == 1 || r_1 >= num_tiles)) ? 1 : 0);
                int head_1 = ((w_1 < num_tiles) ? 0 : ((is_q_1 == 1) ? s_1 : s_1 + 1));
                int ri_1 = ((w_1 < num_tiles) ? w_1 : ((last_s_1 == 1) ? r_1 : ((is_q_1 == 1) ? r_1 - num_tiles : r_1)));
                if (is_q_1 == 0) {
                    int s_work_1 = ri_1 * num_heads + head_1;
                    int s_seg_1 = tile_table[2 * ri_1];
                    int s_tile_in_seg_1 = tile_table[2 * ri_1 + 1];
                    int s_sb_1 = seg_begin[s_seg_1];
                    int s_length_1 = seg_len[s_seg_1];
                    int s_t0_1 = seg_tile_begin[s_seg_1];
                    int s_t1_1 = seg_tile_begin[s_seg_1 + 1];
                    int s_row_begin_1 = s_sb_1 + s_tile_in_seg_1 * 128;
                    int s_seg_end_1 = s_sb_1 + s_length_1;
                    int s_row_end_1 = ((s_seg_end_1 > s_row_begin_1 + 128) ? s_row_begin_1 + 128 : s_seg_end_1);
                    float ksum_1[8];
                    float vmax_1[8];
                    for (int e_9 = 0; e_9 < 8; e_9++) {
                        ksum_1[e_9] = 0.0f;
                        vmax_1[e_9] = 0.0f;
                    }
                    #pragma unroll 1
                    for (int row_1 = s_row_begin_1 + rgrp_1; row_1 < s_row_end_1; row_1 += 16) {
                        int s_base_1 = (row_1 * num_heads + head_1) * 128 + col_1;
                        float _vec_load_s0_1[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(K + s_base_1 + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_s0_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_s0_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_s1_1[8];
                        {
                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(V + s_base_1 + 0);
                            uint4 _vld_4[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_4[_blk] = _vptr_4[_blk];
                                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_s1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                    (&_vec_load_s1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        for (int e_10 = 0; e_10 < 8; e_10++) {
                            float s_kf_1 = _vec_load_s0_1[e_10];
                            float s_vf_1 = _vec_load_s1_1[e_10];
                            ksum_1[e_10] = ksum_1[e_10] + s_kf_1;
                            vmax_1[e_10] = fmaxf(vmax_1[e_10], fabsf(s_vf_1));
                        }
                    }
                    for (int e_11 = 0; e_11 < 8; e_11++) {
                        red_sum[2048 + rgrp_1 * 128 + col_1 + e_11] = ksum_1[e_11];
                        red_max[2048 + rgrp_1 * 128 + col_1 + e_11] = vmax_1[e_11];
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    int single_1 = ((s_t1_1 - s_t0_1 == 1) ? 1 : 0);
                    if (wtid_1 < 128) {
                        float total_1 = 0.0f;
                        float vmx_1 = 0.0f;
                        for (int g_1 = 0; g_1 < 16; g_1++) {
                            total_1 = total_1 + red_sum[2048 + g_1 * 128 + wtid_1];
                            vmx_1 = fmaxf(vmx_1, red_max[2048 + g_1 * 128 + wtid_1]);
                        }
                        if (single_1 == 1) {
                            float msum_1 = 0.0f;
                            float mmax_1 = 0.0f;
                            msum_1 = msum_1 + total_1;
                            mmax_1 = fmaxf(mmax_1, vmx_1);
                            float _fdiv_5_1 = __fdiv_rn(msum_1, (float)s_length_1);
                            *(reinterpret_cast<float*>(mean_k + ((s_seg_1 * num_heads + head_1) * 128 + wtid_1)) + (0)) = _fdiv_5_1;
                            float _fdiv_6_1 = __fdiv_rn(fmaxf(mmax_1, 1e-12f), 448.0f);
                            *(reinterpret_cast<float*>(v_scale + ((s_seg_1 * num_heads + head_1) * 128 + wtid_1)) + (0)) = _fdiv_6_1;
                        } else {
                            *(reinterpret_cast<float*>(partials + (s_work_1 * 256 + wtid_1)) + (0)) = total_1;
                            *(reinterpret_cast<float*>(partials + (s_work_1 * 256 + 128 + wtid_1)) + (0)) = vmx_1;
                        }
                    }
                    if (single_1 == 0) {
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        if (wtid_1 == 0) {
                            unsigned int old_1;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(old_1) : "l"(&counters[s_seg_1 * num_heads + head_1]), "r"(static_cast<uint32_t>(1)) : "memory");
                            int last_1 = (((int)old_1 == s_t1_1 - s_t0_1 - 1) ? 1 : 0);
                            flag[4] = last_1;
                            if (last_1 == 1) {
                                *(reinterpret_cast<unsigned int*>(counters + (s_seg_1 * num_heads + head_1)) + (0)) = (unsigned int)0;
                            }
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        if (flag[4] == 1) {
                            int chan_1 = wtid_1 & 127;
                            int is_max_1 = wtid_1 >> 7;
                            float acc_1[1];
                            float fold_v_1[8];
                            acc_1[0] = 0.0f;
                            #pragma unroll 1
                            for (int tt_1 = s_t0_1; tt_1 < s_t1_1; tt_1 += 8) {
                                for (int j_2 = 0; j_2 < 8; j_2++) {
                                    int ttj_1 = ((s_t1_1 > tt_1 + j_2) ? tt_1 + j_2 : s_t1_1 - 1);
                                    fold_v_1[j_2] = partials[(ttj_1 * num_heads + head_1) * 256 + is_max_1 * 128 + chan_1];
                                }
                                for (int j_3 = 0; j_3 < 8; j_3++) {
                                    if (s_t1_1 > tt_1 + j_3) {
                                        acc_1[0] = ((is_max_1 == 0) ? acc_1[0] + fold_v_1[j_3] : fmaxf(acc_1[0], fold_v_1[j_3]));
                                    }
                                }
                            }
                            if (is_max_1 == 0) {
                                float _fdiv_7_1 = __fdiv_rn(acc_1[0], (float)s_length_1);
                                *(reinterpret_cast<float*>(mean_k + ((s_seg_1 * num_heads + head_1) * 128 + chan_1)) + (0)) = _fdiv_7_1;
                            } else {
                                float _fdiv_8_1 = __fdiv_rn(fmaxf(acc_1[0], 1e-12f), 448.0f);
                                *(reinterpret_cast<float*>(v_scale + ((s_seg_1 * num_heads + head_1) * 128 + chan_1)) + (0)) = _fdiv_8_1;
                            }
                        }
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    if (wtid_1 == 0) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                } else {
                    if (wtid_1 == 0) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head_1))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(num_tiles)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    int tok_base_1 = ri_1 * 128;
                    float kvals_1[64];
                    float kmax_1 = 0.0f;
                    for (int ps_2 = 0; ps_2 < 4; ps_2++) {
                        int tok_1 = tok_base_1 + ps_2 * 32 + trow_1;
                        int valid_1 = ((tok_1 < total_tokens) ? 1 : 0);
                        int tok_c_1 = ((tok_1 < total_tokens) ? tok_1 : total_tokens - 1);
                        int base_1 = (tok_c_1 * num_heads + head_1) * 128 + col16_1;
                        float _vec_load_0_1[16];
                        {
                            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(Q + base_1 + 0);
                            uint4 _vld_9[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vld_9[_blk].x), "=r"(_vld_9[_blk].y), "=r"(_vld_9[_blk].z), "=r"(_vld_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                    (&_vec_load_0_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_1_1[16];
                        {
                            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(K + base_1 + 0);
                            uint4 _vld_10[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vld_10[_blk].x), "=r"(_vld_10[_blk].y), "=r"(_vld_10[_blk].z), "=r"(_vld_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                    (&_vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        int lo_1 = 0;
                        int hi_1 = num_segments - 1;
                        #pragma unroll 1
                        for (int step_2 = 0; step_2 < nsteps_1; step_2++) {
                            int mid_1 = lo_1 + hi_1 + 1 >> 1;
                            if (tok_c_1 >= cu_seqlens[mid_1]) {
                                lo_1 = mid_1;
                            } else {
                                hi_1 = mid_1 - 1;
                            }
                        }
                        int seg_1 = lo_1;
                        float qamax_1 = 0.0f;
                        for (int e_12 = 0; e_12 < 16; e_12++) {
                            float qf_1 = _vec_load_0_1[e_12];
                            qamax_1 = fmaxf(qamax_1, fabsf(qf_1));
                        }
                        float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, qamax_1, 4);
                        qamax_1 = fmaxf(qamax_1, _shfl_xor_sync_11_1);
                        float _shfl_xor_sync_12_1 = __shfl_xor_sync(0xFFFFFFFF, qamax_1, 2);
                        qamax_1 = fmaxf(qamax_1, _shfl_xor_sync_12_1);
                        float _shfl_xor_sync_13_1 = __shfl_xor_sync(0xFFFFFFFF, qamax_1, 1);
                        qamax_1 = fmaxf(qamax_1, _shfl_xor_sync_13_1);
                        qamax_1 = fmaxf(qamax_1, 1e-12f);
                        float qs_1 = __fdiv_rn(qamax_1, 448.0f);
                        float qinv_1 = __fdiv_rn(448.0f, qamax_1);
                        float qn_1[16];
                        for (int e_13 = 0; e_13 < 16; e_13++) {
                            float qf2_1 = _vec_load_0_1[e_13];
                            qn_1[e_13] = qf2_1 * qinv_1;
                        }
                        unsigned int qw_1[4];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(qn_1[0]), "f"(qn_1[1]),
                                                   "f"(qn_1[2]), "f"(qn_1[3]));
                            qw_1[0] = _packed;
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
                                : "=r"(_packed) : "f"(qn_1[4]), "f"(qn_1[5]),
                                                   "f"(qn_1[6]), "f"(qn_1[7]));
                            qw_1[1] = _packed;
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
                                : "=r"(_packed) : "f"(qn_1[8]), "f"(qn_1[9]),
                                                   "f"(qn_1[10]), "f"(qn_1[11]));
                            qw_1[2] = _packed;
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
                                : "=r"(_packed) : "f"(qn_1[12]), "f"(qn_1[13]),
                                                   "f"(qn_1[14]), "f"(qn_1[15]));
                            qw_1[3] = _packed;
                        }
                        if (valid_1 == 1) {
                            reinterpret_cast<int4*>(q8 + (base_1 >> 2))[0] = reinterpret_cast<int4*>(qw_1)[0];
                            if (sub_1 == 0) {
                                *(reinterpret_cast<float*>(q_scale + (tok_1 * num_heads + head_1)) + (0)) = qs_1;
                            }
                        }
                        int mbase_1 = (seg_1 * num_heads + head_1) * 128 + col16_1;
                        for (int quad_2 = 0; quad_2 < 4; quad_2++) {
                            float _vec_load_2_1[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase_1 + quad_2 * 4) + 0);
                                _vec_load_2_1[0 + 0] = _v4.x;
                                _vec_load_2_1[0 + 1] = _v4.y;
                                _vec_load_2_1[0 + 2] = _v4.z;
                                _vec_load_2_1[0 + 3] = _v4.w;
                            }
                            for (int e_14 = 0; e_14 < 4; e_14++) {
                                float kf_1 = _vec_load_1_1[quad_2 * 4 + e_14];
                                float kc_1 = kf_1 - _vec_load_2_1[e_14];
                                kvals_1[ps_2 * 16 + quad_2 * 4 + e_14] = ((valid_1 == 1) ? kc_1 : 0.0f);
                                kmax_1 = fmaxf(kmax_1, fabsf(kvals_1[ps_2 * 16 + quad_2 * 4 + e_14]));
                            }
                        }
                    }
                    for (int stage_2 = 0; stage_2 < 5; stage_2++) {
                        float _shfl_xor_sync_15_1 = __shfl_xor_sync(0xFFFFFFFF, kmax_1, 16 >> stage_2);
                        kmax_1 = fmaxf(kmax_1, _shfl_xor_sync_15_1);
                    }
                    if (lane == 0) {
                        amax_part[8 + (warp & 7)] = kmax_1;
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    kmax_1 = amax_part[8 + (lane & 7)];
                    for (int stage_3 = 0; stage_3 < 3; stage_3++) {
                        float _shfl_xor_sync_16_1 = __shfl_xor_sync(0xFFFFFFFF, kmax_1, 4 >> stage_3);
                        kmax_1 = fmaxf(kmax_1, _shfl_xor_sync_16_1);
                    }
                    kmax_1 = fmaxf(kmax_1, 1e-12f);
                    float kinv_1 = __fdiv_rn(448.0f, kmax_1);
                    if (wtid_1 == 0) {
                        float _fdiv_17_1 = __fdiv_rn(kmax_1, 448.0f);
                        *(reinterpret_cast<float*>(k_scale + (head_1 * num_kblocks + ri_1)) + (0)) = _fdiv_17_1;
                    }
                    for (int ps_3 = 0; ps_3 < 4; ps_3++) {
                        int tok2_1 = tok_base_1 + ps_3 * 32 + trow_1;
                        float kn_1[16];
                        for (int e_15 = 0; e_15 < 16; e_15++) {
                            kn_1[e_15] = kvals_1[ps_3 * 16 + e_15] * kinv_1;
                        }
                        unsigned int kw_1[4];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(kn_1[0]), "f"(kn_1[1]),
                                                   "f"(kn_1[2]), "f"(kn_1[3]));
                            kw_1[0] = _packed;
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
                                : "=r"(_packed) : "f"(kn_1[4]), "f"(kn_1[5]),
                                                   "f"(kn_1[6]), "f"(kn_1[7]));
                            kw_1[1] = _packed;
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
                                : "=r"(_packed) : "f"(kn_1[8]), "f"(kn_1[9]),
                                                   "f"(kn_1[10]), "f"(kn_1[11]));
                            kw_1[2] = _packed;
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
                                : "=r"(_packed) : "f"(kn_1[12]), "f"(kn_1[13]),
                                                   "f"(kn_1[14]), "f"(kn_1[15]));
                            kw_1[3] = _packed;
                        }
                        if (tok2_1 < total_tokens) {
                            reinterpret_cast<int4*>(k8 + ((tok2_1 * num_heads + head_1) * 128 + col16_1 >> 2))[0] = reinterpret_cast<int4*>(kw_1)[0];
                        }
                    }
                    int task_1 = wtid_1;
                    int cg_1 = task_1 >> 5;
                    int chunk_1 = (task_1 & 31) >> 2;
                    int t8_1 = task_1 & 3;
                    int c0_1 = cg_1 * 16;
                    float vq_1[64];
                    for (int i_2 = 0; i_2 < 4; i_2++) {
                        int key_local_1 = chunk_1 * 16 + (i_2 >> 1) * 8 + 2 * t8_1 + (i_2 & 1);
                        int vtok_1 = tok_base_1 + key_local_1;
                        int vvalid_1 = ((vtok_1 < total_tokens) ? 1 : 0);
                        int vtok_c_1 = ((vtok_1 < total_tokens) ? vtok_1 : total_tokens - 1);
                        float _vec_load_3_1[16];
                        {
                            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(V + ((vtok_c_1 * num_heads + head_1) * 128 + c0_1) + 0);
                            uint4 _vld_18[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vld_18[_blk].x), "=r"(_vld_18[_blk].y), "=r"(_vld_18[_blk].z), "=r"(_vld_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                    (&_vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        int vlo_1 = 0;
                        int vhi_1 = num_segments - 1;
                        #pragma unroll 1
                        for (int step_3 = 0; step_3 < nsteps_1; step_3++) {
                            int vmid_1 = vlo_1 + vhi_1 + 1 >> 1;
                            if (vtok_c_1 >= cu_seqlens[vmid_1]) {
                                vlo_1 = vmid_1;
                            } else {
                                vhi_1 = vmid_1 - 1;
                            }
                        }
                        int vseg_1 = vlo_1;
                        int sbase_1 = (vseg_1 * num_heads + head_1) * 128 + c0_1;
                        for (int quad_3 = 0; quad_3 < 4; quad_3++) {
                            float _vec_load_4_1[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (sbase_1 + quad_3 * 4) + 0);
                                _vec_load_4_1[0 + 0] = _v4.x;
                                _vec_load_4_1[0 + 1] = _v4.y;
                                _vec_load_4_1[0 + 2] = _v4.z;
                                _vec_load_4_1[0 + 3] = _v4.w;
                            }
                            for (int e_16 = 0; e_16 < 4; e_16++) {
                                float vf_1 = _vec_load_3_1[quad_3 * 4 + e_16];
                                float _fdiv_20_1 = __fdiv_rn(vf_1, _vec_load_4_1[e_16]);
                                vq_1[i_2 * 16 + quad_3 * 4 + e_16] = ((vvalid_1 == 1) ? _fdiv_20_1 : 0.0f);
                            }
                        }
                    }
                    int vt_word_1 = (head_1 * 128 + c0_1) * total_padded + tok_base_1 + chunk_1 * 16 + 4 * t8_1 >> 2;
                    for (int e_17 = 0; e_17 < 16; e_17++) {
                        float four_1[4];
                        for (int i_3 = 0; i_3 < 4; i_3++) {
                            four_1[i_3] = vq_1[i_3 * 16 + e_17];
                        }
                        unsigned int word_1[1];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(four_1[0]), "f"(four_1[1]),
                                                   "f"(four_1[2]), "f"(four_1[3]));
                            word_1[0] = _packed;
                        }
                        *(reinterpret_cast<unsigned int*>(vt8 + (vt_word_1 + e_17 * (total_padded >> 2))) + (0)) = word_1[0];
                    }
                }
                if (wtid_1 == 0) {
                    item_cell[4 + (it_1 + 1 & 1)] = (int)nxt_r_1[0];
                }
                asm volatile("bar.sync 2, 256;" ::: "memory");
                if (wtid_1 == 0) {
                    if (is_q_1 == 1) {
                        unsigned int qold_1;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(qold_1) : "l"(&hctr[2 * num_heads + head_1]), "r"(static_cast<uint32_t>(1)) : "memory");
                        if ((int)qold_1 == num_kblocks - 1) {
                            *(reinterpret_cast<unsigned int*>(hctr + (num_heads + head_1)) + (0)) = (unsigned int)0;
                            *(reinterpret_cast<unsigned int*>(hctr + (2 * num_heads + head_1)) + (0)) = (unsigned int)0;
                        }
                    }
                    unsigned int hold_1;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(hold_1) : "l"(&hctr[head_1]), "r"(static_cast<uint32_t>(1)) : "memory");
                    if ((int)hold_1 == per_1 - 1) {
                        *(reinterpret_cast<unsigned int*>(hctr + head_1) + (0)) = (unsigned int)0;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 3, 512;" ::: "memory");
        }
    }
}

} // extern "C"
