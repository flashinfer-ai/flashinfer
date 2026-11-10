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
#define SMEM_AMAX_PART_OFF 0
#define SMEM_AMAX_PART_STAGE_BYTES 32
#define SMEM_AMAX_PART_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_quant_varlen_attention_e2abeffeb2ef21daba83(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ cu_seqlens, float* __restrict__ mean_k, float* __restrict__ v_scale, unsigned int* __restrict__ q8, unsigned int* __restrict__ k8, unsigned int* __restrict__ vt8, float* __restrict__ q_scale, float* __restrict__ k_scale, int total_tokens, int total_padded, int num_segments, int num_heads, int num_kblocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* amax_part = reinterpret_cast<float*>(smem_raw + 0);
    const int amax_part_addr = smem + 0;

    // === Task calls (declaration order) ===
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    int _count_leading_zeros_0 = __clz((unsigned int)(num_segments - 1));
    int nsteps = 32 - _count_leading_zeros_0;
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
            int base = (tok_c * num_heads + head) * 128 + col16;
            float _vec_load_0[16];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(Q + base + 0);
                uint4 _vld_1[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_1[16];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(K + base + 0);
                uint4 _vld_2[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
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
            for (int e = 0; e < 16; e++) {
                float qf = _vec_load_0[e];
                qamax = fmaxf(qamax, fabsf(qf));
            }
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, qamax, 4);
            qamax = fmaxf(qamax, _shfl_xor_sync_3);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, qamax, 2);
            qamax = fmaxf(qamax, _shfl_xor_sync_4);
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, qamax, 1);
            qamax = fmaxf(qamax, _shfl_xor_sync_5);
            qamax = fmaxf(qamax, 1e-12f);
            float qs = __fdiv_rn(qamax, 448.0f);
            float qinv = __fdiv_rn(448.0f, qamax);
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
                    kmax = fmaxf(kmax, fabsf(kvals[ps * 16 + quad * 4 + e_2]));
                }
            }
        }
        for (int stage = 0; stage < 5; stage++) {
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, kmax, 16 >> stage);
            kmax = fmaxf(kmax, _shfl_xor_sync_7);
        }
        if (lane == 0) {
            amax_part[warp] = kmax;
        }
        __syncthreads();
        kmax = amax_part[lane & 7];
        for (int stage_1 = 0; stage_1 < 3; stage_1++) {
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, kmax, 4 >> stage_1);
            kmax = fmaxf(kmax, _shfl_xor_sync_8);
        }
        kmax = fmaxf(kmax, 1e-12f);
        float kinv = __fdiv_rn(448.0f, kmax);
        if (tid == 0) {
            float _fdiv_9 = __fdiv_rn(kmax, 448.0f);
            *(reinterpret_cast<float*>(k_scale + (head * num_kblocks + blk)) + (0)) = _fdiv_9;
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
        int task = tid;
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
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(V + ((vtok_c * num_heads + head) * 128 + c0) + 0);
                uint4 _vld_10[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_10[_blk] = _vptr_10[_blk];
                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
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
                for (int e_4 = 0; e_4 < 4; e_4++) {
                    float vf = _vec_load_3[quad_1 * 4 + e_4];
                    float _fdiv_12 = __fdiv_rn(vf, _vec_load_4[e_4]);
                    vq[i * 16 + quad_1 * 4 + e_4] = ((vvalid == 1) ? _fdiv_12 : 0.0f);
                }
            }
        }
        int vt_word = (head * 128 + c0) * total_padded + tok_base + chunk * 16 + 4 * t8 >> 2;
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
        __syncthreads();
    }
}

} // extern "C"
