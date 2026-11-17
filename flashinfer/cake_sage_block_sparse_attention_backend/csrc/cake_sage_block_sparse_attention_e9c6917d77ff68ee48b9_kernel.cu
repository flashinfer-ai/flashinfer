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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(8,1,1) void
kernel_cake_sage_block_sparse_attention_e9c6917d77ff68ee48b9(__nv_bfloat16* q_in, __nv_bfloat16* k_in, __nv_bfloat16* v_in, uint8_t* q_out, uint8_t* k_out, uint8_t* v_out, float* q_scale, float* k_scale, float* v_scale, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int grid_heads, int k_scale_groups, int passes)
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
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* vmax_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int vmax_smem_addr = smem + 0;
    float* part = reinterpret_cast<float*>(smem_raw + 8192);
    const int part_addr = smem + 8192;
    float* fin = reinterpret_cast<float*>(smem_raw + 8704);
    const int fin_addr = smem + 8704;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int rank = cta_rank;
    int cid = bid / 8;
    int head = cid % grid_heads;
    int batch = cid / grid_heads;
    int bh = batch * num_heads + head;
    int kv_bh = batch * num_kv_heads + head;
    int warp_0 = warp;
    int lane_1 = lane;
    int half = lane_1 / 8;
    int sub = lane_1 % 8;
    int channel0 = sub * 16;
    int q_head = ((head < num_heads) ? 1 : 0);
    int kv_head = ((head < num_kv_heads) ? 1 : 0);
    unsigned int vmax[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        vmax[j] = 0;
    }
    for (int ps = 0; ps < passes; ps++) {
        int tok_base = (ps * 8 + rank) * 256 + warp_0 * 16;
        int token0 = tok_base + half;
        int q_active = ((q_head != 0 && tok_base < seqlen_q) ? 1 : 0);
        int kv_active = ((kv_head != 0 && tok_base < seqlen_k) ? 1 : 0);
        unsigned int qw[32];
        unsigned int kw[32];
        unsigned int vw[32];
        int q_off[4];
        int q_ok[4];
        int k_off[4];
        int k_ok[4];
        if (q_active != 0) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int token = token0 + p * 4;
                int in_range = ((token < seqlen_q) ? 1 : 0);
                int load_token = ((in_range != 0) ? token : seqlen_q - 1);
                int offset = ((batch * seqlen_q + load_token) * num_heads + head) * 128 + channel0;
                q_ok[p] = in_range;
                q_off[p] = offset;
                {
                    asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                        : "=r"(qw[p * 8 + 0]), "=r"(qw[p * 8 + 1]), "=r"(qw[p * 8 + 2]), "=r"(qw[p * 8 + 3]), "=r"(qw[p * 8 + 4]), "=r"(qw[p * 8 + 5]), "=r"(qw[p * 8 + 6]), "=r"(qw[p * 8 + 7]) : "l"((const void*)((const char*)(q_in + q_off[p] + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        if (kv_active != 0) {
            #pragma unroll
            for (int p_1 = 0; p_1 < 4; p_1++) {
                int token_1 = token0 + p_1 * 4;
                int in_range_1 = ((token_1 < seqlen_k) ? 1 : 0);
                int load_token_1 = ((in_range_1 != 0) ? token_1 : seqlen_k - 1);
                int offset_1 = ((batch * seqlen_k + load_token_1) * num_kv_heads + head) * 128 + channel0;
                k_ok[p_1] = in_range_1;
                k_off[p_1] = offset_1;
                {
                    asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                        : "=r"(kw[p_1 * 8 + 0]), "=r"(kw[p_1 * 8 + 1]), "=r"(kw[p_1 * 8 + 2]), "=r"(kw[p_1 * 8 + 3]), "=r"(kw[p_1 * 8 + 4]), "=r"(kw[p_1 * 8 + 5]), "=r"(kw[p_1 * 8 + 6]), "=r"(kw[p_1 * 8 + 7]) : "l"((const void*)((const char*)(k_in + k_off[p_1] + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                }
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(vw[p_1 * 8 + 0]), "=r"(vw[p_1 * 8 + 1]), "=r"(vw[p_1 * 8 + 2]), "=r"(vw[p_1 * 8 + 3]), "=r"(vw[p_1 * 8 + 4]), "=r"(vw[p_1 * 8 + 5]), "=r"(vw[p_1 * 8 + 6]), "=r"(vw[p_1 * 8 + 7]) : "l"((const void*)((const char*)(v_in + k_off[p_1] + 0) + 0)) : "memory");
                }
            }
        }
        if (q_active != 0) {
            #pragma unroll
            for (int p_2 = 0; p_2 < 4; p_2++) {
                int token_2 = token0 + p_2 * 4;
                uint32_t _bf16x2_abs_0;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(qw[p_2 * 8]));
                unsigned int packed = _bf16x2_abs_0;
                #pragma unroll
                for (int j_1 = 1; j_1 < 8; j_1++) {
                    uint32_t _bf16x2_abs_1;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(qw[p_2 * 8 + j_1]));
                    uint32_t _bf16x2_max_0;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(packed), "r"(_bf16x2_abs_1));
                    packed = _bf16x2_max_0;
                }
                float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(packed));
                float amax = fmaxf(_cvt_f32_0.x, _cvt_f32_0.y);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, amax, 4);
                amax = fmaxf(amax, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
                amax = fmaxf(amax, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
                amax = fmaxf(amax, _shfl_xor_sync_5);
                amax = fmaxf(amax, 1e-06f);
                float values[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(qw[p_2 * 8 + j_2]));
                    values[2 * j_2] = _cvt_f32_1.x;
                    values[2 * j_2 + 1] = _cvt_f32_1.y;
                }
                const float2 _scale2_6 = {448.0f / amax, 448.0f / amax};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_6);
                if (q_ok[p_2] != 0) {
                    unsigned int _packed_0[4];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[0]), "f"(values[1]),
                                               "f"(values[2]), "f"(values[3]));
                        _packed_0[0] = _packed;
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
                            : "=r"(_packed) : "f"(values[4]), "f"(values[5]),
                                               "f"(values[6]), "f"(values[7]));
                        _packed_0[1] = _packed;
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
                            : "=r"(_packed) : "f"(values[8]), "f"(values[9]),
                                               "f"(values[10]), "f"(values[11]));
                        _packed_0[2] = _packed;
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
                            : "=r"(_packed) : "f"(values[12]), "f"(values[13]),
                                               "f"(values[14]), "f"(values[15]));
                        _packed_0[3] = _packed;
                    }
                    reinterpret_cast<int4*>(q_out + q_off[p_2])[0] = reinterpret_cast<int4*>(_packed_0)[0];
                }
                if (q_ok[p_2] != 0 && sub == 0) {
                    q_scale[bh * seqlen_q + token_2] = amax / 448.0f;
                }
            }
        }
        if (kv_active != 0) {
            int group_index = tok_base / 16;
            unsigned int kmax = 0;
            #pragma unroll
            for (int p_3 = 0; p_3 < 4; p_3++) {
                uint32_t _bf16x2_abs_2;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(kw[p_3 * 8]));
                unsigned int kabs = _bf16x2_abs_2;
                #pragma unroll
                for (int j_3 = 1; j_3 < 8; j_3++) {
                    uint32_t _bf16x2_abs_3;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(kw[p_3 * 8 + j_3]));
                    uint32_t _bf16x2_max_1;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(kabs), "r"(_bf16x2_abs_3));
                    kabs = _bf16x2_max_1;
                }
                unsigned int masked = ((k_ok[p_3] != 0) ? kabs : 0);
                uint32_t _bf16x2_max_2;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(kmax), "r"(masked));
                kmax = _bf16x2_max_2;
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    uint32_t _bf16x2_abs_4;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(vw[p_3 * 8 + j_4]));
                    unsigned int vmasked = ((k_ok[p_3] != 0) ? _bf16x2_abs_4 : 0);
                    uint32_t _bf16x2_max_3;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(vmax[j_4]), "r"(vmasked));
                    vmax[j_4] = _bf16x2_max_3;
                }
            }
            float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(kmax));
            float group_max = fmaxf(_cvt_f32_2.x, _cvt_f32_2.y);
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, group_max, 16);
            group_max = fmaxf(group_max, _shfl_xor_sync_7);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, group_max, 8);
            group_max = fmaxf(group_max, _shfl_xor_sync_8);
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, group_max, 4);
            group_max = fmaxf(group_max, _shfl_xor_sync_9);
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
            group_max = fmaxf(group_max, _shfl_xor_sync_10);
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
            group_max = fmaxf(group_max, _shfl_xor_sync_11);
            group_max = fmaxf(group_max, 1e-06f);
            #pragma unroll
            for (int p_4 = 0; p_4 < 4; p_4++) {
                float values_1[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(kw[p_4 * 8 + j_5]));
                    values_1[2 * j_5] = _cvt_f32_3.x;
                    values_1[2 * j_5 + 1] = _cvt_f32_3.y;
                }
                const float2 _scale2_12 = {448.0f / group_max, 448.0f / group_max};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values_1)[_ls], _scale2_12);
                if (k_ok[p_4] != 0) {
                    unsigned int _packed_1[4];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values_1[0]), "f"(values_1[1]),
                                               "f"(values_1[2]), "f"(values_1[3]));
                        _packed_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(values_1[4]), "f"(values_1[5]),
                                               "f"(values_1[6]), "f"(values_1[7]));
                        _packed_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(values_1[8]), "f"(values_1[9]),
                                               "f"(values_1[10]), "f"(values_1[11]));
                        _packed_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(values_1[12]), "f"(values_1[13]),
                                               "f"(values_1[14]), "f"(values_1[15]));
                        _packed_1[3] = _packed;
                    }
                    reinterpret_cast<int4*>(k_out + k_off[p_4])[0] = reinterpret_cast<int4*>(_packed_1)[0];
                }
            }
            if (lane_1 == 0 && group_index < k_scale_groups) {
                k_scale[kv_bh * k_scale_groups + group_index] = group_max / 448.0f;
            }
        }
    }
    #pragma unroll
    for (int j_6 = 0; j_6 < 8; j_6++) {
        unsigned int _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_6], 8);
        uint32_t _bf16x2_max_4;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(vmax[j_6]), "r"(_shfl_xor_sync_13));
        vmax[j_6] = _bf16x2_max_4;
    }
    #pragma unroll
    for (int j_7 = 0; j_7 < 8; j_7++) {
        unsigned int _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_7], 16);
        uint32_t _bf16x2_max_5;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(vmax[j_7]), "r"(_shfl_xor_sync_14));
        vmax[j_7] = _bf16x2_max_5;
    }
    if (half == 0) {
        #pragma unroll
        for (int j_8 = 0; j_8 < 8; j_8++) {
            float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vmax[j_8]));
            vmax_smem[warp_0 * 128 + channel0 + 2 * j_8] = _cvt_f32_4.x;
            vmax_smem[warp_0 * 128 + channel0 + 2 * j_8 + 1] = _cvt_f32_4.y;
        }
    }
    asm volatile("barrier.sync 2, 512;" ::: "memory");
    if (tid < 128) {
        float cta_max = vmax_smem[tid];
        #pragma unroll
        for (int other = 1; other < 16; other++) {
            cta_max = fmaxf(cta_max, vmax_smem[other * 128 + tid]);
        }
        part[tid] = cta_max;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid < 128) {
        float peer[8];
        uint32_t _cluster_mapa_15;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_15) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(0));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[0])) : "r"(_cluster_mapa_15));
        uint32_t _cluster_mapa_16;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_16) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(1));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[1])) : "r"(_cluster_mapa_16));
        uint32_t _cluster_mapa_17;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_17) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(2));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[2])) : "r"(_cluster_mapa_17));
        uint32_t _cluster_mapa_18;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_18) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(3));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[3])) : "r"(_cluster_mapa_18));
        uint32_t _cluster_mapa_19;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_19) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(4));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[4])) : "r"(_cluster_mapa_19));
        uint32_t _cluster_mapa_20;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_20) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(5));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[5])) : "r"(_cluster_mapa_20));
        uint32_t _cluster_mapa_21;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_21) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(6));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[6])) : "r"(_cluster_mapa_21));
        uint32_t _cluster_mapa_22;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_22) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(7));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[7])) : "r"(_cluster_mapa_22));
        float cl_max = peer[0];
        cl_max = fmaxf(cl_max, peer[1]);
        cl_max = fmaxf(cl_max, peer[2]);
        cl_max = fmaxf(cl_max, peer[3]);
        cl_max = fmaxf(cl_max, peer[4]);
        cl_max = fmaxf(cl_max, peer[5]);
        cl_max = fmaxf(cl_max, peer[6]);
        cl_max = fmaxf(cl_max, peer[7]);
        fin[tid] = cl_max;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (kv_head != 0) {
        float amax_v[16];
        #pragma unroll
        for (int j_9 = 0; j_9 < 16; j_9++) {
            amax_v[j_9] = fin[channel0 + j_9];
        }
        float inv[16];
        #pragma unroll
        for (int j_10 = 0; j_10 < 16; j_10++) {
            inv[j_10] = 448.0f / fmaxf(amax_v[j_10], 1e-06f);
        }
        if (rank == 0 && tid < 8) {
            #pragma unroll
            for (int j_11 = 0; j_11 < 16; j_11++) {
                v_scale[kv_bh * 128 + channel0 + j_11] = fmaxf(amax_v[j_11], 1e-06f) / 448.0f;
            }
        }
        for (int ps_1 = 0; ps_1 < passes; ps_1++) {
            int tok_base_1 = (ps_1 * 8 + rank) * 256 + warp_0 * 16;
            int token0_1 = tok_base_1 + half;
            if (tok_base_1 < seqlen_k) {
                unsigned int vw_1[32];
                int v_off[4];
                int v_ok[4];
                #pragma unroll
                for (int p_5 = 0; p_5 < 4; p_5++) {
                    int token_3 = token0_1 + p_5 * 4;
                    int in_range_2 = ((token_3 < seqlen_k) ? 1 : 0);
                    int load_token_2 = ((in_range_2 != 0) ? token_3 : seqlen_k - 1);
                    int offset_2 = ((batch * seqlen_k + load_token_2) * num_kv_heads + head) * 128 + channel0;
                    v_ok[p_5] = in_range_2;
                    v_off[p_5] = offset_2;
                    {
                        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                            : "=r"(vw_1[p_5 * 8 + 0]), "=r"(vw_1[p_5 * 8 + 1]), "=r"(vw_1[p_5 * 8 + 2]), "=r"(vw_1[p_5 * 8 + 3]), "=r"(vw_1[p_5 * 8 + 4]), "=r"(vw_1[p_5 * 8 + 5]), "=r"(vw_1[p_5 * 8 + 6]), "=r"(vw_1[p_5 * 8 + 7]) : "l"((const void*)((const char*)(v_in + offset_2 + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                #pragma unroll
                for (int p_6 = 0; p_6 < 4; p_6++) {
                    float values_2[16];
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 8; j_12++) {
                        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vw_1[p_6 * 8 + j_12]));
                        values_2[2 * j_12] = _cvt_f32_5.x;
                        values_2[2 * j_12 + 1] = _cvt_f32_5.y;
                    }
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 16; j_13++) {
                        values_2[j_13] = values_2[j_13] * inv[j_13];
                    }
                    if (v_ok[p_6] != 0) {
                        unsigned int _packed_2[4];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values_2[0]), "f"(values_2[1]),
                                                   "f"(values_2[2]), "f"(values_2[3]));
                            _packed_2[0] = _packed;
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
                                : "=r"(_packed) : "f"(values_2[4]), "f"(values_2[5]),
                                                   "f"(values_2[6]), "f"(values_2[7]));
                            _packed_2[1] = _packed;
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
                                : "=r"(_packed) : "f"(values_2[8]), "f"(values_2[9]),
                                                   "f"(values_2[10]), "f"(values_2[11]));
                            _packed_2[2] = _packed;
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
                                : "=r"(_packed) : "f"(values_2[12]), "f"(values_2[13]),
                                                   "f"(values_2[14]), "f"(values_2[15]));
                            _packed_2[3] = _packed;
                        }
                        reinterpret_cast<int4*>(v_out + v_off[p_6])[0] = reinterpret_cast<int4*>(_packed_2)[0];
                    }
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
