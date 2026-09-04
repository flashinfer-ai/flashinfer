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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "FlashInfer requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) FlashInferTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) FlashInferTensorMapPack { FlashInferTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMIXED_OFF 0
#define SMEM_SMIXED_STAGE_BYTES 2048
#define SMEM_SMIXED_STRIDE 2048
#define SMEM_SRECURRENCE_OFF 2048
#define SMEM_SRECURRENCE_STAGE_BYTES 256
#define SMEM_SRECURRENCE_STRIDE 256
#define SMEM_SOUTPUTSCALE_OFF 2304
#define SMEM_SOUTPUTSCALE_STAGE_BYTES 512
#define SMEM_SOUTPUTSCALE_STRIDE 512
#define SMEM_SGATEDECAY_OFF 2816
#define SMEM_SGATEDECAY_STAGE_BYTES 768
#define SMEM_SGATEDECAY_STRIDE 768
#define SMEM_SBETA_OFF 3584
#define SMEM_SBETA_STAGE_BYTES 12
#define SMEM_SBETA_STRIDE 12
#define SMEM_SFIRSTSLABRMSPARTIAL_OFF 3600
#define SMEM_SFIRSTSLABRMSPARTIAL_STAGE_BYTES 64
#define SMEM_SFIRSTSLABRMSPARTIAL_STRIDE 64
#define SMEM_SASYNCSTATE_OFF 3712
#define SMEM_SASYNCSTATE_STAGE_BYTES 24576
#define SMEM_SASYNCSTATE_STRIDE 24576
#define SMEM_TOTAL 28288
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, 3) void
kernel_kernel(__nv_bfloat16* __restrict__ x, float* __restrict__ weight, __nv_bfloat16* __restrict__ conv_state, __nv_bfloat16* __restrict__ raw_gate, __nv_bfloat16* __restrict__ raw_beta, float* __restrict__ A_log, float* __restrict__ dt_bias, int* __restrict__ state_indices, float* __restrict__ state, __nv_bfloat16* __restrict__ output_gate, float* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int x_row_stride, int conv_slot_stride, int beta_row_stride, int state_slot_stride, int output_gate_row_stride, int H, int use_lower_bound, float lower_bound_log2, float norm_eps)
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
    float* sMixed = reinterpret_cast<float*>(smem_raw + 0);
    const int sMixed_addr = smem + 0;
    __nv_bfloat16* sRecurrence = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int sRecurrence_addr = smem + 2048;
    float* sOutputScale = reinterpret_cast<float*>(smem_raw + 2304);
    const int sOutputScale_addr = smem + 2304;
    float* sGateDecay = reinterpret_cast<float*>(smem_raw + 2816);
    const int sGateDecay_addr = smem + 2816;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 3584);
    const int sBeta_addr = smem + 3584;
    float* sFirstSlabRmsPartial = reinterpret_cast<float*>(smem_raw + 3600);
    const int sFirstSlabRmsPartial_addr = smem + 3600;
    unsigned int* sAsyncState = reinterpret_cast<unsigned int*>(smem_raw + 3712);
    const int sAsyncState_addr = smem + 3712;

    // === Task calls (dependency order) ===
    int row = blockIdx.y;
    int head = blockIdx.x;
    if (row % 2 != 0) {
        head = 95 - head;
    }
    int tid_0 = tid;
    int lane_1 = lane;
    int group = tid_0 / 16;
    int lane_group = tid_0 % 16;
    int k_start = lane_group * 8;
    int qk_smem_start = lane_group * 12;
    int state_owner_row_base = group / 2 * 8 + group % 2;
    int hidden = 12288;
    int qkv_size = 3 * hidden;
    int requested_slot = state_indices[row];
    int slot = requested_slot;
    int is_live = 1;
    float state_regs[32];
    unsigned int state_carriers[4];
    float r_q[8];
    float r_k[8];
    float r_decay[8];
    if (tid_0 < 96) {
        int qkv_idx = tid_0 / 32;
        int channel_lane = tid_0 - qkv_idx * 32;
        int channel_start = channel_lane * 4;
        int channel_base = qkv_idx * hidden + head * 128 + channel_start;
        #pragma unroll
        for (int width_idx = 0; width_idx < 4; width_idx++) {
            int weight_base = (qkv_idx * 4 + width_idx) * hidden + head * 128 + channel_start;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(weight + weight_base);
                state_regs[width_idx * 4 + 0] = _v4.x;
                state_regs[width_idx * 4 + 1] = _v4.y;
                state_regs[width_idx * 4 + 2] = _v4.z;
                state_regs[width_idx * 4 + 3] = _v4.w;
            }
        }
        int conv_base = slot * conv_slot_stride + channel_base;
        #pragma unroll
        for (int history_idx = 0; history_idx < 3; history_idx++) {
            {
                uint2 _vld_1;
                _vld_1 = *reinterpret_cast<const uint2*>(conv_state + conv_base + history_idx * qkv_size);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&state_regs[16 + history_idx * 4 + _pair * 2])[0]), "=f"((&state_regs[16 + history_idx * 4 + _pair * 2])[1])
                        : "r"(_vpairs_1[_pair]));
                }
            }
        }
        int x_base = row * x_row_stride + channel_base;
        #pragma unroll
        for (int channel_idx = 0; channel_idx < 4; channel_idx++) {
            r_q[channel_idx] = (float)x[x_base + channel_idx];
        }
        float2 _f2_0 = make_float2(state_regs[16], state_regs[17]);
        float2 _f2_1 = make_float2(state_regs[0], state_regs[1]);
        float2 _mul_f32x2_0;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
        float2 acc_pair0 = _mul_f32x2_0;
        float2 _f2_2 = make_float2(state_regs[18], state_regs[19]);
        float2 _f2_3 = make_float2(state_regs[2], state_regs[3]);
        float2 _mul_f32x2_1;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_3));
        float2 acc_pair1 = _mul_f32x2_1;
        float2 _f2_4 = make_float2(state_regs[20], state_regs[21]);
        float2 _f2_5 = make_float2(state_regs[4], state_regs[5]);
        acc_pair0 = fma_f32x2_rn_ftz(_f2_4, _f2_5, acc_pair0);
        float2 _f2_6 = make_float2(state_regs[22], state_regs[23]);
        float2 _f2_7 = make_float2(state_regs[6], state_regs[7]);
        acc_pair1 = fma_f32x2_rn_ftz(_f2_6, _f2_7, acc_pair1);
        float2 _f2_8 = make_float2(state_regs[24], state_regs[25]);
        float2 _f2_9 = make_float2(state_regs[8], state_regs[9]);
        acc_pair0 = fma_f32x2_rn_ftz(_f2_8, _f2_9, acc_pair0);
        float2 _f2_10 = make_float2(state_regs[26], state_regs[27]);
        float2 _f2_11 = make_float2(state_regs[10], state_regs[11]);
        acc_pair1 = fma_f32x2_rn_ftz(_f2_10, _f2_11, acc_pair1);
        float2 _f2_12 = make_float2(r_q[0], r_q[1]);
        float2 _f2_13 = make_float2(state_regs[12], state_regs[13]);
        acc_pair0 = fma_f32x2_rn_ftz(_f2_12, _f2_13, acc_pair0);
        float2 _f2_14 = make_float2(r_q[2], r_q[3]);
        float2 _f2_15 = make_float2(state_regs[14], state_regs[15]);
        acc_pair1 = fma_f32x2_rn_ftz(_f2_14, _f2_15, acc_pair1);
        float acc0 = acc_pair0.x;
        float acc1 = acc_pair0.y;
        float acc2 = acc_pair1.x;
        float acc3 = acc_pair1.y;
        float _tanh_approx_0;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(acc0 * 0.5f));
        float tanh0 = _tanh_approx_0;
        float _tanh_approx_1;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(acc1 * 0.5f));
        float tanh1 = _tanh_approx_1;
        float _tanh_approx_2;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(acc2 * 0.5f));
        float tanh2 = _tanh_approx_2;
        float _tanh_approx_3;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(acc3 * 0.5f));
        float tanh3 = _tanh_approx_3;
        float silu0 = acc0 * (tanh0 * 0.5f + 0.5f);
        float silu1 = acc1 * (tanh1 * 0.5f + 0.5f);
        float silu2 = acc2 * (tanh2 * 0.5f + 0.5f);
        float silu3 = acc3 * (tanh3 * 0.5f + 0.5f);
        int qk_segment = (3 - qkv_idx) / 2;
        int smem_channel_start = channel_start + channel_start / 8 * 4 * qk_segment;
        if (qkv_idx == 2) {
            int value_owner_smem_base = 384 + channel_start / 8 * 8 + channel_start % 8 / 2;
            sMixed[value_owner_smem_base] = (float)(__nv_bfloat16)silu0;
            sMixed[value_owner_smem_base + 4] = (float)(__nv_bfloat16)silu1;
            sMixed[value_owner_smem_base + 1] = (float)(__nv_bfloat16)silu2;
            sMixed[value_owner_smem_base + 4 + 1] = (float)(__nv_bfloat16)silu3;
        } else {
            int smem_qkv_base = qkv_idx * 192 + smem_channel_start;
            sMixed[smem_qkv_base] = (float)(__nv_bfloat16)silu0;
            sMixed[smem_qkv_base + 1] = (float)(__nv_bfloat16)silu1;
            sMixed[smem_qkv_base + 2] = (float)(__nv_bfloat16)silu2;
            sMixed[smem_qkv_base + 3] = (float)(__nv_bfloat16)silu3;
        }
        if (is_live != 0) {
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(state_regs[20 + 0], state_regs[20 + 1]);
                _pk[1] = __floats2bfloat162_rn(state_regs[20 + 2], state_regs[20 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(conv_state))[conv_base]) = _pk2;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(state_regs[24 + 0], state_regs[24 + 1]);
                _pk[1] = __floats2bfloat162_rn(state_regs[24 + 2], state_regs[24 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(conv_state))[conv_base + qkv_size]) = _pk2;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(r_q[0 + 0], r_q[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(r_q[0 + 2], r_q[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(conv_state))[conv_base + 2 * qkv_size]) = _pk2;
            }
        }
        if (qkv_idx == 1) {
            silu0 = (float)(__nv_bfloat16)silu0;
            silu1 = (float)(__nv_bfloat16)silu1;
            silu2 = (float)(__nv_bfloat16)silu2;
            silu3 = (float)(__nv_bfloat16)silu3;
            float2 _f2_16 = make_float2(0.0f, 0.0f);
            float2 producer_norm_pair = _f2_16;
            float2 _f2_17 = make_float2(silu0, silu1);
            float2 producer_value_pair = _f2_17;
            producer_norm_pair = fma_f32x2_rn_ftz(producer_value_pair, producer_value_pair, producer_norm_pair);
            float2 _f2_18 = make_float2(silu2, silu3);
            producer_value_pair = _f2_18;
            producer_norm_pair = fma_f32x2_rn_ftz(producer_value_pair, producer_value_pair, producer_norm_pair);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, producer_norm_pair.x, 1);
            float producer_peer_even = _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, producer_norm_pair.y, 1);
            float producer_peer_odd = _shfl_xor_1;
            float producer_norm = producer_norm_pair.x + producer_peer_even + (producer_norm_pair.y + producer_peer_odd);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, producer_norm, 16);
            producer_norm += _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, producer_norm, 8);
            producer_norm += _shfl_xor_3;
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, producer_norm, 4);
            producer_norm += _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, producer_norm, 2);
            producer_norm += _shfl_xor_5;
            if (lane_1 == 0) {
                float _rsqrt_0 = rsqrtf(producer_norm + 1e-06f);
                sBeta[2] = _rsqrt_0;
            }
        }
    }
    if (tid_0 >= 128) {
        int k_idx = tid_0 % 128;
        int gate_idx = (row * 96 + head) * 128 + k_idx;
        float _expf_0 = __expf(A_log[head]);
        float A = _expf_0;
        float gate = (float)raw_gate[gate_idx] + dt_bias[head * 128 + k_idx];
        float decay_log2 = 0.0f;
        if (use_lower_bound != 0) {
            float _tanh_approx_4;
            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(A * gate * 0.5f));
            decay_log2 = lower_bound_log2 * (_tanh_approx_4 * 0.5f + 0.5f);
        } else {
            float softplus = gate;
            if (gate <= 20.0f) {
                float _expf_1 = __expf(gate);
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_1));
                softplus = _log2_0 * 0.6931471805599453f;
            }
            float log_decay = (-A) * softplus;
            decay_log2 = log_decay * 1.4426950408889634f;
        }
        int gate_smem_idx = k_idx + k_idx / 8 * 4;
        float _exp2_0 = approx_exp2(decay_log2);
        sGateDecay[gate_smem_idx] = _exp2_0;
        float output_gate_value = (float)output_gate[row * output_gate_row_stride + head * 128 + k_idx];
        float _tanh_approx_5;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(output_gate_value * 0.5f));
        sOutputScale[k_idx] = norm_weight[k_idx] * (_tanh_approx_5 * 0.5f + 0.5f);
        if (tid_0 == 128) {
            float beta_raw = (float)raw_beta[row * beta_row_stride + head];
            float _tanh_approx_6;
            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_6) : "f"(beta_raw * 0.5f));
            sBeta[0] = _tanh_approx_6 * 0.5f + 0.5f;
        }
    }
    int state_head_base = slot * state_slot_stride + head * 128 * 128;
    int state_group_base = state_head_base + state_owner_row_base * 128 + k_start;
    #pragma unroll
    for (int local_row = 0; local_row < 4; local_row++) {
        {
            unsigned _ldv8_2_0;
            unsigned _ldv8_2_1;
            unsigned _ldv8_2_2;
            unsigned _ldv8_2_3;
            unsigned _ldv8_2_4;
            unsigned _ldv8_2_5;
            unsigned _ldv8_2_6;
            unsigned _ldv8_2_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(state + (state_group_base + local_row * 2 * 128))) : "memory");
            state_regs[local_row * 8 + 0] = __uint_as_float(_ldv8_2_0);
            state_regs[local_row * 8 + 1] = __uint_as_float(_ldv8_2_1);
            state_regs[local_row * 8 + 2] = __uint_as_float(_ldv8_2_2);
            state_regs[local_row * 8 + 3] = __uint_as_float(_ldv8_2_3);
            state_regs[local_row * 8 + 4] = __uint_as_float(_ldv8_2_4);
            state_regs[local_row * 8 + 5] = __uint_as_float(_ldv8_2_5);
            state_regs[local_row * 8 + 6] = __uint_as_float(_ldv8_2_6);
            state_regs[local_row * 8 + 7] = __uint_as_float(_ldv8_2_7);
        }
    }
    int prejoin_async_state_group_row = group * 3;
    int prejoin_async_state_src_base = state_group_base + 8192;
    #pragma unroll
    for (int prejoin_local_row = 0; prejoin_local_row < 3; prejoin_local_row++) {
        int prejoin_source_row = prejoin_local_row;
        if (prejoin_local_row == 2) {
            prejoin_source_row = 3;
        }
        int prejoin_elem = (prejoin_async_state_group_row + prejoin_local_row) * 128 + k_start;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(sAsyncState_addr + (unsigned int)(prejoin_elem * 4)), "l"(state + (prejoin_async_state_src_base + prejoin_source_row * 2 * 128)));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(sAsyncState_addr + (unsigned int)(prejoin_elem * 4) + 16), "l"(state + (prejoin_async_state_src_base + prejoin_source_row * 2 * 128 + 4)));
    }
    asm volatile("cp.async.commit_group;");
    __syncthreads();
    float2 _f2_19 = make_float2(0.0f, 0.0f);
    float2 q_sq_pair = _f2_19;
    float2 _f2_20 = make_float2(0.0f, 0.0f);
    float2 k_sq_pair = _f2_20;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r_q[i] = 0.0f;
        r_k[i] = 0.0f;
    }
    #pragma unroll
    for (int i_pair = 0; i_pair < 4; i_pair++) {
        int i0 = i_pair * 2;
        int i1 = i0 + 1;
        if (lane_1 < 16) {
            float2 _f2_21 = make_float2(sMixed[qk_smem_start + i0], sMixed[qk_smem_start + i1]);
            float2 q_pair = _f2_21;
            float2 _f2_22 = make_float2(sMixed[192 + qk_smem_start + i0], sMixed[192 + qk_smem_start + i1]);
            float2 k_pair = _f2_22;
            r_q[i0] = q_pair.x;
            r_q[i1] = q_pair.y;
            r_k[i0] = k_pair.x;
            r_k[i1] = k_pair.y;
            q_sq_pair = fma_f32x2_rn_ftz(q_pair, q_pair, q_sq_pair);
        }
    }
    float q_sq = q_sq_pair.x + q_sq_pair.y;
    float k_sq = k_sq_pair.x + k_sq_pair.y;
    float q_scale = 0.0f;
    float k_scale = 0.0f;
    if (lane_1 < 16) {
        k_scale = sBeta[2];
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, k_scale, lane_group);
    k_scale = _shfl_0;
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, r_k[i_1], lane_group);
        r_k[i_1] = _shfl_1;
    }
    float beta = sBeta[0];
    #pragma unroll
    for (int i_pair_1 = 0; i_pair_1 < 4; i_pair_1++) {
        int i0_1 = i_pair_1 * 2;
        int i1_1 = i0_1 + 1;
        r_decay[i0_1] = sGateDecay[qk_smem_start + i0_1];
        r_decay[i1_1] = sGateDecay[qk_smem_start + i1_1];
    }
    #pragma unroll
    for (int value_tile = 0; value_tile < 2; value_tile++) {
        int tile_base = value_tile * 64;
        int state_tile_base = state_group_base + tile_base * 128;
        if (value_tile > 0) {
            #pragma unroll
            for (int local_row_1 = 0; local_row_1 < 3; local_row_1++) {
                int async_state_local_row = local_row_1;
                if (local_row_1 == 2) {
                    async_state_local_row = 3;
                }
                int async_state_elem = (group * 3 + local_row_1) * 128 + k_start;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&state_regs[async_state_local_row * 8])), "=r"(*reinterpret_cast<uint32_t*>(&state_regs[(async_state_local_row * 8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_regs[(async_state_local_row * 8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_regs[(async_state_local_row * 8) + 3]))
                    : "r"(sAsyncState_addr + (unsigned int)(async_state_elem * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&state_regs[async_state_local_row * 8 + 4])), "=r"(*reinterpret_cast<uint32_t*>(&state_regs[(async_state_local_row * 8 + 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_regs[(async_state_local_row * 8 + 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_regs[(async_state_local_row * 8 + 4) + 3]))
                    : "r"(sAsyncState_addr + (unsigned int)(async_state_elem * 4) + 16));
            }
            {
                unsigned _ldv8_3_0;
                unsigned _ldv8_3_1;
                unsigned _ldv8_3_2;
                unsigned _ldv8_3_3;
                unsigned _ldv8_3_4;
                unsigned _ldv8_3_5;
                unsigned _ldv8_3_6;
                unsigned _ldv8_3_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(state + (state_tile_base + 512))) : "memory");
                state_regs[16 + 0] = __uint_as_float(_ldv8_3_0);
                state_regs[16 + 1] = __uint_as_float(_ldv8_3_1);
                state_regs[16 + 2] = __uint_as_float(_ldv8_3_2);
                state_regs[16 + 3] = __uint_as_float(_ldv8_3_3);
                state_regs[16 + 4] = __uint_as_float(_ldv8_3_4);
                state_regs[16 + 5] = __uint_as_float(_ldv8_3_5);
                state_regs[16 + 6] = __uint_as_float(_ldv8_3_6);
                state_regs[16 + 7] = __uint_as_float(_ldv8_3_7);
            }
        }
        #pragma unroll
        for (int row_group = 0; row_group < 4 / ((1) ? 4 : 2); row_group++) {
            int local_row_a = row_group * ((1) ? 4 : 2);
            int local_row_b = local_row_a + 1;
            int local_row_c = local_row_a + 2;
            int local_row_d = local_row_a + 3;
            float2 _f2_23 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_a = _f2_23;
            float2 _f2_24 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_b = _f2_24;
            float2 _f2_25 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_c = _f2_25;
            float2 _f2_26 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_d = _f2_26;
            #pragma unroll
            for (int i_pair_2 = 0; i_pair_2 < 4; i_pair_2++) {
                int i0_2 = i_pair_2 * 2;
                int i1_2 = i0_2 + 1;
                int reg_offset_a = local_row_a * 8 + i0_2;
                int reg_offset_b = local_row_b * 8 + i0_2;
                int reg_offset_c = local_row_c * 8 + i0_2;
                int reg_offset_d = local_row_d * 8 + i0_2;
                float2 _f2_27 = make_float2(r_decay[i0_2], r_decay[i1_2]);
                float2 decay_pair = _f2_27;
                float2 _f2_28 = make_float2(r_k[i0_2], r_k[i1_2]);
                float2 key_pair = _f2_28;
                float2 _f2_29 = make_float2(state_regs[reg_offset_a], state_regs[reg_offset_a + 1]);
                float2 _mul_f32x2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_29), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_a = _mul_f32x2_2;
                float2 _f2_30 = make_float2(state_regs[reg_offset_c], state_regs[reg_offset_c + 1]);
                float2 _mul_f32x2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_30), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_c = _mul_f32x2_3;
                state_regs[reg_offset_a] = state_pair_a.x;
                state_regs[reg_offset_a + 1] = state_pair_a.y;
                state_regs[reg_offset_c] = state_pair_c.x;
                state_regs[reg_offset_c + 1] = state_pair_c.y;
                state_key_pair_a = fma_f32x2_rn_ftz(key_pair, state_pair_a, state_key_pair_a);
                state_key_pair_c = fma_f32x2_rn_ftz(key_pair, state_pair_c, state_key_pair_c);
                float2 _f2_31 = make_float2(state_regs[reg_offset_b], state_regs[reg_offset_b + 1]);
                float2 _mul_f32x2_4;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_31), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_b = _mul_f32x2_4;
                float2 _f2_32 = make_float2(state_regs[reg_offset_d], state_regs[reg_offset_d + 1]);
                float2 _mul_f32x2_5;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_d = _mul_f32x2_5;
                state_regs[reg_offset_b] = state_pair_b.x;
                state_regs[reg_offset_b + 1] = state_pair_b.y;
                state_regs[reg_offset_d] = state_pair_d.x;
                state_regs[reg_offset_d + 1] = state_pair_d.y;
                state_key_pair_b = fma_f32x2_rn_ftz(key_pair, state_pair_b, state_key_pair_b);
                state_key_pair_d = fma_f32x2_rn_ftz(key_pair, state_pair_d, state_key_pair_d);
            }
            float state_key_dot_a = state_key_pair_a.x + state_key_pair_a.y;
            float state_key_dot_b = state_key_pair_b.x + state_key_pair_b.y;
            float state_key_dot_c = state_key_pair_c.x + state_key_pair_c.y;
            float state_key_dot_d = state_key_pair_d.x + state_key_pair_d.y;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 8);
            state_key_dot_a += _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 8);
            state_key_dot_b += _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_c, 8);
            state_key_dot_c += _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_d, 8);
            state_key_dot_d += _shfl_xor_9;
            if (value_tile == 0) {
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 8);
                q_sq += _shfl_xor_10;
            }
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 4);
            state_key_dot_a += _shfl_xor_11;
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 4);
            state_key_dot_b += _shfl_xor_12;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_c, 4);
            state_key_dot_c += _shfl_xor_13;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_d, 4);
            state_key_dot_d += _shfl_xor_14;
            if (value_tile == 0) {
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 4);
                q_sq += _shfl_xor_15;
            }
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 2);
            state_key_dot_a += _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 2);
            state_key_dot_b += _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_c, 2);
            state_key_dot_c += _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_d, 2);
            state_key_dot_d += _shfl_xor_19;
            if (value_tile == 0) {
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 2);
                q_sq += _shfl_xor_20;
            }
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 1);
            state_key_dot_a += _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 1);
            state_key_dot_b += _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_c, 1);
            state_key_dot_c += _shfl_xor_23;
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_d, 1);
            state_key_dot_d += _shfl_xor_24;
            if (value_tile == 0) {
                float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 1);
                q_sq += _shfl_xor_25;
                if (lane_1 < 16) {
                    float _rsqrt_1 = rsqrtf(q_sq + 1e-06f);
                    q_scale = _rsqrt_1 * 0.08838834764831845f;
                }
            }
            state_key_dot_a *= k_scale;
            state_key_dot_b *= k_scale;
            state_key_dot_c *= k_scale;
            state_key_dot_d *= k_scale;
            int value_row_a = tile_base + state_owner_row_base + local_row_a * 2;
            int value_row_b = tile_base + state_owner_row_base + local_row_b * 2;
            int value_row_c = tile_base + state_owner_row_base + local_row_c * 2;
            int value_row_d = tile_base + state_owner_row_base + local_row_d * 2;
            int value_smem_row_a = tile_base + group * 4 + local_row_a;
            int value_smem_row_b = tile_base + group * 4 + local_row_b;
            int value_smem_row_c = tile_base + group * 4 + local_row_c;
            int value_smem_row_d = tile_base + group * 4 + local_row_d;
            float value_a = sMixed[384 + value_smem_row_a];
            float value_b = sMixed[384 + value_smem_row_b];
            float value_c = sMixed[384 + value_smem_row_c];
            float value_d = sMixed[384 + value_smem_row_d];
            float delta_a = (value_a - state_key_dot_a) * beta;
            float delta_b = (value_b - state_key_dot_b) * beta;
            float delta_c = (value_c - state_key_dot_c) * beta;
            float delta_d = (value_d - state_key_dot_d) * beta;
            float delta_key_scale_a = delta_a * k_scale;
            float delta_key_scale_b = delta_b * k_scale;
            float delta_key_scale_c = delta_c * k_scale;
            float delta_key_scale_d = delta_d * k_scale;
            if (value_tile == 0) {
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, q_scale, lane_group);
                q_scale = _shfl_2;
                #pragma unroll
                for (int q_relay_i = 0; q_relay_i < 8; q_relay_i++) {
                    float _shfl_3 = __shfl_sync(0xFFFFFFFF, r_q[q_relay_i], lane_group);
                    r_q[q_relay_i] = _shfl_3;
                }
            }
            float recurrence_value_a = 0.0f;
            float recurrence_value_b = 0.0f;
            float2 _f2_33 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_a = _f2_33;
            float2 _f2_34 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_b = _f2_34;
            float recurrence_value_c = 0.0f;
            float recurrence_value_d = 0.0f;
            float2 _f2_35 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_c = _f2_35;
            float2 _f2_36 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_d = _f2_36;
            #pragma unroll
            for (int i_pair_3 = 0; i_pair_3 < 4; i_pair_3++) {
                int i0_3 = i_pair_3 * 2;
                int i1_3 = i0_3 + 1;
                int reg_offset_a_1 = local_row_a * 8 + i0_3;
                int reg_offset_b_1 = local_row_b * 8 + i0_3;
                int reg_offset_c_1 = local_row_c * 8 + i0_3;
                int reg_offset_d_1 = local_row_d * 8 + i0_3;
                float2 _f2_37 = make_float2(r_k[i0_3], r_k[i1_3]);
                float2 key_pair_1 = _f2_37;
                float2 _f2_38 = make_float2(r_q[i0_3], r_q[i1_3]);
                float2 query_pair = _f2_38;
                float2 _f2_39 = make_float2(delta_key_scale_a, delta_key_scale_a);
                float2 _f2_40 = make_float2(state_regs[reg_offset_a_1], state_regs[reg_offset_a_1 + 1]);
                float2 updated_pair_a = fma_f32x2_rn_ftz(_f2_39, key_pair_1, _f2_40);
                float2 _f2_41 = make_float2(delta_key_scale_c, delta_key_scale_c);
                float2 _f2_42 = make_float2(state_regs[reg_offset_c_1], state_regs[reg_offset_c_1 + 1]);
                float2 updated_pair_c = fma_f32x2_rn_ftz(_f2_41, key_pair_1, _f2_42);
                state_regs[reg_offset_a_1] = updated_pair_a.x;
                state_regs[reg_offset_a_1 + 1] = updated_pair_a.y;
                state_regs[reg_offset_c_1] = updated_pair_c.x;
                state_regs[reg_offset_c_1 + 1] = updated_pair_c.y;
                state_query_pair_a = fma_f32x2_rn_ftz(updated_pair_a, query_pair, state_query_pair_a);
                state_query_pair_c = fma_f32x2_rn_ftz(updated_pair_c, query_pair, state_query_pair_c);
            }
            #pragma unroll
            for (int post_i_pair = 0; post_i_pair < 4; post_i_pair++) {
                int post_i0 = post_i_pair * 2;
                int post_i1 = post_i0 + 1;
                int post_reg_offset_b = local_row_b * 8 + post_i0;
                int post_reg_offset_d = local_row_d * 8 + post_i0;
                float2 _f2_43 = make_float2(r_k[post_i0], r_k[post_i1]);
                float2 post_key_pair = _f2_43;
                float2 _f2_44 = make_float2(r_q[post_i0], r_q[post_i1]);
                float2 post_query_pair = _f2_44;
                float2 _f2_45 = make_float2(delta_key_scale_b, delta_key_scale_b);
                float2 _f2_46 = make_float2(state_regs[post_reg_offset_b], state_regs[post_reg_offset_b + 1]);
                float2 post_updated_pair_b = fma_f32x2_rn_ftz(_f2_45, post_key_pair, _f2_46);
                float2 _f2_47 = make_float2(delta_key_scale_d, delta_key_scale_d);
                float2 _f2_48 = make_float2(state_regs[post_reg_offset_d], state_regs[post_reg_offset_d + 1]);
                float2 post_updated_pair_d = fma_f32x2_rn_ftz(_f2_47, post_key_pair, _f2_48);
                state_regs[post_reg_offset_b] = post_updated_pair_b.x;
                state_regs[post_reg_offset_b + 1] = post_updated_pair_b.y;
                state_regs[post_reg_offset_d] = post_updated_pair_d.x;
                state_regs[post_reg_offset_d + 1] = post_updated_pair_d.y;
                state_query_pair_b = fma_f32x2_rn_ftz(post_updated_pair_b, post_query_pair, state_query_pair_b);
                state_query_pair_d = fma_f32x2_rn_ftz(post_updated_pair_d, post_query_pair, state_query_pair_d);
            }
            if (is_live != 0) {
                {
                    unsigned _stv8_4_0 = __float_as_uint(state_regs[local_row_a * 8 + 0]);
                    unsigned _stv8_4_1 = __float_as_uint(state_regs[local_row_a * 8 + 1]);
                    unsigned _stv8_4_2 = __float_as_uint(state_regs[local_row_a * 8 + 2]);
                    unsigned _stv8_4_3 = __float_as_uint(state_regs[local_row_a * 8 + 3]);
                    unsigned _stv8_4_4 = __float_as_uint(state_regs[local_row_a * 8 + 4]);
                    unsigned _stv8_4_5 = __float_as_uint(state_regs[local_row_a * 8 + 5]);
                    unsigned _stv8_4_6 = __float_as_uint(state_regs[local_row_a * 8 + 6]);
                    unsigned _stv8_4_7 = __float_as_uint(state_regs[local_row_a * 8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (state_tile_base + local_row_a * 2 * 128))), "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3), "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7) : "memory");
                }
                {
                    unsigned _stv8_5_0 = __float_as_uint(state_regs[local_row_b * 8 + 0]);
                    unsigned _stv8_5_1 = __float_as_uint(state_regs[local_row_b * 8 + 1]);
                    unsigned _stv8_5_2 = __float_as_uint(state_regs[local_row_b * 8 + 2]);
                    unsigned _stv8_5_3 = __float_as_uint(state_regs[local_row_b * 8 + 3]);
                    unsigned _stv8_5_4 = __float_as_uint(state_regs[local_row_b * 8 + 4]);
                    unsigned _stv8_5_5 = __float_as_uint(state_regs[local_row_b * 8 + 5]);
                    unsigned _stv8_5_6 = __float_as_uint(state_regs[local_row_b * 8 + 6]);
                    unsigned _stv8_5_7 = __float_as_uint(state_regs[local_row_b * 8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (state_tile_base + local_row_b * 2 * 128))), "r"(_stv8_5_0), "r"(_stv8_5_1), "r"(_stv8_5_2), "r"(_stv8_5_3), "r"(_stv8_5_4), "r"(_stv8_5_5), "r"(_stv8_5_6), "r"(_stv8_5_7) : "memory");
                }
                {
                    unsigned _stv8_6_0 = __float_as_uint(state_regs[local_row_c * 8 + 0]);
                    unsigned _stv8_6_1 = __float_as_uint(state_regs[local_row_c * 8 + 1]);
                    unsigned _stv8_6_2 = __float_as_uint(state_regs[local_row_c * 8 + 2]);
                    unsigned _stv8_6_3 = __float_as_uint(state_regs[local_row_c * 8 + 3]);
                    unsigned _stv8_6_4 = __float_as_uint(state_regs[local_row_c * 8 + 4]);
                    unsigned _stv8_6_5 = __float_as_uint(state_regs[local_row_c * 8 + 5]);
                    unsigned _stv8_6_6 = __float_as_uint(state_regs[local_row_c * 8 + 6]);
                    unsigned _stv8_6_7 = __float_as_uint(state_regs[local_row_c * 8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (state_tile_base + local_row_c * 2 * 128))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
                }
                {
                    unsigned _stv8_7_0 = __float_as_uint(state_regs[local_row_d * 8 + 0]);
                    unsigned _stv8_7_1 = __float_as_uint(state_regs[local_row_d * 8 + 1]);
                    unsigned _stv8_7_2 = __float_as_uint(state_regs[local_row_d * 8 + 2]);
                    unsigned _stv8_7_3 = __float_as_uint(state_regs[local_row_d * 8 + 3]);
                    unsigned _stv8_7_4 = __float_as_uint(state_regs[local_row_d * 8 + 4]);
                    unsigned _stv8_7_5 = __float_as_uint(state_regs[local_row_d * 8 + 5]);
                    unsigned _stv8_7_6 = __float_as_uint(state_regs[local_row_d * 8 + 6]);
                    unsigned _stv8_7_7 = __float_as_uint(state_regs[local_row_d * 8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (state_tile_base + local_row_d * 2 * 128))), "r"(_stv8_7_0), "r"(_stv8_7_1), "r"(_stv8_7_2), "r"(_stv8_7_3), "r"(_stv8_7_4), "r"(_stv8_7_5), "r"(_stv8_7_6), "r"(_stv8_7_7) : "memory");
                }
            }
            recurrence_value_a = state_query_pair_a.x + state_query_pair_a.y;
            recurrence_value_b = state_query_pair_b.x + state_query_pair_b.y;
            recurrence_value_c = state_query_pair_c.x + state_query_pair_c.y;
            recurrence_value_d = state_query_pair_d.x + state_query_pair_d.y;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 8);
            recurrence_value_a += _shfl_xor_26;
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 8);
            recurrence_value_b += _shfl_xor_27;
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_c, 8);
            recurrence_value_c += _shfl_xor_28;
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_d, 8);
            recurrence_value_d += _shfl_xor_29;
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 4);
            recurrence_value_a += _shfl_xor_30;
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 4);
            recurrence_value_b += _shfl_xor_31;
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_c, 4);
            recurrence_value_c += _shfl_xor_32;
            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_d, 4);
            recurrence_value_d += _shfl_xor_33;
            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 2);
            recurrence_value_a += _shfl_xor_34;
            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 2);
            recurrence_value_b += _shfl_xor_35;
            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_c, 2);
            recurrence_value_c += _shfl_xor_36;
            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_d, 2);
            recurrence_value_d += _shfl_xor_37;
            float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 1);
            recurrence_value_a += _shfl_xor_38;
            float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 1);
            recurrence_value_b += _shfl_xor_39;
            float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_c, 1);
            recurrence_value_c += _shfl_xor_40;
            float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_d, 1);
            recurrence_value_d += _shfl_xor_41;
            recurrence_value_a *= q_scale;
            recurrence_value_b *= q_scale;
            recurrence_value_c *= q_scale;
            recurrence_value_d *= q_scale;
            if (lane_group == 0) {
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(recurrence_value_a);
                __nv_bfloat16 publication_value_a = _cvt_bf16_0;
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(recurrence_value_c);
                __nv_bfloat16 publication_value_c = _cvt_bf16_1;
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(recurrence_value_b);
                __nv_bfloat16 publication_value_b = _cvt_bf16_2;
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(recurrence_value_d);
                __nv_bfloat16 publication_value_d = _cvt_bf16_3;
                sRecurrence[value_row_a] = publication_value_a;
                sRecurrence[value_row_b] = publication_value_b;
                sRecurrence[value_row_c] = publication_value_c;
                sRecurrence[value_row_d] = publication_value_d;
                if (value_tile == 0) {
                    float _cvt_f32_0 = __bfloat162float(publication_value_a);
                    float first_slab_value = _cvt_f32_0;
                    float first_slab_sum_squares = first_slab_value * first_slab_value;
                    float _cvt_f32_1 = __bfloat162float(publication_value_b);
                    first_slab_value = _cvt_f32_1;
                    first_slab_sum_squares += first_slab_value * first_slab_value;
                    float _cvt_f32_2 = __bfloat162float(publication_value_c);
                    first_slab_value = _cvt_f32_2;
                    first_slab_sum_squares += first_slab_value * first_slab_value;
                    float _cvt_f32_3 = __bfloat162float(publication_value_d);
                    first_slab_value = _cvt_f32_3;
                    first_slab_sum_squares += first_slab_value * first_slab_value;
                    sFirstSlabRmsPartial[group] = first_slab_sum_squares;
                }
            }
        }
        if (value_tile == 0) {
            asm volatile("cp.async.wait_group 0;");
        }
    }
    if (warp == 0) {
        asm volatile("barrier.sync 6, 256;" ::: "memory");
    } else {
        asm volatile("barrier.arrive 6, 256;" ::: "memory");
    }
    if (warp == 0) {
        float output_values[4];
        __nv_bfloat162 recurrence_pair0 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[lane_1 * 2];
        __nv_bfloat162 recurrence_pair1 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[lane_1 * 2 + 1];
        output_values[0] = (float)recurrence_pair0.x;
        output_values[1] = (float)recurrence_pair0.y;
        output_values[2] = (float)recurrence_pair1.x;
        output_values[3] = (float)recurrence_pair1.y;
        float sum_squares = 0.0f;
        if (lane_1 < 16) {
            sum_squares = sFirstSlabRmsPartial[lane_1];
        } else {
            #pragma unroll
            for (int channel_idx_1 = 0; channel_idx_1 < 4; channel_idx_1++) {
                float value = output_values[channel_idx_1];
                sum_squares += value * value;
            }
        }
        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 16);
        sum_squares += _shfl_xor_42;
        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 8);
        sum_squares += _shfl_xor_43;
        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 4);
        sum_squares += _shfl_xor_44;
        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 2);
        sum_squares += _shfl_xor_45;
        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 1);
        sum_squares += _shfl_xor_46;
        float _rsqrt_2 = rsqrtf(sum_squares * 0.0078125f + norm_eps);
        float inverse_rms = _rsqrt_2;
        int output_base = (row * 96 + head) * 128 + lane_1 * 4;
        float output_scales[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&output_scales[0])), "=r"(*reinterpret_cast<uint32_t*>(&output_scales[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&output_scales[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&output_scales[(0) + 3]))
            : "r"(sOutputScale_addr + (unsigned int)(lane_1 * 16)));
        #pragma unroll
        for (int channel_idx_2 = 0; channel_idx_2 < 4; channel_idx_2++) {
            if (is_live != 0) {
                output_values[channel_idx_2] = output_values[channel_idx_2] * inverse_rms * output_scales[channel_idx_2];
            } else {
                output_values[channel_idx_2] = 0.0f;
            }
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(output))[output_base]) = _pk2;
        }
    }
}

} // extern "C"
