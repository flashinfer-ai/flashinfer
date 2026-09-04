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
#define SMEM_SFIRSTSLABRMSPARTIAL_OFF 0
#define SMEM_SFIRSTSLABRMSPARTIAL_STAGE_BYTES 4
#define SMEM_SFIRSTSLABRMSPARTIAL_STRIDE 4
#define SMEM_SASYNCSTATE_OFF 3712
#define SMEM_SASYNCSTATE_STAGE_BYTES 32768
#define SMEM_SASYNCSTATE_STRIDE 32768
#define SMEM_TOTAL 36480
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

__global__ __launch_bounds__(256) void
kernel_kernel(__nv_bfloat16* __restrict__ x, float* __restrict__ weight, __nv_bfloat16* __restrict__ conv_state, __nv_bfloat16* __restrict__ raw_gate, __nv_bfloat16* __restrict__ raw_beta, float* __restrict__ A_log, float* __restrict__ dt_bias, int* __restrict__ state_indices, __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ output_gate, float* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int x_row_stride, int conv_slot_stride, int beta_row_stride, int state_slot_stride, int output_gate_row_stride, int H, int use_lower_bound, float lower_bound_log2, float norm_eps)
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
    float* sFirstSlabRmsPartial = reinterpret_cast<float*>(smem_raw + 0);
    const int sFirstSlabRmsPartial_addr = smem + 0;
    unsigned int* sAsyncState = reinterpret_cast<unsigned int*>(smem_raw + 3712);
    const int sAsyncState_addr = smem + 3712;

    // === Task calls (dependency order) ===
    int row = blockIdx.y;
    int head = blockIdx.x;
    int tid_0 = tid;
    int lane_1 = lane;
    int group = tid_0 / 16;
    int lane_group = tid_0 - group * 16;
    int k_start = lane_group * 8;
    int qk_smem_start = lane_group * 12;
    int state_owner_row_base = group * 4;
    int hidden = H * 128;
    int qkv_size = 3 * hidden;
    int requested_slot = state_indices[row];
    int slot = requested_slot;
    int is_live = 1;
    if (slot <= 0) {
        slot = 0;
    }
    is_live = ((requested_slot > 0) ? 1 : 0);
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
        float _tanh_approx_0;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(acc_pair0.x * 0.5f));
        float silu0 = acc_pair0.x * (_tanh_approx_0 * 0.5f + 0.5f);
        float _tanh_approx_1;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(acc_pair0.y * 0.5f));
        float silu1 = acc_pair0.y * (_tanh_approx_1 * 0.5f + 0.5f);
        float _tanh_approx_2;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(acc_pair1.x * 0.5f));
        float silu2 = acc_pair1.x * (_tanh_approx_2 * 0.5f + 0.5f);
        float _tanh_approx_3;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(acc_pair1.y * 0.5f));
        float silu3 = acc_pair1.y * (_tanh_approx_3 * 0.5f + 0.5f);
        int qk_segment = (3 - qkv_idx) / 2;
        int smem_channel_start = channel_start + channel_start / 8 * 4 * qk_segment;
        int smem_qkv_base = qkv_idx * 192 + smem_channel_start;
        sMixed[smem_qkv_base] = (float)(__nv_bfloat16)silu0;
        sMixed[smem_qkv_base + 1] = (float)(__nv_bfloat16)silu1;
        sMixed[smem_qkv_base + 2] = (float)(__nv_bfloat16)silu2;
        sMixed[smem_qkv_base + 3] = (float)(__nv_bfloat16)silu3;
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
    }
    if (tid_0 >= 128) {
        int k_idx = tid_0 - 128;
        int gate_idx = (row * H + head) * 128 + k_idx;
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
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(state + state_group_base + local_row * 128);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&state_carriers[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&state_regs[local_row * 8 + _pair * 2])[0]), "=f"((&state_regs[local_row * 8 + _pair * 2])[1])
                : "r"(state_carriers[_pair]));
        }
    }
    __syncthreads();
    float2 _f2_16 = make_float2(0.0f, 0.0f);
    float2 q_sq_pair = _f2_16;
    float2 _f2_17 = make_float2(0.0f, 0.0f);
    float2 k_sq_pair = _f2_17;
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
            float2 _f2_18 = make_float2(sMixed[qk_smem_start + i0], sMixed[qk_smem_start + i1]);
            float2 q_pair = _f2_18;
            float2 _f2_19 = make_float2(sMixed[192 + qk_smem_start + i0], sMixed[192 + qk_smem_start + i1]);
            float2 k_pair = _f2_19;
            r_q[i0] = q_pair.x;
            r_q[i1] = q_pair.y;
            r_k[i0] = k_pair.x;
            r_k[i1] = k_pair.y;
            q_sq_pair = fma_f32x2_rn_ftz(q_pair, q_pair, q_sq_pair);
            k_sq_pair = fma_f32x2_rn_ftz(k_pair, k_pair, k_sq_pair);
        }
    }
    float q_sq = q_sq_pair.x + q_sq_pair.y;
    float k_sq = k_sq_pair.x + k_sq_pair.y;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 8);
    q_sq += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 8);
    k_sq += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 4);
    q_sq += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 4);
    k_sq += _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 2);
    q_sq += _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 2);
    k_sq += _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 1);
    q_sq += _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 1);
    k_sq += _shfl_xor_7;
    float q_scale = 0.0f;
    float k_scale = 0.0f;
    if (lane_1 < 16) {
        float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
        q_scale = _rsqrt_0 * 0.08838834764831845f;
        float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
        k_scale = _rsqrt_1;
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, q_scale, lane_group);
    q_scale = _shfl_0;
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, k_scale, lane_group);
    k_scale = _shfl_1;
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, r_q[i_1], lane_group);
        r_q[i_1] = _shfl_2;
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, r_k[i_1], lane_group);
        r_k[i_1] = _shfl_3;
    }
    float beta = sBeta[0];
    #pragma unroll
    for (int i_pair_1 = 0; i_pair_1 < 4; i_pair_1++) {
        int i0_1 = i_pair_1 * 2;
        int i1_1 = i0_1 + 1;
        r_decay[i0_1] = sGateDecay[qk_smem_start + i0_1];
        r_decay[i1_1] = sGateDecay[qk_smem_start + i1_1];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    int async_state_group_row = group * 4;
    int async_state_src_base = state_group_base + 8192;
    #pragma unroll
    for (int local_row_1 = 0; local_row_1 < 4; local_row_1++) {
        int async_state_elem = (async_state_group_row + local_row_1) * 128 + k_start;
        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
            :: "r"(sAsyncState_addr + (unsigned int)(async_state_elem * 2)), "l"(state + (async_state_src_base + local_row_1 * 128)));
    }
    asm volatile("cp.async.commit_group;");
    #pragma unroll
    for (int value_tile = 0; value_tile < 2; value_tile++) {
        int tile_base = value_tile * 64;
        int state_tile_base = state_group_base + tile_base * 128;
        if (value_tile > 0) {
            asm volatile("cp.async.wait_group 0;");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            #pragma unroll
            for (int local_row_2 = 0; local_row_2 < 4; local_row_2++) {
                int async_state_local_row = local_row_2;
                int async_state_elem_1 = (group * 4 + local_row_2) * 128 + k_start;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 3]))
                    : "r"(sAsyncState_addr + (unsigned int)(async_state_elem_1 * 2)));
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&state_regs[async_state_local_row * 8 + _pair * 2])[0]), "=f"((&state_regs[async_state_local_row * 8 + _pair * 2])[1])
                        : "r"(state_carriers[_pair]));
                }
            }
        }
        #pragma unroll
        for (int row_group = 0; row_group < 4 / ((0) ? 4 : 2); row_group++) {
            int local_row_a = row_group * ((0) ? 4 : 2);
            int local_row_b = local_row_a + 1;
            float2 _f2_20 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_a = _f2_20;
            float2 _f2_21 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_b = _f2_21;
            #pragma unroll
            for (int i_pair_2 = 0; i_pair_2 < 4; i_pair_2++) {
                int i0_2 = i_pair_2 * 2;
                int i1_2 = i0_2 + 1;
                int reg_offset_a = local_row_a * 8 + i0_2;
                int reg_offset_b = local_row_b * 8 + i0_2;
                float2 _f2_22 = make_float2(r_decay[i0_2], r_decay[i1_2]);
                float2 decay_pair = _f2_22;
                float2 _f2_23 = make_float2(r_k[i0_2], r_k[i1_2]);
                float2 key_pair = _f2_23;
                float2 _f2_24 = make_float2(state_regs[reg_offset_a], state_regs[reg_offset_a + 1]);
                float2 _mul_f32x2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_a = _mul_f32x2_2;
                float2 _f2_25 = make_float2(state_regs[reg_offset_b], state_regs[reg_offset_b + 1]);
                float2 _mul_f32x2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_25), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_b = _mul_f32x2_3;
                state_regs[reg_offset_a] = state_pair_a.x;
                state_regs[reg_offset_a + 1] = state_pair_a.y;
                state_regs[reg_offset_b] = state_pair_b.x;
                state_regs[reg_offset_b + 1] = state_pair_b.y;
                state_key_pair_a = fma_f32x2_rn_ftz(state_pair_a, key_pair, state_key_pair_a);
                state_key_pair_b = fma_f32x2_rn_ftz(state_pair_b, key_pair, state_key_pair_b);
            }
            float state_key_dot_a = state_key_pair_a.x + state_key_pair_a.y;
            float state_key_dot_b = state_key_pair_b.x + state_key_pair_b.y;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 8);
            state_key_dot_a += _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 8);
            state_key_dot_b += _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 4);
            state_key_dot_a += _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 4);
            state_key_dot_b += _shfl_xor_11;
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 2);
            state_key_dot_a += _shfl_xor_12;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 2);
            state_key_dot_b += _shfl_xor_13;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 1);
            state_key_dot_a += _shfl_xor_14;
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 1);
            state_key_dot_b += _shfl_xor_15;
            state_key_dot_a *= k_scale;
            state_key_dot_b *= k_scale;
            int value_row_a = tile_base + state_owner_row_base + local_row_a;
            int value_row_b = tile_base + state_owner_row_base + local_row_b;
            int value_smem_row_a = value_row_a;
            int value_smem_row_b = value_row_b;
            float value_a = sMixed[384 + value_smem_row_a];
            float value_b = sMixed[384 + value_smem_row_b];
            float delta_a = (value_a - state_key_dot_a) * beta;
            float delta_b = (value_b - state_key_dot_b) * beta;
            float delta_key_scale_a = delta_a * k_scale;
            float delta_key_scale_b = delta_b * k_scale;
            float recurrence_value_a = 0.0f;
            float recurrence_value_b = 0.0f;
            float2 _f2_26 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_a = _f2_26;
            float2 _f2_27 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_b = _f2_27;
            #pragma unroll
            for (int i_pair_3 = 0; i_pair_3 < 4; i_pair_3++) {
                int i0_3 = i_pair_3 * 2;
                int i1_3 = i0_3 + 1;
                int reg_offset_a_1 = local_row_a * 8 + i0_3;
                int reg_offset_b_1 = local_row_b * 8 + i0_3;
                float2 _f2_28 = make_float2(r_k[i0_3], r_k[i1_3]);
                float2 key_pair_1 = _f2_28;
                float2 _f2_29 = make_float2(delta_key_scale_a, delta_key_scale_a);
                float2 _f2_30 = make_float2(state_regs[reg_offset_a_1], state_regs[reg_offset_a_1 + 1]);
                float2 updated_pair_a = fma_f32x2_rn_ftz(_f2_29, key_pair_1, _f2_30);
                float2 _f2_31 = make_float2(delta_key_scale_b, delta_key_scale_b);
                float2 _f2_32 = make_float2(state_regs[reg_offset_b_1], state_regs[reg_offset_b_1 + 1]);
                float2 updated_pair_b = fma_f32x2_rn_ftz(_f2_31, key_pair_1, _f2_32);
                state_regs[reg_offset_a_1] = updated_pair_a.x;
                state_regs[reg_offset_a_1 + 1] = updated_pair_a.y;
                state_regs[reg_offset_b_1] = updated_pair_b.x;
                state_regs[reg_offset_b_1 + 1] = updated_pair_b.y;
                float2 _f2_33 = make_float2(r_q[i0_3], r_q[i1_3]);
                float2 query_pair = _f2_33;
                state_query_pair_a = fma_f32x2_rn_ftz(updated_pair_a, query_pair, state_query_pair_a);
                state_query_pair_b = fma_f32x2_rn_ftz(updated_pair_b, query_pair, state_query_pair_b);
            }
            if (is_live != 0) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(state_regs[local_row_a * 8 + 0], state_regs[local_row_a * 8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(state_regs[local_row_a * 8 + 2], state_regs[local_row_a * 8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(state_regs[local_row_a * 8 + 4], state_regs[local_row_a * 8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(state_regs[local_row_a * 8 + 6], state_regs[local_row_a * 8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[state_tile_base + local_row_a * 128 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(state_regs[local_row_b * 8 + 0], state_regs[local_row_b * 8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(state_regs[local_row_b * 8 + 2], state_regs[local_row_b * 8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(state_regs[local_row_b * 8 + 4], state_regs[local_row_b * 8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(state_regs[local_row_b * 8 + 6], state_regs[local_row_b * 8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[state_tile_base + local_row_b * 128 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            recurrence_value_a = state_query_pair_a.x + state_query_pair_a.y;
            recurrence_value_b = state_query_pair_b.x + state_query_pair_b.y;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 8);
            recurrence_value_a += _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 8);
            recurrence_value_b += _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 4);
            recurrence_value_a += _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 4);
            recurrence_value_b += _shfl_xor_19;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 2);
            recurrence_value_a += _shfl_xor_20;
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 2);
            recurrence_value_b += _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 1);
            recurrence_value_a += _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 1);
            recurrence_value_b += _shfl_xor_23;
            recurrence_value_a *= q_scale;
            recurrence_value_b *= q_scale;
            if (lane_group == 0) {
                sRecurrence[value_row_a] = recurrence_value_a;
                sRecurrence[value_row_b] = recurrence_value_b;
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        float output_values[4];
        __nv_bfloat162 recurrence_pair0 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[lane_1 * 2];
        __nv_bfloat162 recurrence_pair1 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[lane_1 * 2 + 1];
        output_values[0] = (float)recurrence_pair0.x;
        output_values[1] = (float)recurrence_pair0.y;
        output_values[2] = (float)recurrence_pair1.x;
        output_values[3] = (float)recurrence_pair1.y;
        float sum_squares = 0.0f;
        #pragma unroll
        for (int channel_idx_1 = 0; channel_idx_1 < 4; channel_idx_1++) {
            float value = output_values[channel_idx_1];
            sum_squares += value * value;
        }
        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 16);
        sum_squares += _shfl_xor_24;
        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 8);
        sum_squares += _shfl_xor_25;
        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 4);
        sum_squares += _shfl_xor_26;
        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 2);
        sum_squares += _shfl_xor_27;
        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, sum_squares, 1);
        sum_squares += _shfl_xor_28;
        float _rsqrt_2 = rsqrtf(sum_squares * 0.0078125f + norm_eps);
        float inverse_rms = _rsqrt_2;
        int output_base = (row * H + head) * 128 + lane_1 * 4;
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
