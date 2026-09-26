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
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_PARTIAL_OFF 0
#define SMEM_SMEM_PARTIAL_STAGE_BYTES 16
#define SMEM_SMEM_PARTIAL_STRIDE 16
#define SMEM_TOTAL 128
#define THREADS 128
#define HIDDEN 5376
#define SCALE_COLS 336
#define PACKED_COLS 2688

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

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

__global__ __launch_bounds__(128) void
kernel_cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, float* __restrict__ x_global_scale, uint8_t* __restrict__ activation_q, uint8_t* __restrict__ activation_sf, __nv_bfloat16* __restrict__ debug_adaln_bf16, int write_debug, float eps, int M)
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
    float* smem_partial = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_partial_addr = smem + 0;

    // === Task calls (dependency order) ===
    int thread = warp * 32 + lane;
    int row_in_cta = thread / 64;
    int t = thread % 64;
    int warp_in_row = warp % 2;
    int row = bid * 2 + row_in_cta;
    unsigned int x_carrier[44];
    #pragma unroll
    for (int q = 0; q < 44; q++) {
        x_carrier[q] = 0;
    }
    if (row < M) {
        unsigned long long row_base = (unsigned long long)row * (unsigned long long)HIDDEN;
        #pragma unroll
        for (int k = 0; k < 10; k++) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row_base + (unsigned long long)(k * 512) + (unsigned long long)(t * 8));
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_carrier[4 * k]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
        }
        if (t < 32) {
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + row_base + 5120 + (unsigned long long)(t * 8));
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&x_carrier[40]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
        }
    }
    float x_carrier_f32[88];
    #pragma unroll
    for (int _pair = 0; _pair < 44; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&x_carrier_f32[_pair * 2])[0]), "=f"((&x_carrier_f32[_pair * 2])[1])
            : "r"(x_carrier[_pair]));
    }
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 88; i++) {
        float _fma_0 = __fmaf_rn(x_carrier_f32[i], x_carrier_f32[i], sum_sq);
        sum_sq = _fma_0;
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
    sum_sq += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
    sum_sq += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
    sum_sq += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
    sum_sq += _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
    sum_sq += _shfl_xor_4;
    if (lane == 0) {
        smem_partial[row_in_cta * 2 + warp_in_row] = sum_sq;
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    float total_sq = smem_partial[row_in_cta * 2];
    total_sq += smem_partial[row_in_cta * 2 + 1];
    float _fdiv_rn_0 = __fdiv_rn(total_sq, 5376.0f);
    float mean_sq = _fdiv_rn_0;
    float _rsqrt_0 = rsqrtf(mean_sq + eps);
    float rstd = _rsqrt_0;
    int table_row = -1;
    if (row < M) {
        table_row = adaln_index[row];
    }
    float global_scale = x_global_scale[0];
    float _rcp_0 = approx_rcp(6.0f);
    float rcp_six = _rcp_0;
    float _rcp_1 = approx_rcp(global_scale);
    float rcp_global_scale = _rcp_1;
    float values[8];
    float normalized[8];
    float absolute[8];
    float quant_values[8];
    unsigned int packed[1];
    #pragma unroll
    for (int k_1 = 0; k_1 < 10; k_1++) {
        int col = k_1 * 512 + t * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            values[j] = 0.0f;
            normalized[j] = 0.0f;
        }
        if (row < M) {
            if (table_row >= 0 && table_row < 9) {
                unsigned long long table_base = (unsigned long long)table_row * (unsigned long long)HIDDEN;
                float _vec_load_0[8];
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x_norm_weight + col + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_2[_pair]));
                        }
                    }
                }
                float _vec_load_1[8];
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(adaln_scale + (table_base + (unsigned long long)col) + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                float _vec_load_2[8];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(adaln_shift + (table_base + (unsigned long long)col) + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_4[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    float scaled = x_carrier_f32[k_1 * 8 + j_1];
                    float _mul_0 = scaled * rstd;
                    scaled = _mul_0;
                    float _mul_1 = scaled * _vec_load_0[j_1];
                    scaled = _mul_1;
                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(scaled);
                    float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                    normalized[j_1] = _cvt_f32_0;
                }
                const float2 _add2_5 = {1.0f, 1.0f};
                #pragma unroll
                for (int _la = 0; _la < 4; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(_vec_load_1)[_la], _add2_5);
                uint32_t _vec_load_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_1[_lp*2 + 0], _vec_load_1[_lp*2+1 + 0]));
                    _vec_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _vec_load_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_1_bf16_f32[_pair * 2])[0]), "=f"((&_vec_load_1_bf16_f32[_pair * 2])[1])
                        : "r"(_vec_load_1_bf16[_pair]));
                }
                #pragma unroll
                for (int pair = 0; pair < 4; pair++) {
                    int pair_offset = pair * 2;
                    float2 _f2_0 = make_float2(normalized[pair_offset], normalized[pair_offset + 1]);
                    float2 normalized_pair = _f2_0;
                    float2 _f2_1 = make_float2(_vec_load_1_bf16_f32[pair_offset], _vec_load_1_bf16_f32[pair_offset + 1]);
                    float2 scale_pair = _f2_1;
                    float2 _f2_2 = make_float2(_vec_load_2[pair_offset], _vec_load_2[pair_offset + 1]);
                    float2 shift_pair = _f2_2;
                    float2 adaln_pair = fma_f32x2_rn_ftz(normalized_pair, scale_pair, shift_pair);
                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(adaln_pair.x);
                    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                    values[pair_offset] = _cvt_f32_1;
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(adaln_pair.y);
                    float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                    values[pair_offset + 1] = _cvt_f32_2;
                }
            }
            if (write_debug == 2) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(normalized[0 + 0], normalized[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(normalized[0 + 2], normalized[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(normalized[0 + 4], normalized[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(normalized[0 + 6], normalized[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            } else if (write_debug != 0) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values[0 + 0], values[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values[0 + 2], values[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values[0 + 4], values[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values[0 + 6], values[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            absolute[j_2] = values[j_2];
        }
        float _fabs_0 = fabsf(absolute[0]);
        absolute[0] = _fabs_0;
        float _fabs_1 = fabsf(absolute[1]);
        absolute[1] = _fabs_1;
        float _fabs_2 = fabsf(absolute[2]);
        absolute[2] = _fabs_2;
        float _fabs_3 = fabsf(absolute[3]);
        absolute[3] = _fabs_3;
        float _fabs_4 = fabsf(absolute[4]);
        absolute[4] = _fabs_4;
        float _fabs_5 = fabsf(absolute[5]);
        absolute[5] = _fabs_5;
        float _fabs_6 = fabsf(absolute[6]);
        absolute[6] = _fabs_6;
        float _fabs_7 = fabsf(absolute[7]);
        absolute[7] = _fabs_7;
        float absolute_max = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            absolute_max = max_noftz(absolute_max, absolute[_lr]);
        }
        float amax = absolute_max;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
        float _max_0 = max_noftz(amax, _shfl_xor_5);
        amax = _max_0;
        float sf_value = global_scale * (amax * rcp_six);
        float _fp8_rt_0;
        uint16_t _e4m3x2_6;
        uint32_t _f16x2_6;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_6) : "f"(0.0f), "f"(sf_value));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6) : "h"(_e4m3x2_6));
        uint16_t _fp8_h0_6 = (uint16_t)(_f16x2_6 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_6));
        float sf_rounded = _fp8_rt_0;
        float _rcp_2 = approx_rcp(sf_rounded * rcp_global_scale);
        float _min_0 = fminf(_rcp_2, 3.4028234663852886e+38f);
        float output_scale = _min_0;
        if (row < M) {
            #pragma unroll
            for (int j_3 = 0; j_3 < 8; j_3++) {
                quant_values[j_3] = values[j_3] * output_scale;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            *(reinterpret_cast<int*>(activation_q + ((unsigned long long)row * (unsigned long long)PACKED_COLS + (unsigned long long)(k_1 * 256 + t * 4))) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            int scale_col = k_1 * 32 + t / 2;
            unsigned long long scale_offset = (unsigned long long)(scale_col % 4 + scale_col / 4 * 512 + row % 32 * 16 + row % 128 / 32 * 4) + (unsigned long long)(row / 128) * (unsigned long long)(128 * SCALE_COLS);
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                *(reinterpret_cast<unsigned char*>(activation_sf + scale_offset) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        }
    }
    if (t < 32) {
        int col_1 = 5120 + t * 8;
        #pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
            values[j_4] = 0.0f;
            normalized[j_4] = 0.0f;
        }
        if (row < M) {
            if (table_row >= 0 && table_row < 9) {
                unsigned long long table_base_1 = (unsigned long long)table_row * (unsigned long long)HIDDEN;
                float _vec_load_3[8];
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x_norm_weight + col_1 + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_7[_pair]));
                        }
                    }
                }
                float _vec_load_4[8];
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(adaln_scale + (table_base_1 + (unsigned long long)col_1) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_8[_pair]));
                        }
                    }
                }
                float _vec_load_5[8];
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(adaln_shift + (table_base_1 + (unsigned long long)col_1) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_9[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    float scaled_1 = x_carrier_f32[80 + j_5];
                    float _mul_2 = scaled_1 * rstd;
                    scaled_1 = _mul_2;
                    float _mul_3 = scaled_1 * _vec_load_3[j_5];
                    scaled_1 = _mul_3;
                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(scaled_1);
                    float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                    normalized[j_5] = _cvt_f32_3;
                }
                const float2 _add2_10 = {1.0f, 1.0f};
                #pragma unroll
                for (int _la = 0; _la < 4; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(_vec_load_4)[_la], _add2_10);
                uint32_t _vec_load_4_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_4[_lp*2 + 0], _vec_load_4[_lp*2+1 + 0]));
                    _vec_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _vec_load_4_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_4_bf16_f32[_pair * 2])[0]), "=f"((&_vec_load_4_bf16_f32[_pair * 2])[1])
                        : "r"(_vec_load_4_bf16[_pair]));
                }
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                    int pair_offset_1 = pair_1 * 2;
                    float2 _f2_3 = make_float2(normalized[pair_offset_1], normalized[pair_offset_1 + 1]);
                    float2 normalized_pair_1 = _f2_3;
                    float2 _f2_4 = make_float2(_vec_load_4_bf16_f32[pair_offset_1], _vec_load_4_bf16_f32[pair_offset_1 + 1]);
                    float2 scale_pair_1 = _f2_4;
                    float2 _f2_5 = make_float2(_vec_load_5[pair_offset_1], _vec_load_5[pair_offset_1 + 1]);
                    float2 shift_pair_1 = _f2_5;
                    float2 adaln_pair_1 = fma_f32x2_rn_ftz(normalized_pair_1, scale_pair_1, shift_pair_1);
                    __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(adaln_pair_1.x);
                    float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                    values[pair_offset_1] = _cvt_f32_4;
                    __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(adaln_pair_1.y);
                    float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                    values[pair_offset_1 + 1] = _cvt_f32_5;
                }
            }
            if (write_debug == 2) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(normalized[0 + 0], normalized[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(normalized[0 + 2], normalized[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(normalized[0 + 4], normalized[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(normalized[0 + 6], normalized[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            } else if (write_debug != 0) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values[0 + 0], values[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values[0 + 2], values[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values[0 + 4], values[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values[0 + 6], values[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            absolute[j_6] = values[j_6];
        }
        float _fabs_8 = fabsf(absolute[0]);
        absolute[0] = _fabs_8;
        float _fabs_9 = fabsf(absolute[1]);
        absolute[1] = _fabs_9;
        float _fabs_10 = fabsf(absolute[2]);
        absolute[2] = _fabs_10;
        float _fabs_11 = fabsf(absolute[3]);
        absolute[3] = _fabs_11;
        float _fabs_12 = fabsf(absolute[4]);
        absolute[4] = _fabs_12;
        float _fabs_13 = fabsf(absolute[5]);
        absolute[5] = _fabs_13;
        float _fabs_14 = fabsf(absolute[6]);
        absolute[6] = _fabs_14;
        float _fabs_15 = fabsf(absolute[7]);
        absolute[7] = _fabs_15;
        float absolute_max_1 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            absolute_max_1 = max_noftz(absolute_max_1, absolute[_lr]);
        }
        float amax_1 = absolute_max_1;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, amax_1, 1);
        float _max_1 = max_noftz(amax_1, _shfl_xor_6);
        amax_1 = _max_1;
        float sf_value_1 = global_scale * (amax_1 * rcp_six);
        float _fp8_rt_1;
        uint16_t _e4m3x2_11;
        uint32_t _f16x2_11;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_11) : "f"(0.0f), "f"(sf_value_1));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_11) : "h"(_e4m3x2_11));
        uint16_t _fp8_h0_11 = (uint16_t)(_f16x2_11 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_11));
        float sf_rounded_1 = _fp8_rt_1;
        float _rcp_3 = approx_rcp(sf_rounded_1 * rcp_global_scale);
        float _min_1 = fminf(_rcp_3, 3.4028234663852886e+38f);
        float output_scale_1 = _min_1;
        if (row < M) {
            #pragma unroll
            for (int j_7 = 0; j_7 < 8; j_7++) {
                quant_values[j_7] = values[j_7] * output_scale_1;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            *(reinterpret_cast<int*>(activation_q + ((unsigned long long)row * (unsigned long long)PACKED_COLS + (unsigned long long)(2560 + t * 4))) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            int scale_col_1 = 320 + t / 2;
            unsigned long long scale_offset_1 = (unsigned long long)(scale_col_1 % 4 + scale_col_1 / 4 * 512 + row % 32 * 16 + row % 128 / 32 * 4) + (unsigned long long)(row / 128) * (unsigned long long)(128 * SCALE_COLS);
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value_1));
                *(reinterpret_cast<unsigned char*>(activation_sf + scale_offset_1) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        }
    }
}

} // extern "C"
