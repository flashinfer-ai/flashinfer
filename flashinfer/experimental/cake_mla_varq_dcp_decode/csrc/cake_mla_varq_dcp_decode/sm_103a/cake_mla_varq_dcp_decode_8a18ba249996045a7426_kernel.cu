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
#define THREADS 512
#define USE_PDL 1

#include <math_constants.h>

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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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

__global__ __launch_bounds__(512) void
kernel_cake_mla_varq_dcp_decode_8a18ba249996045a7426(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, unsigned int* __restrict__ unit_flags, unsigned int* __restrict__ split_meta, unsigned int* __restrict__ merge_ctl, int* __restrict__ cum_seq_lens_q, int num_heads, int tiles_max, int max_records)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int stride_m = gridDim.x;
    int jobs_max = max_records * 8;
    int job_t = bid;
    unsigned int n_pub = 0;
    unsigned int n_slots_all = 0;
    unsigned int fin_early = 0;
    int counted = 0;
    #pragma unroll 1
    for (int _job_iter = 0; _job_iter < max_records * 8 + 1; _job_iter++) {
        if (job_t >= jobs_max) {
            break;
        }
        int rec_m = job_t / 8;
        int job_m = job_t - rec_m * 8;
        #pragma unroll 1
        for (int _poll_n = 0; _poll_n < 67108864; _poll_n++) {
            if (n_pub != 0) {
                break;
            }
            unsigned int c_l0 = 0;
            unsigned int s_l0 = 0;
            if (lane == 0) {
                unsigned int _load_acquire_0;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_0) : "l"((reinterpret_cast<unsigned int*>(merge_ctl) + (1))) : "memory");
                c_l0 = _load_acquire_0;
                s_l0 = merge_ctl[3];
            }
            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, c_l0, 0);
            n_pub = _shfl_0;
            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, s_l0, 0);
            n_slots_all = _shfl_1;
        }
        if (rec_m >= (int)n_pub - 1) {
            break;
        }
        unsigned int meta_m = split_meta[rec_m * 2];
        int item_m = (int)split_meta[rec_m * 2 + 1];
        int units_m = (int)(meta_m & 511);
        int slot_base_m = (int)(meta_m >> 9);
        int b_m = item_m / tiles_max;
        int mt_m = item_m - b_m * tiles_max;
        int q_begin_m = cum_seq_lens_q[b_m];
        int q_end_m = cum_seq_lens_q[b_m + 1];
        int out_base_m = q_begin_m * num_heads + mt_m * 128;
        int rows_rem_m = (q_end_m - q_begin_m) * num_heads - mt_m * 128;
        int valid_rows_m = ((rows_rem_m < 128) ? rows_rem_m : 128);
        int all_set = 0;
        #pragma unroll 1
        for (int _poll = 0; _poll < 67108864; _poll++) {
            unsigned int min_f = 1;
            #pragma unroll
            for (int fi = 0; fi < 8; fi++) {
                int u_f = lane + fi * 32;
                if (u_f < units_m) {
                    unsigned int _load_acquire_1;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_1) : "l"((reinterpret_cast<unsigned int*>(unit_flags) + (slot_base_m + u_f))) : "memory");
                    unsigned int f_u = _load_acquire_1;
                    min_f = ((min_f < f_u) ? min_f : f_u);
                }
            }
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(min_f));
            unsigned int all_f = _warp_redux_u32_0;
            if (all_f != 0) {
                all_set = 1;
                break;
            }
        }
        if (all_set == 0) {
            asm volatile("trap;" ::: "memory");
        }
        int next_job = job_t + stride_m;
        int is_last = ((next_job >= jobs_max) ? 1 : 0);
        if (next_job / 8 >= (int)n_pub - 1) {
            is_last = 1;
        }
        if (is_last != 0) {
            asm volatile("barrier.sync 8, 512;" ::: "memory");
            if (warp == 0) {
                if (lane == 0) {
                    unsigned int _atomic_old_0;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_0) : "l"(&merge_ctl[2]), "r"(static_cast<uint32_t>(1)) : "memory");
                    fin_early = _atomic_old_0;
                }
            }
            counted = 1;
        }
        int row_m = job_m * 16 + warp;
        int d_base_m = lane * 16;
        int stat_m = slot_base_m * 128 + row_m;
        unsigned int pa_0[16];
        unsigned int pb_0[16];
        #pragma unroll
        for (int k0 = 0; k0 < 4; k0++) {
            if (units_m > k0) {
                int po_base_0 = (stat_m + k0 * 128) * 512 + d_base_m;
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + po_base_0 + 0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&pa_0[k0 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + (po_base_0 + 8) + 0);
                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&pb_0[k0 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_1[_blk] = _vptr_1[_blk];
                    }
                }
            }
        }
        float local_m[8];
        float local_w[8];
        float thread_max = -CAKE_INF;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int split_i = lane + i * 32;
            local_m[i] = -CAKE_INF;
            local_w[i] = 0.0f;
            if (split_i < units_m) {
                local_m[i] = partial_lse[stat_m + split_i * 128];
                float _max_0 = max_noftz(thread_max, local_m[i]);
                thread_max = _max_0;
            }
        }
        float _warp_reduce_0 = thread_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float global_max = _warp_reduce_0;
        float thread_sum = 0.0f;
        #pragma unroll
        for (int i2 = 0; i2 < 8; i2++) {
            int split_i2 = lane + i2 * 32;
            if (split_i2 < units_m) {
                float _exp2_0 = approx_exp2(local_m[i2] - global_max);
                local_w[i2] = ((local_m[i2] == -CAKE_INF) ? 0.0f : _exp2_0);
                thread_sum = thread_sum + local_w[i2];
            }
        }
        float _warp_reduce_1 = thread_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float global_sum = _warp_reduce_1;
        float _rcp_0 = approx_rcp(global_sum);
        float inv_sum_m = ((global_sum > 0.0f) ? _rcp_0 : 0.0f);
        int row_ok = row_m < valid_rows_m;
        if (lane == 0) {
            if (row_ok != 0) {
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(global_sum));
                float lse_m = global_max + _log2_0;
                float lse_nat_m = ((global_sum > 0.0f) ? lse_m * 0.6931471805599453f : -CAKE_INF);
                *(reinterpret_cast<float*>(LSE + (out_base_m + row_m)) + (0)) = lse_nat_m;
            }
        }
        float acc_m[16];
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            acc_m[e] = 0.0f;
        }
        #pragma unroll
        for (int k1 = 0; k1 < 4; k1++) {
            float _shfl_2;
            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(local_w[0]), "r"(k1));
            float w_0 = _shfl_2;
            w_0 = ((units_m > k1) ? w_0 : 0.0f);
            #pragma unroll
            for (int j1 = 0; j1 < 4; j1++) {
                unsigned int wa_0 = pa_0[k1 * 4 + j1];
                unsigned int wb_0 = pb_0[k1 * 4 + j1];
                acc_m[2 * j1] = acc_m[2 * j1] + ((w_0 > 0.0f) ? w_0 * __uint_as_float(wa_0 << 16) : 0.0f);
                acc_m[2 * j1 + 1] = acc_m[2 * j1 + 1] + ((w_0 > 0.0f) ? w_0 * __uint_as_float(wa_0 & 4294901760) : 0.0f);
                acc_m[8 + 2 * j1] = acc_m[8 + 2 * j1] + ((w_0 > 0.0f) ? w_0 * __uint_as_float(wb_0 << 16) : 0.0f);
                acc_m[8 + 2 * j1 + 1] = acc_m[8 + 2 * j1 + 1] + ((w_0 > 0.0f) ? w_0 * __uint_as_float(wb_0 & 4294901760) : 0.0f);
            }
        }
        #pragma unroll 1
        for (int chunk = 4; chunk < units_m; chunk += 4) {
            float w_c[4];
            unsigned int pa_c[16];
            unsigned int pb_c[16];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int sp_k = chunk + k;
                int sp_c = ((sp_k < units_m) ? sp_k : units_m - 1);
                int grp_c = sp_c >> 5;
                float w_src = local_w[0];
                #pragma unroll
                for (int gsel = 1; gsel < 8; gsel++) {
                    w_src = ((grp_c == gsel) ? local_w[gsel] : w_src);
                }
                float _shfl_3;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_3) : "f"(w_src), "r"(sp_c & 31));
                float w_k = _shfl_3;
                w_c[k] = ((sp_k < units_m) ? w_k : 0.0f);
                if (sp_k < units_m) {
                    int po_base_k = (stat_m + sp_k * 128) * 512 + d_base_m;
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + po_base_k + 0);
                        uint4* _vdst_2 = reinterpret_cast<uint4*>(&pa_c[k * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_2[_blk] = _vptr_2[_blk];
                        }
                    }
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_O + (po_base_k + 8) + 0);
                        uint4* _vdst_3 = reinterpret_cast<uint4*>(&pb_c[k * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_3[_blk] = _vptr_3[_blk];
                        }
                    }
                }
            }
            #pragma unroll
            for (int k2 = 0; k2 < 4; k2++) {
                #pragma unroll
                for (int j2 = 0; j2 < 4; j2++) {
                    unsigned int wa_c = pa_c[k2 * 4 + j2];
                    unsigned int wb_c = pb_c[k2 * 4 + j2];
                    acc_m[2 * j2] = acc_m[2 * j2] + ((w_c[k2] > 0.0f) ? w_c[k2] * __uint_as_float(wa_c << 16) : 0.0f);
                    acc_m[2 * j2 + 1] = acc_m[2 * j2 + 1] + ((w_c[k2] > 0.0f) ? w_c[k2] * __uint_as_float(wa_c & 4294901760) : 0.0f);
                    acc_m[8 + 2 * j2] = acc_m[8 + 2 * j2] + ((w_c[k2] > 0.0f) ? w_c[k2] * __uint_as_float(wb_c << 16) : 0.0f);
                    acc_m[8 + 2 * j2 + 1] = acc_m[8 + 2 * j2 + 1] + ((w_c[k2] > 0.0f) ? w_c[k2] * __uint_as_float(wb_c & 4294901760) : 0.0f);
                }
            }
        }
        if (row_ok != 0) {
            {
                const float2 _prescale2_4 = {inv_sum_m, inv_sum_m};
                #if __CUDA_ARCH__ >= 1000
                #pragma unroll
                for (int _ps = 0; _ps < 8; _ps++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc_m[0])[_ps], _prescale2_4);
                #else
                #pragma unroll
                for (int _ps = 0; _ps < 16; _ps++)
                    acc_m[0 + _ps] *= inv_sum_m;
                #endif
                {
                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(acc_m[0 + 0], acc_m[0 + 1]);
                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(acc_m[0 + 2], acc_m[0 + 3]);
                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(acc_m[0 + 4], acc_m[0 + 5]);
                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(acc_m[0 + 6], acc_m[0 + 7]);
                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(acc_m[0 + 8], acc_m[0 + 9]);
                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(acc_m[0 + 10], acc_m[0 + 11]);
                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(acc_m[0 + 12], acc_m[0 + 13]);
                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(acc_m[0 + 14], acc_m[0 + 15]);
                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(&((__nv_bfloat16*)(O + ((out_base_m + row_m) * 512 + d_base_m)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                }
            }
        }
        job_t = job_t + stride_m;
    }
    if (counted == 0) {
        asm volatile("barrier.sync 8, 512;" ::: "memory");
        if (warp == 0) {
            if (lane == 0) {
                unsigned int _atomic_old_1;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_1) : "l"(&merge_ctl[2]), "r"(static_cast<uint32_t>(1)) : "memory");
                fin_early = _atomic_old_1;
            }
        }
    }
    if (warp == 0) {
        unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, fin_early, 0);
        unsigned int fin_m = _shfl_4;
        if ((int)fin_m == gridDim.x - 1) {
            int n_slots_m = (int)n_slots_all;
            #pragma unroll 1
            for (int w_r = lane; w_r < n_slots_m; w_r += 32) {
                *(reinterpret_cast<unsigned int*>(unit_flags + w_r) + (0)) = 0;
            }
            if (lane < 4) {
                *(reinterpret_cast<unsigned int*>(merge_ctl + lane) + (0)) = 0;
            }
        }
    }
}

} // extern "C"
