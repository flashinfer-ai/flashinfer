/*
 * Copyright (c) 2023 by FlashInfer team.
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
#define SMEM_COUNTS_OFF 0
#define SMEM_COUNTS_STAGE_BYTES 4096
#define SMEM_COUNTS_STRIDE 4096
#define SMEM_OFFSETS_OFF 4096
#define SMEM_OFFSETS_STAGE_BYTES 4096
#define SMEM_OFFSETS_STRIDE 4096
#define SMEM_SCATTER_OFFSETS_OFF 8192
#define SMEM_SCATTER_OFFSETS_STAGE_BYTES 4096
#define SMEM_SCATTER_OFFSETS_STRIDE 4096
#define SMEM_CHUNK_PREFIXES_OFF 12288
#define SMEM_CHUNK_PREFIXES_STAGE_BYTES 128
#define SMEM_CHUNK_PREFIXES_STRIDE 128
#define SMEM_TOTAL 12416
#define THREADS 512

#include <math_constants.h>

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
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_cf193d0feedc02c4cb8d(int* __restrict__ topk_ids, __nv_bfloat16* __restrict__ s2b_x, float* __restrict__ s2b_qx, uint8_t* __restrict__ s2b_packed, uint8_t* __restrict__ s2b_scales, int s2b_num_tokens, int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ route_map, int* __restrict__ token_to_permuted, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ total_tiles, int total_pairs, int num_experts, int max_tiles, int top_k, int tile_n, int* __restrict__ fc2_work_counter, int fc2_pool_ctas)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

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

    // Kernel setup ops
    int* counts = reinterpret_cast<int*>(smem_raw + 0);
    const int counts_addr = smem + 0;
    int* offsets = reinterpret_cast<int*>(smem_raw + 4096);
    const int offsets_addr = smem + 4096;
    int* scatter_offsets = reinterpret_cast<int*>(smem_raw + 8192);
    const int scatter_offsets_addr = smem + 8192;
    int* chunk_prefixes = reinterpret_cast<int*>(smem_raw + 12288);
    const int chunk_prefixes_addr = smem + 12288;

    // === Task calls (dependency order) ===
    if (blockIdx.x != 0) {
        int token = blockIdx.x - 1;
        int group = tid;
        const int groups_per_row = 224;
        if (token < s2b_num_tokens && group < groups_per_row) {
            unsigned int packed_source[8];
            float source[16];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(s2b_x + (token * 3584 + group * 16) + 0);
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&packed_source[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&source[_pair * 2])[0]), "=f"((&source[_pair * 2])[1])
                    : "r"(packed_source[_pair]));
            }
            uint32_t _bf16x2_abs_0;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(packed_source[0]));
            uint32_t _bf16x2_abs_1;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(packed_source[1]));
            uint32_t _bf16x2_max_0;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
            uint32_t _bf16x2_abs_2;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(packed_source[2]));
            uint32_t _bf16x2_max_1;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_max_0), "r"(_bf16x2_abs_2));
            uint32_t _bf16x2_abs_3;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(packed_source[3]));
            uint32_t _bf16x2_max_2;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_1), "r"(_bf16x2_abs_3));
            uint32_t _bf16x2_abs_4;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(packed_source[4]));
            uint32_t _bf16x2_max_3;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(_bf16x2_max_2), "r"(_bf16x2_abs_4));
            uint32_t _bf16x2_abs_5;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(packed_source[5]));
            uint32_t _bf16x2_max_4;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(_bf16x2_max_3), "r"(_bf16x2_abs_5));
            uint32_t _bf16x2_abs_6;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(packed_source[6]));
            uint32_t _bf16x2_max_5;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(_bf16x2_max_4), "r"(_bf16x2_abs_6));
            uint32_t _bf16x2_abs_7;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(packed_source[7]));
            uint32_t _bf16x2_max_6;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(_bf16x2_max_5), "r"(_bf16x2_abs_7));
            uint16_t _bf16_max_0;
            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(_bf16x2_max_6 & 65535)), "h"((uint16_t)(_bf16x2_max_6 >> 16)));
            float _cvt_f32_bf16_0;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
            float block_max = _cvt_f32_bf16_0;
            float global_encode = s2b_qx[0];
            float scale_value = 0.0f;
            if (block_max != 0.0f) {
                scale_value = block_max * (global_encode * 0.16666666666666666f);
            }
            float _fp8_rt_0;
            uint16_t _e4m3x2_1;
            uint32_t _f16x2_1;
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(scale_value));
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
            uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
            float rounded_scale = _fp8_rt_0;
            float output_scale = 0.0f;
            if (block_max != 0.0f) {
                float _fdiv_rn_0 = __fdiv_rn(1.0f, global_encode);
                float global_decode = _fdiv_rn_0;
                float _fdiv_rn_1 = __fdiv_rn(1.0f, rounded_scale * global_decode);
                float _min_0 = fminf(_fdiv_rn_1, 3.4028234663852886e+38f);
                output_scale = _min_0;
            }
            const float2 _scale2_2 = {output_scale, output_scale};
            #pragma unroll
            for (int _ls = 0; _ls < 8; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(source)[_ls], _scale2_2);
            uint32_t _fp4_0[2];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]), "f"(source[4]), "f"(source[5]), "f"(source[6]), "f"(source[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(source[8]), "f"(source[9]), "f"(source[10]), "f"(source[11]), "f"(source[12]), "f"(source[13]), "f"(source[14]), "f"(source[15]));
            int packed_byte = token * 1792 + group * 8;
            *(reinterpret_cast<int*>(s2b_packed + packed_byte) + (0)) = _fp4_0[0];
            *(reinterpret_cast<int*>(s2b_packed + (packed_byte + 4)) + (0)) = _fp4_0[1];
            {
                unsigned short _fp8_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale_value));
                *(reinterpret_cast<unsigned char*>(s2b_scales + (token * groups_per_row + group)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
            }
        }
    } else {
        int expert_init = tid;
        counts[expert_init] = 0;
        scatter_offsets[expert_init] = 0;
        int expert_init_0 = tid + 512;
        counts[expert_init_0] = 0;
        scatter_offsets[expert_init_0] = 0;
        for (int tile_init = tid; tile_init < max_tiles; tile_init += 512) {
            tile_expert[tile_init] = 0;
            tile_mn_limit[tile_init] = tile_init * tile_n;
        }
        for (int row_init = tid; row_init < max_tiles * tile_n + 1; row_init += 512) {
            route_map[row_init] = 0;
        }
        __syncthreads();
        for (int pair_count = tid; pair_count < total_pairs; pair_count += 512) {
            int expert_count = topk_ids[pair_count];
            atomicAdd(&counts[expert_count], 1);
        }
        __syncthreads();
        int first_expert = tid;
        int second_expert = tid + 512;
        int first_tiles = 0;
        int second_tiles = 0;
        if (first_expert < num_experts) {
            int first_count = counts[first_expert];
            first_tiles = (first_count + tile_n - 1) / tile_n;
        }
        if (second_expert < num_experts) {
            int second_count = counts[second_expert];
            second_tiles = (second_count + tile_n - 1) / tile_n;
        }
        int first_inclusive = first_tiles;
        int second_inclusive = second_tiles;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 1, 32);
        int first_peer = _shfl_up_0;
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 1, 32);
        int second_peer = _shfl_up_1;
        if (lane >= 1) {
            first_inclusive = first_inclusive + first_peer;
            second_inclusive = second_inclusive + second_peer;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 2, 32);
        int first_peer_1 = _shfl_up_2;
        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 2, 32);
        int second_peer_2 = _shfl_up_3;
        if (lane >= 2) {
            first_inclusive = first_inclusive + first_peer_1;
            second_inclusive = second_inclusive + second_peer_2;
        }
        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 4, 32);
        int first_peer_3 = _shfl_up_4;
        int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 4, 32);
        int second_peer_4 = _shfl_up_5;
        if (lane >= 4) {
            first_inclusive = first_inclusive + first_peer_3;
            second_inclusive = second_inclusive + second_peer_4;
        }
        int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 8, 32);
        int first_peer_5 = _shfl_up_6;
        int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 8, 32);
        int second_peer_6 = _shfl_up_7;
        if (lane >= 8) {
            first_inclusive = first_inclusive + first_peer_5;
            second_inclusive = second_inclusive + second_peer_6;
        }
        int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, first_inclusive, 16, 32);
        int first_peer_7 = _shfl_up_8;
        int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, second_inclusive, 16, 32);
        int second_peer_8 = _shfl_up_9;
        if (lane >= 16) {
            first_inclusive = first_inclusive + first_peer_7;
            second_inclusive = second_inclusive + second_peer_8;
        }
        if (lane == 31) {
            chunk_prefixes[warp] = first_inclusive;
            chunk_prefixes[warp + 16] = second_inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            int chunk_inclusive = chunk_prefixes[lane];
            int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 1, 32);
            int chunk_peer = _shfl_up_10;
            if (lane >= 1) {
                chunk_inclusive = chunk_inclusive + chunk_peer;
            }
            int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 2, 32);
            int chunk_peer_0 = _shfl_up_11;
            if (lane >= 2) {
                chunk_inclusive = chunk_inclusive + chunk_peer_0;
            }
            int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 4, 32);
            int chunk_peer_1 = _shfl_up_12;
            if (lane >= 4) {
                chunk_inclusive = chunk_inclusive + chunk_peer_1;
            }
            int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 8, 32);
            int chunk_peer_2 = _shfl_up_13;
            if (lane >= 8) {
                chunk_inclusive = chunk_inclusive + chunk_peer_2;
            }
            int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive, 16, 32);
            int chunk_peer_3 = _shfl_up_14;
            if (lane >= 16) {
                chunk_inclusive = chunk_inclusive + chunk_peer_3;
            }
            chunk_prefixes[lane] = chunk_inclusive;
        }
        __syncthreads();
        int first_chunk_base = 0;
        if (warp > 0) {
            first_chunk_base = chunk_prefixes[warp - 1];
        }
        int second_chunk_base = chunk_prefixes[warp + 15];
        if (first_expert < num_experts) {
            int first_offset = first_chunk_base + first_inclusive - first_tiles;
            offsets[first_expert] = first_offset;
            expert_tile_offsets[first_expert] = first_offset;
        }
        if (second_expert < num_experts) {
            int second_offset = second_chunk_base + second_inclusive - second_tiles;
            offsets[second_expert] = second_offset;
            expert_tile_offsets[second_expert] = second_offset;
        }
        if (tid == 0) {
            total_tiles[0] = chunk_prefixes[31];
            fc2_work_counter[0] = fc2_pool_ctas;
        }
        __syncthreads();
        for (int pair_scatter = tid; pair_scatter < total_pairs; pair_scatter += 512) {
            int expert_scatter = topk_ids[pair_scatter];
            int _atomic_old_0 = atomicAdd(&scatter_offsets[expert_scatter], 1);
            int local_row = _atomic_old_0;
            int local_tile = local_row / tile_n;
            int tile = offsets[expert_scatter] + local_tile;
            int grouped_row = tile * tile_n + local_row % tile_n;
            route_map[grouped_row] = pair_scatter / top_k;
            token_to_permuted[pair_scatter] = grouped_row;
            if (local_row % tile_n == 0) {
                int remaining = counts[expert_scatter] - local_tile * tile_n;
                int valid = remaining;
                if (valid > tile_n) {
                    valid = tile_n;
                }
                tile_expert[tile] = expert_scatter;
                tile_mn_limit[tile] = tile * tile_n + valid;
            }
        }
        __syncthreads();
        int expert_store = tid;
        if (expert_store < num_experts) {
            expert_counts[expert_store] = counts[expert_store];
            expert_scatter_offsets[expert_store] = scatter_offsets[expert_store];
        }
        int expert_store_9 = tid + 512;
        if (expert_store_9 < num_experts) {
            expert_counts[expert_store_9] = counts[expert_store_9];
            expert_scatter_offsets[expert_store_9] = scatter_offsets[expert_store_9];
        }
    }
}

} // extern "C"
