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
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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
#define SMEM_SMEM_RSTD_OFF 0
#define SMEM_SMEM_RSTD_STAGE_BYTES 4
#define SMEM_SMEM_RSTD_STRIDE 4
#define SMEM_SMEM_INDEX_OFF 4
#define SMEM_SMEM_INDEX_STAGE_BYTES 4
#define SMEM_SMEM_INDEX_STRIDE 4
#define SMEM_SMEM_PARTIAL_OFF 8
#define SMEM_SMEM_PARTIAL_STAGE_BYTES 84
#define SMEM_SMEM_PARTIAL_STRIDE 84
#define SMEM_TOTAL 128
#define THREADS 672
#define M 7368
#define HIDDEN 5376
#define SCALE_COLS 168

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

__global__ __launch_bounds__(672) void
kernel_cake_minimax_h3_mxfp8_pre_attention_sm100a_d6013cb5364e1777b1bd(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, uint8_t* __restrict__ activation_q, uint8_t* __restrict__ activation_sf, float eps)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_rstd = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_rstd_addr = smem + 0;
    int* smem_index = reinterpret_cast<int*>(smem_raw + 4);
    const int smem_index_addr = smem + 4;
    float* smem_partial = reinterpret_cast<float*>(smem_raw + 8);
    const int smem_partial_addr = smem + 8;

    // === Task calls (dependency order) ===
    int row_slot = 0;
    int warp_in_row = warp;
    int thread_in_row = (unsigned int)(warp_in_row * 32) + lane;
    int row = bid;
    float sum_sq = 0.0f;
    unsigned int x_carrier[4];
    if (row < M) {
        unsigned long long row_base = (unsigned long long)row * (unsigned long long)HIDDEN;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row_base + (unsigned long long)(thread_in_row * 8));
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_carrier[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
        float x_carrier_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&x_carrier_f32[_pair * 2])[0]), "=f"((&x_carrier_f32[_pair * 2])[1])
                : "r"(x_carrier[_pair]));
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float _fma_0 = __fmaf_rn(x_carrier_f32[j], x_carrier_f32[j], sum_sq);
            sum_sq = _fma_0;
        }
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
    sum_sq += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
    sum_sq += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
    sum_sq += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
    sum_sq += _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
    sum_sq += _shfl_xor_4;
    if (lane == 0) {
        smem_partial[warp_in_row] = sum_sq;
    }
    asm volatile("barrier.sync 1, 672;" ::: "memory");
    sum_sq = 0.0f;
    if (warp_in_row == 0) {
        if (lane < 21) {
            sum_sq = smem_partial[lane];
        }
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
        sum_sq += _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
        sum_sq += _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
        sum_sq += _shfl_xor_7;
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
        sum_sq += _shfl_xor_8;
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
        sum_sq += _shfl_xor_9;
        if (lane == 0) {
            if (row < M) {
                float _rsqrt_0;
                asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(_rsqrt_0) : "f"(sum_sq / (float)HIDDEN + eps));
                smem_rstd[row_slot] = _rsqrt_0;
                smem_index[row_slot] = adaln_index[row];
            } else {
                smem_rstd[row_slot] = 0.0f;
                smem_index[row_slot] = 0;
            }
        }
    }
    asm volatile("barrier.sync 1, 672;" ::: "memory");
    int scale_col = thread_in_row / 4;
    int lane_in_block = thread_in_row % 4;
    if (row < M && scale_col < SCALE_COLS) {
        int block_col = scale_col * 32;
        int col = block_col + lane_in_block * 8;
        int table_row = smem_index[row_slot];
        float values[8];
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            values[j_1] = 0.0f;
        }
        if (table_row >= 0 && table_row < 9) {
            unsigned long long row_base_1 = (unsigned long long)row * (unsigned long long)HIDDEN;
            unsigned long long table_base = (unsigned long long)table_row * (unsigned long long)HIDDEN;
            float x_carrier_f32_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&x_carrier_f32_1[_pair * 2])[0]), "=f"((&x_carrier_f32_1[_pair * 2])[1])
                    : "r"(x_carrier[_pair]));
            }
            float _vec_load_0[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x_norm_weight + col + 0);
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
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            float _vec_load_1[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(adaln_scale + (table_base + (unsigned long long)col) + 0);
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
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_2[_pair]));
                    }
                }
            }
            float _vec_load_2[8];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(adaln_shift + (table_base + (unsigned long long)col) + 0);
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
                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_3[_pair]));
                    }
                }
            }
            float rstd = smem_rstd[row_slot];
            const float2 _scale2_4 = {rstd, rstd};
            #pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(x_carrier_f32_1)[_ls], _scale2_4);
            #pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(x_carrier_f32_1)[_ls], reinterpret_cast<const float2*>(_vec_load_0)[_ls]);
            uint32_t x_carrier_f32_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(x_carrier_f32_1[_lp*2 + 0], x_carrier_f32_1[_lp*2+1 + 0]));
                x_carrier_f32_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float x_carrier_f32_bf16_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&x_carrier_f32_bf16_f32[_pair * 2])[0]), "=f"((&x_carrier_f32_bf16_f32[_pair * 2])[1])
                    : "r"(x_carrier_f32_bf16[_pair]));
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
                float2 _f2_0 = make_float2(x_carrier_f32_bf16_f32[pair_offset], x_carrier_f32_bf16_f32[pair_offset + 1]);
                float2 normalized_pair = _f2_0;
                float2 _f2_1 = make_float2(_vec_load_1_bf16_f32[pair_offset], _vec_load_1_bf16_f32[pair_offset + 1]);
                float2 scale_pair = _f2_1;
                float2 _f2_2 = make_float2(_vec_load_2[pair_offset], _vec_load_2[pair_offset + 1]);
                float2 shift_pair = _f2_2;
                float2 adaln_pair = fma_f32x2_rn_ftz(normalized_pair, scale_pair, shift_pair);
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(adaln_pair.x);
                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                values[pair_offset] = _cvt_f32_0;
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(adaln_pair.y);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                values[pair_offset + 1] = _cvt_f32_1;
            }
        }
        float absolute[8];
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
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
        float _max_0 = max_noftz(amax, _shfl_xor_10);
        amax = _max_0;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
        float _max_1 = max_noftz(amax, _shfl_xor_11);
        amax = _max_1;
        float _rcp_0 = approx_rcp(448.0f);
        float block_scale = amax * _rcp_0;
        int scale_bits;
        scale_bits = reinterpret_cast<int*>(&block_scale)[0];
        int exponent = scale_bits >> 23 & 255;
        int mantissa = scale_bits & 8388607;
        int round_up = ((mantissa != 0 && (exponent != 0 || mantissa > 4194304)) ? 1 : 0);
        int scale_code = exponent + round_up;
        scale_code = ((scale_code < 255) ? scale_code : 254);
        scale_code = ((block_scale > 0.0f) ? scale_code : 0);
        int scale_value_bits = scale_code << 23;
        float scale_value;
        scale_value = reinterpret_cast<float*>(&scale_value_bits)[0];
        float inverse_scale;
        inverse_scale = 0.0f;
        if (scale_value != 0.0f) {
            float _rcp_1 = approx_rcp(scale_value);
            inverse_scale = _rcp_1;
        }
        float quant_values[8];
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            quant_values[j_3] = values[j_3];
        }
        const float2 _scale2_6 = {inverse_scale, inverse_scale};
        #pragma unroll
        for (int _ls = 0; _ls < 4; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(quant_values)[_ls], _scale2_6);
        {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0]) : "f"(quant_values[0 + 0]), "f"(quant_values[0 + 1]), "f"(quant_values[0 + 2]), "f"(quant_values[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1]) : "f"(quant_values[0 + 4]), "f"(quant_values[0 + 5]), "f"(quant_values[0 + 6]), "f"(quant_values[0 + 7]));
            *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(activation_q + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col)) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
        }
        unsigned long long scale_offset = (unsigned long long)(scale_col % 4 + scale_col / 4 * 512 + row % 32 * 16 + row % 128 / 32 * 4) + (unsigned long long)(row / 128) * (unsigned long long)(128 * SCALE_COLS);
        if (lane_in_block == 0) {
            *(reinterpret_cast<unsigned char*>(activation_sf + scale_offset) + (0)) = (unsigned char)(scale_code);
        }
    }
}

} // extern "C"
