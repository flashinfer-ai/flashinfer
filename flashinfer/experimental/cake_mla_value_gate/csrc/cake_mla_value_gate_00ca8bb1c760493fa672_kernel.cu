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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mla_value_gate_00ca8bb1c760493fa672(__nv_bfloat16* latent, __nv_bfloat16* __restrict__ w_uv, __nv_bfloat16* gate, int* valid_lengths, __nv_bfloat16* out, int rows, int heads, int rows_per_request, int gate_pitch)
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

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int staged_addr = smem + 0;

    // === Task calls (declaration order) ===
    int group = lane >> 2;
    int quarter = lane & 3;
    int head = blockIdx.y;
    int first_row = blockIdx.z * 16;
    int value = blockIdx.x * 32 + warp * 8 + group;
    int column = head * 128 + blockIdx.x * 32 + warp * 8 + 2 * quarter;
    int read_rows[2];
    int requests[2];
    read_rows[0] = ((first_row + group < rows) ? first_row + group : first_row);
    requests[0] = (int)((unsigned int)read_rows[0] / (unsigned int)rows_per_request);
    read_rows[1] = ((first_row + group + 8 < rows) ? first_row + group + 8 : first_row);
    requests[1] = (int)((unsigned int)read_rows[1] / (unsigned int)rows_per_request);
    unsigned int weights[64];
    #pragma unroll
    for (int c = 0; c < 16; c++) {
        {
            uint32_t _uv4_0_0;
            uint32_t _uv4_0_1;
            uint32_t _uv4_0_2;
            uint32_t _uv4_0_3;
            asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(w_uv + (long long)head * 65536) + (value * 512 + 32 * c + 8 * quarter >> 1))) : "memory");
            weights[4 * c + 0] = _uv4_0_0;
            weights[4 * c + 1] = _uv4_0_1;
            weights[4 * c + 2] = _uv4_0_2;
            weights[4 * c + 3] = _uv4_0_3;
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int stage_row = tid >> 6;
    int stage_col = 8 * (tid & 63);
    bool live_0 = first_row + stage_row < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)((stage_row * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_0) ? (long long)stage_row * (long long)heads * 512 : 0))), "r"(((live_0) ? 16 : 0)));
    bool live_1 = first_row + stage_row + 2 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 2) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_1) ? (long long)(stage_row + 2) * (long long)heads * 512 : 0))), "r"(((live_1) ? 16 : 0)));
    bool live_2 = first_row + stage_row + 4 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 4) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_2) ? (long long)(stage_row + 4) * (long long)heads * 512 : 0))), "r"(((live_2) ? 16 : 0)));
    bool live_3 = first_row + stage_row + 6 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 6) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_3) ? (long long)(stage_row + 6) * (long long)heads * 512 : 0))), "r"(((live_3) ? 16 : 0)));
    bool live_4 = first_row + stage_row + 8 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 8) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_4) ? (long long)(stage_row + 8) * (long long)heads * 512 : 0))), "r"(((live_4) ? 16 : 0)));
    bool live_5 = first_row + stage_row + 10 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 10) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_5) ? (long long)(stage_row + 10) * (long long)heads * 512 : 0))), "r"(((live_5) ? 16 : 0)));
    bool live_6 = first_row + stage_row + 12 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 12) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_6) ? (long long)(stage_row + 12) * (long long)heads * 512 : 0))), "r"(((live_6) ? 16 : 0)));
    bool live_7 = first_row + stage_row + 14 < rows;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(staged_addr + (unsigned int)(((stage_row + 14) * 544 + stage_col) * 2)), "l"(latent + (((long long)first_row * (long long)heads + (long long)head) * 512 + (long long)stage_col + ((live_7) ? (long long)(stage_row + 14) * (long long)heads * 512 : 0))), "r"(((live_7) ? 16 : 0)));
    asm volatile("cp.async.commit_group;");
    unsigned int gate_words[2];
    int request_lengths[2];
    {
        request_lengths[0] = *reinterpret_cast<const int*>(valid_lengths + requests[0]);
    }
    {
        request_lengths[1] = *reinterpret_cast<const int*>(valid_lengths + requests[1]);
    }
    {
        uint32_t _scalar_bits_1;
        asm volatile("ld.global.cg.b32 %0, [%1];"
            : "=r"(_scalar_bits_1) : "l"((const void*)(reinterpret_cast<unsigned int*>(gate + (long long)read_rows[0] * (long long)gate_pitch) + (column >> 1))) : "memory");
        gate_words[0] = (unsigned int)_scalar_bits_1;
    }
    {
        uint32_t _scalar_bits_2;
        asm volatile("ld.global.cg.b32 %0, [%1];"
            : "=r"(_scalar_bits_2) : "l"((const void*)(reinterpret_cast<unsigned int*>(gate + (long long)read_rows[1] * (long long)gate_pitch) + (column >> 1))) : "memory");
        gate_words[1] = (unsigned int)_scalar_bits_2;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    float chains[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        chains[k] = 0.0f;
    }
    #pragma unroll
    for (int c_1 = 0; c_1 < 16; c_1++) {
        unsigned int upper[4];
        unsigned int lower[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&upper[0])), "=r"(*reinterpret_cast<uint32_t*>(&upper[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&upper[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&upper[(0) + 3]))
            : "r"(staged_addr + (unsigned int)((group * 544 + 32 * c_1 + 8 * quarter) * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lower[0])), "=r"(*reinterpret_cast<uint32_t*>(&lower[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lower[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lower[(0) + 3]))
            : "r"(staged_addr + (unsigned int)(((group + 8) * 544 + 32 * c_1 + 8 * quarter) * 2)));
        unsigned int fragment[8];
        fragment[0] = upper[0];
        fragment[1] = lower[0];
        fragment[2] = upper[1];
        fragment[3] = lower[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(chains[0]), "+f"(chains[1]), "+f"(chains[2]), "+f"(chains[3])
            : "r"(fragment[0]), "r"(fragment[1]), "r"(fragment[2]), "r"(fragment[3]), "r"((weights + 4 * c_1)[0]), "r"((weights + 4 * c_1)[1]));
        fragment[4] = upper[2];
        fragment[5] = lower[2];
        fragment[6] = upper[3];
        fragment[7] = lower[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(chains[4]), "+f"(chains[(4) + 1]), "+f"(chains[(4) + 2]), "+f"(chains[(4) + 3])
            : "r"(fragment[4]), "r"(fragment[(4) + 1]), "r"(fragment[(4) + 2]), "r"(fragment[(4) + 3]), "r"((weights + (4 * c_1 + 2))[0]), "r"((weights + (4 * c_1 + 2))[1]));
    }
    float products[2];
    if (first_row + group < rows) {
        unsigned int word_0 = 0;
        if (request_lengths[0] > 0) {
            float _add_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(chains[0]), "f"(chains[4]));
            float rounded = (float)(__nv_bfloat16)_add_3;
            float exp_neg = approx_exp2(mul_f32_rn_noftz(__uint_as_float(gate_words[0] << 16), -1.442695f));
            float denominator;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(denominator) : "f"(exp_neg), "f"(1.0f));
            float estimate = approx_rcp(denominator);
            float error;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(error) : "f"(denominator), "f"(estimate), "f"(-1.0f));
            float _neg_4;
            asm("neg.f32 %0, %1;" : "=f"(_neg_4) : "f"(error));
            float sigmoid;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sigmoid) : "f"(estimate), "f"(_neg_4), "f"(estimate));
            unsigned int exponent = __as_u32(denominator) + 25165824 & 2139095040;
            if (exponent <= 33554431) {
                float _fdiv_5 = __fdiv_rn(1.0f, denominator);
                sigmoid = _fdiv_5;
            }
            products[0] = mul_f32_rn_noftz(rounded, sigmoid);
            float _add_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(chains[1]), "f"(chains[5]));
            float rounded_0 = (float)(__nv_bfloat16)_add_6;
            float exp_neg_0 = approx_exp2(mul_f32_rn_noftz(__uint_as_float(gate_words[0] & 4294901760u), -1.442695f));
            float denominator_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(denominator_0) : "f"(exp_neg_0), "f"(1.0f));
            float estimate_0 = approx_rcp(denominator_0);
            float error_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(error_0) : "f"(denominator_0), "f"(estimate_0), "f"(-1.0f));
            float _neg_7;
            asm("neg.f32 %0, %1;" : "=f"(_neg_7) : "f"(error_0));
            float sigmoid_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sigmoid_0) : "f"(estimate_0), "f"(_neg_7), "f"(estimate_0));
            unsigned int exponent_0 = __as_u32(denominator_0) + 25165824 & 2139095040;
            if (exponent_0 <= 33554431) {
                float _fdiv_8 = __fdiv_rn(1.0f, denominator_0);
                sigmoid_0 = _fdiv_8;
            }
            products[1] = mul_f32_rn_noftz(rounded_0, sigmoid_0);
            word_0 = __as_u32(__float22bfloat162_rn(make_float2(products[0], products[1])));
        }
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out) + ((long long)(first_row + group) * (long long)(heads * 64) + (long long)(column >> 1))) + (0)) = word_0;
    }
    if (first_row + group + 8 < rows) {
        unsigned int word_1 = 0;
        if (request_lengths[1] > 0) {
            float _add_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(chains[2]), "f"(chains[6]));
            float rounded_1 = (float)(__nv_bfloat16)_add_9;
            float exp_neg_1 = approx_exp2(mul_f32_rn_noftz(__uint_as_float(gate_words[1] << 16), -1.442695f));
            float denominator_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(denominator_1) : "f"(exp_neg_1), "f"(1.0f));
            float estimate_1 = approx_rcp(denominator_1);
            float error_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(error_1) : "f"(denominator_1), "f"(estimate_1), "f"(-1.0f));
            float _neg_10;
            asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(error_1));
            float sigmoid_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sigmoid_1) : "f"(estimate_1), "f"(_neg_10), "f"(estimate_1));
            unsigned int exponent_1 = __as_u32(denominator_1) + 25165824 & 2139095040;
            if (exponent_1 <= 33554431) {
                float _fdiv_11 = __fdiv_rn(1.0f, denominator_1);
                sigmoid_1 = _fdiv_11;
            }
            products[0] = mul_f32_rn_noftz(rounded_1, sigmoid_1);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(chains[3]), "f"(chains[7]));
            float rounded_1_1 = (float)(__nv_bfloat16)_add_12;
            float exp_neg_1_1 = approx_exp2(mul_f32_rn_noftz(__uint_as_float(gate_words[1] & 4294901760u), -1.442695f));
            float denominator_1_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(denominator_1_1) : "f"(exp_neg_1_1), "f"(1.0f));
            float estimate_1_1 = approx_rcp(denominator_1_1);
            float error_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(error_1_1) : "f"(denominator_1_1), "f"(estimate_1_1), "f"(-1.0f));
            float _neg_13;
            asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(error_1_1));
            float sigmoid_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sigmoid_1_1) : "f"(estimate_1_1), "f"(_neg_13), "f"(estimate_1_1));
            unsigned int exponent_1_1 = __as_u32(denominator_1_1) + 25165824 & 2139095040;
            if (exponent_1_1 <= 33554431) {
                float _fdiv_14 = __fdiv_rn(1.0f, denominator_1_1);
                sigmoid_1_1 = _fdiv_14;
            }
            products[1] = mul_f32_rn_noftz(rounded_1_1, sigmoid_1_1);
            word_1 = __as_u32(__float22bfloat162_rn(make_float2(products[0], products[1])));
        }
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out) + ((long long)(first_row + group + 8) * (long long)(heads * 64) + (long long)(column >> 1))) + (0)) = word_1;
    }
}

} // extern "C"
