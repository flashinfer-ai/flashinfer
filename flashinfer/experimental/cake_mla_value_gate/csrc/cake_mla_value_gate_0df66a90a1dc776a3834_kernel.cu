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
kernel_cake_mla_value_gate_0df66a90a1dc776a3834(__nv_bfloat16* latent, __nv_bfloat16* __restrict__ w_uv, __nv_bfloat16* gate, int* valid_lengths, __nv_bfloat16* out, int rows, int heads, int rows_per_request, int gate_pitch)
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
    int first_row = blockIdx.z * 8;
    int value = blockIdx.x * 32 + warp * 8 + group;
    int column = head * 128 + blockIdx.x * 32 + warp * 8 + 2 * quarter;
    int read_rows[2];
    int requests[2];
    read_rows[0] = ((first_row + group < rows) ? first_row + group : first_row);
    requests[0] = (int)((unsigned int)read_rows[0] / (unsigned int)rows_per_request);
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
    asm volatile("cp.async.commit_group;");
    unsigned int gate_words[2];
    int request_lengths[2];
    {
        request_lengths[0] = *reinterpret_cast<const int*>(valid_lengths + requests[0]);
    }
    {
        uint32_t _scalar_bits_1;
        asm volatile("ld.global.cg.b32 %0, [%1];"
            : "=r"(_scalar_bits_1) : "l"((const void*)(reinterpret_cast<unsigned int*>(gate + (long long)read_rows[0] * (long long)gate_pitch) + (column >> 1))) : "memory");
        gate_words[0] = (unsigned int)_scalar_bits_1;
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&upper[0])), "=r"(*reinterpret_cast<uint32_t*>(&upper[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&upper[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&upper[(0) + 3]))
            : "r"(staged_addr + (unsigned int)((group * 544 + 32 * c_1 + 8 * quarter) * 2)));
        unsigned int fragment[8];
        fragment[0] = upper[0];
        fragment[1] = upper[0];
        fragment[2] = upper[1];
        fragment[3] = upper[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(chains[0]), "+f"(chains[1]), "+f"(chains[2]), "+f"(chains[3])
            : "r"(fragment[0]), "r"(fragment[1]), "r"(fragment[2]), "r"(fragment[3]), "r"((weights + 4 * c_1)[0]), "r"((weights + 4 * c_1)[1]));
        fragment[4] = upper[2];
        fragment[5] = upper[2];
        fragment[6] = upper[3];
        fragment[7] = upper[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(chains[4]), "+f"(chains[(4) + 1]), "+f"(chains[(4) + 2]), "+f"(chains[(4) + 3])
            : "r"(fragment[4]), "r"(fragment[(4) + 1]), "r"(fragment[(4) + 2]), "r"(fragment[(4) + 3]), "r"((weights + (4 * c_1 + 2))[0]), "r"((weights + (4 * c_1 + 2))[1]));
    }
    float sigmoids[4];
    float denominators[4];
    float products[2];
    unsigned int words[2] = {0};
    float exp_neg = approx_exp2(mul_f32_rn_noftz(__uint_as_float(gate_words[0] << 16), -1.442695f));
    float denominator;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(denominator) : "f"(exp_neg), "f"(1.0f));
    float estimate = approx_rcp(denominator);
    float error;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(error) : "f"(denominator), "f"(estimate), "f"(-1.0f));
    float _neg_3;
    asm("neg.f32 %0, %1;" : "=f"(_neg_3) : "f"(error));
    float _fma_2;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(estimate), "f"(_neg_3), "f"(estimate));
    sigmoids[0] = _fma_2;
    denominators[0] = denominator;
    float exp_neg_0 = approx_exp2(mul_f32_rn_noftz(__uint_as_float(gate_words[0] & 4294901760u), -1.442695f));
    float denominator_0;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(denominator_0) : "f"(exp_neg_0), "f"(1.0f));
    float estimate_0 = approx_rcp(denominator_0);
    float error_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(error_0) : "f"(denominator_0), "f"(estimate_0), "f"(-1.0f));
    float _neg_5;
    asm("neg.f32 %0, %1;" : "=f"(_neg_5) : "f"(error_0));
    float _fma_4;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(estimate_0), "f"(_neg_5), "f"(estimate_0));
    sigmoids[1] = _fma_4;
    denominators[1] = denominator_0;
    bool irregular = 0;
    irregular = irregular || (__as_u32(denominators[0]) + 25165824 & 2139095040) <= 33554431 && request_lengths[0] > 0;
    irregular = irregular || (__as_u32(denominators[1]) + 25165824 & 2139095040) <= 33554431 && request_lengths[0] > 0;
    if (irregular) {
        if ((__as_u32(denominators[0]) + 25165824 & 2139095040) <= 33554431) {
            float _fdiv_6 = __fdiv_rn(1.0f, denominators[0]);
            sigmoids[0] = _fdiv_6;
        }
        if ((__as_u32(denominators[1]) + 25165824 & 2139095040) <= 33554431) {
            float _fdiv_7 = __fdiv_rn(1.0f, denominators[1]);
            sigmoids[1] = _fdiv_7;
        }
    }
    float _add_8;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(chains[0]), "f"(chains[4]));
    products[0] = mul_f32_rn_noftz((float)(__nv_bfloat16)_add_8, sigmoids[0]);
    float _add_9;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(chains[1]), "f"(chains[5]));
    products[1] = mul_f32_rn_noftz((float)(__nv_bfloat16)_add_9, sigmoids[1]);
    unsigned int packed_0 = __as_u32(__float22bfloat162_rn(make_float2(products[0], products[1])));
    words[0] = ((request_lengths[0] > 0) ? packed_0 : 0);
    if (first_row + group < rows) {
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out) + ((long long)(first_row + group) * (long long)(heads * 64) + (long long)(column >> 1))) + (0)) = words[0];
    }
}

} // extern "C"
