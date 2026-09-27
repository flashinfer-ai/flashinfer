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
#define SMEM_SCRATCH_OFF 0
#define SMEM_SCRATCH_STAGE_BYTES 128
#define SMEM_SCRATCH_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 160

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(160) void
kernel_cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ collective_out, unsigned long long* __restrict__ workspace, int rank, int tokens, float epsilon)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* scratch = reinterpret_cast<float*>(smem_raw + 0);
    const int scratch_addr = smem + 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long control_address = workspace[24];
    int* control = reinterpret_cast<int*>(control_address);
    unsigned int* completion = reinterpret_cast<unsigned int*>(control_address);
    int rotation = control[2];
    int rotation_bytes = control[3];
    int previous_clear_bytes = control[4];
    int data_rotation = rotation % 3;
    int clear_rotation = (rotation + 2) % 3;
    __nv_bfloat16* local_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + rank]);
    long long data_base = (long long)data_rotation * (long long)rotation_bytes / 2;
    long long clear_base = (long long)clear_rotation * (long long)rotation_bytes / 2;
    int total_vectors = tokens * 160;
    int vector_index = bid * 160 + tid;
    __syncthreads();
    if (tid == 0) {
        {
            unsigned int* _lca_p_0 = reinterpret_cast<unsigned int*>(completion) + (0);
            atomicAdd(_lca_p_0, 1u);
        }
    }
    float mean[16];
    #pragma unroll
    for (int element = 0; element < 16; element++) {
        mean[element] = 0.0f;
    }
    #pragma unroll 1
    for (int track = 0; track < 2; track++) {
        long long offset = ((long long)bid * 2 + (long long)track) * 2560 + (long long)(tid * 16);
        float _vec_load_0[16];
        {
            const void* _v8p_1 = (const void*)(x + offset + (0));
            uint32_t _v8_1_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 0])[0]), "=f"((&_vec_load_0[0 + 0])[1])
                : "r"(_v8_1_0[0]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 2])[0]), "=f"((&_vec_load_0[0 + 2])[1])
                : "r"(_v8_1_0[1]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 4])[0]), "=f"((&_vec_load_0[0 + 4])[1])
                : "r"(_v8_1_0[2]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 6])[0]), "=f"((&_vec_load_0[0 + 6])[1])
                : "r"(_v8_1_0[3]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 8])[0]), "=f"((&_vec_load_0[0 + 8])[1])
                : "r"(_v8_1_0[4]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 10])[0]), "=f"((&_vec_load_0[0 + 10])[1])
                : "r"(_v8_1_0[5]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 12])[0]), "=f"((&_vec_load_0[0 + 12])[1])
                : "r"(_v8_1_0[6]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + 14])[0]), "=f"((&_vec_load_0[0 + 14])[1])
                : "r"(_v8_1_0[7]));
        }
        float _vec_load_1[16];
        {
            const void* _v8p_2 = (const void*)(residual + offset + (0));
            uint32_t _v8_2_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 0])[0]), "=f"((&_vec_load_1[0 + 0])[1])
                : "r"(_v8_2_0[0]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 2])[0]), "=f"((&_vec_load_1[0 + 2])[1])
                : "r"(_v8_2_0[1]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 4])[0]), "=f"((&_vec_load_1[0 + 4])[1])
                : "r"(_v8_2_0[2]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 6])[0]), "=f"((&_vec_load_1[0 + 6])[1])
                : "r"(_v8_2_0[3]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 8])[0]), "=f"((&_vec_load_1[0 + 8])[1])
                : "r"(_v8_2_0[4]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 10])[0]), "=f"((&_vec_load_1[0 + 10])[1])
                : "r"(_v8_2_0[5]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 12])[0]), "=f"((&_vec_load_1[0 + 12])[1])
                : "r"(_v8_2_0[6]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + 14])[0]), "=f"((&_vec_load_1[0 + 14])[1])
                : "r"(_v8_2_0[7]));
        }
        float propagated[16];
        float square_sum = 0.0f;
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_vec_load_0[element_1] + _vec_load_1[element_1]);
            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
            propagated[element_1] = _cvt_f32_0;
            float _fma_0 = __fmaf_rn(propagated[element_1], propagated[element_1], square_sum);
            square_sum = _fma_0;
        }
        uint32_t propagated_bf16[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(propagated[_lp*2 + 0], propagated[_lp*2+1 + 0]));
            propagated_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(residual_out + offset), "r"((propagated_bf16)[0]), "r"((propagated_bf16)[1]), "r"((propagated_bf16)[2]), "r"((propagated_bf16)[3]), "r"((propagated_bf16)[4]), "r"((propagated_bf16)[5]), "r"((propagated_bf16)[6]), "r"((propagated_bf16)[7]) : "memory");
        float _warp_reduce_0 = square_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        if (lane == 0) {
            scratch[warp] = _warp_reduce_0;
        }
        __syncthreads();
        if (warp == 0) {
            float partial = ((lane < 5) ? scratch[lane] : 0.0f);
            float _warp_reduce_1 = partial;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            if (lane == 0) {
                scratch[0] = _warp_reduce_1;
            }
        }
        __syncthreads();
        float result = scratch[0];
        float _rsqrt_0 = rsqrtf(result / 2560.0f + epsilon);
        float _vec_load_2[16];
        {
            const void* _v8p_3 = (const void*)(weight + (track * 2560 + tid * 16) + (0));
            uint32_t _v8_3_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_3_0[0]), "=r"(_v8_3_0[1]), "=r"(_v8_3_0[2]), "=r"(_v8_3_0[3]), "=r"(_v8_3_0[4]), "=r"(_v8_3_0[5]), "=r"(_v8_3_0[6]), "=r"(_v8_3_0[7]) : "l"((const char*)_v8p_3 + 0) : "memory");
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 0])[0]), "=f"((&_vec_load_2[0 + 0])[1])
                : "r"(_v8_3_0[0]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 2])[0]), "=f"((&_vec_load_2[0 + 2])[1])
                : "r"(_v8_3_0[1]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 4])[0]), "=f"((&_vec_load_2[0 + 4])[1])
                : "r"(_v8_3_0[2]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 6])[0]), "=f"((&_vec_load_2[0 + 6])[1])
                : "r"(_v8_3_0[3]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 8])[0]), "=f"((&_vec_load_2[0 + 8])[1])
                : "r"(_v8_3_0[4]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 10])[0]), "=f"((&_vec_load_2[0 + 10])[1])
                : "r"(_v8_3_0[5]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 12])[0]), "=f"((&_vec_load_2[0 + 12])[1])
                : "r"(_v8_3_0[6]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0 + 14])[0]), "=f"((&_vec_load_2[0 + 14])[1])
                : "r"(_v8_3_0[7]));
        }
        float normalized[16];
        #pragma unroll
        for (int element_2 = 0; element_2 < 16; element_2++) {
            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(propagated[element_2] * _rsqrt_0 * _vec_load_2[element_2]);
            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
            normalized[element_2] = _cvt_f32_1;
            float _fma_1 = __fmaf_rn(normalized[element_2], 0.5f, mean[element_2]);
            mean[element_2] = _fma_1;
        }
        uint32_t normalized_bf16[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
            normalized_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(norm_out + offset), "r"((normalized_bf16)[0]), "r"((normalized_bf16)[1]), "r"((normalized_bf16)[2]), "r"((normalized_bf16)[3]), "r"((normalized_bf16)[4]), "r"((normalized_bf16)[5]), "r"((normalized_bf16)[6]), "r"((normalized_bf16)[7]) : "memory");
    }
    float contribution[16];
    #pragma unroll
    for (int element_3 = 0; element_3 < 16; element_3++) {
        __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(mean[element_3]);
        float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
        contribution[element_3] = ((_cvt_f32_2 == 0.0f) ? 0.0f : _cvt_f32_2);
    }
    uint32_t contribution_bf16[8];
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contribution[_lp*2 + 0], contribution[_lp*2+1 + 0]));
        contribution_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    int owner = bid % 8;
    __nv_bfloat16* owner_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + owner]);
    long long publish_offset = data_base + ((long long)rank * (long long)total_vectors + (long long)vector_index) * 16;
    asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(owner_payload + publish_offset), "r"((contribution_bf16)[0]), "r"((contribution_bf16)[1]), "r"((contribution_bf16)[2]), "r"((contribution_bf16)[3]), "r"((contribution_bf16)[4]), "r"((contribution_bf16)[5]), "r"((contribution_bf16)[6]), "r"((contribution_bf16)[7]) : "memory");
    unsigned int sentinel[8];
    #pragma unroll
    for (int word = 0; word < 8; word++) {
        sentinel[word] = 2147516416;
    }
    #pragma unroll 1
    for (int clear_index = vector_index; clear_index < previous_clear_bytes / 32; clear_index += num_bids * 160) {
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(local_payload + (clear_base + (long long)clear_index * 16)), "r"((sentinel)[0]), "r"((sentinel)[1]), "r"((sentinel)[2]), "r"((sentinel)[3]), "r"((sentinel)[4]), "r"((sentinel)[5]), "r"((sentinel)[6]), "r"((sentinel)[7]) : "memory");
    }
    long long owner_offset = data_base + ((long long)owner * (long long)total_vectors + (long long)vector_index) * 16;
    if (rank == owner) {
        uint32_t _sysv_poll256_group_0[64];
        do {
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[0]), "=r"(_sysv_poll256_group_0[1]), "=r"(_sysv_poll256_group_0[2]), "=r"(_sysv_poll256_group_0[3]), "=r"(_sysv_poll256_group_0[4]), "=r"(_sysv_poll256_group_0[5]), "=r"(_sysv_poll256_group_0[6]), "=r"(_sysv_poll256_group_0[7]) : "l"(local_payload + (data_base + (long long)(vector_index * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[8]), "=r"(_sysv_poll256_group_0[9]), "=r"(_sysv_poll256_group_0[10]), "=r"(_sysv_poll256_group_0[11]), "=r"(_sysv_poll256_group_0[12]), "=r"(_sysv_poll256_group_0[13]), "=r"(_sysv_poll256_group_0[14]), "=r"(_sysv_poll256_group_0[15]) : "l"(local_payload + (data_base + (long long)((total_vectors + vector_index) * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[16]), "=r"(_sysv_poll256_group_0[17]), "=r"(_sysv_poll256_group_0[18]), "=r"(_sysv_poll256_group_0[19]), "=r"(_sysv_poll256_group_0[20]), "=r"(_sysv_poll256_group_0[21]), "=r"(_sysv_poll256_group_0[22]), "=r"(_sysv_poll256_group_0[23]) : "l"(local_payload + (data_base + (long long)((2 * total_vectors + vector_index) * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[24]), "=r"(_sysv_poll256_group_0[25]), "=r"(_sysv_poll256_group_0[26]), "=r"(_sysv_poll256_group_0[27]), "=r"(_sysv_poll256_group_0[28]), "=r"(_sysv_poll256_group_0[29]), "=r"(_sysv_poll256_group_0[30]), "=r"(_sysv_poll256_group_0[31]) : "l"(local_payload + (data_base + (long long)((3 * total_vectors + vector_index) * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[32]), "=r"(_sysv_poll256_group_0[33]), "=r"(_sysv_poll256_group_0[34]), "=r"(_sysv_poll256_group_0[35]), "=r"(_sysv_poll256_group_0[36]), "=r"(_sysv_poll256_group_0[37]), "=r"(_sysv_poll256_group_0[38]), "=r"(_sysv_poll256_group_0[39]) : "l"(local_payload + (data_base + (long long)((4 * total_vectors + vector_index) * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[40]), "=r"(_sysv_poll256_group_0[41]), "=r"(_sysv_poll256_group_0[42]), "=r"(_sysv_poll256_group_0[43]), "=r"(_sysv_poll256_group_0[44]), "=r"(_sysv_poll256_group_0[45]), "=r"(_sysv_poll256_group_0[46]), "=r"(_sysv_poll256_group_0[47]) : "l"(local_payload + (data_base + (long long)((5 * total_vectors + vector_index) * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[48]), "=r"(_sysv_poll256_group_0[49]), "=r"(_sysv_poll256_group_0[50]), "=r"(_sysv_poll256_group_0[51]), "=r"(_sysv_poll256_group_0[52]), "=r"(_sysv_poll256_group_0[53]), "=r"(_sysv_poll256_group_0[54]), "=r"(_sysv_poll256_group_0[55]) : "l"(local_payload + (data_base + (long long)((6 * total_vectors + vector_index) * 16))) : "memory");
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[56]), "=r"(_sysv_poll256_group_0[57]), "=r"(_sysv_poll256_group_0[58]), "=r"(_sysv_poll256_group_0[59]), "=r"(_sysv_poll256_group_0[60]), "=r"(_sysv_poll256_group_0[61]), "=r"(_sysv_poll256_group_0[62]), "=r"(_sysv_poll256_group_0[63]) : "l"(local_payload + (data_base + (long long)((7 * total_vectors + vector_index) * 16))) : "memory");
        } while ((((_sysv_poll256_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[1] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[3] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[5] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[7] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[9] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[11] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[13] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[15] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[17] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[19] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[21] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[23] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[25] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[27] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[29] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[31] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[33] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[35] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[37] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[39] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[41] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[43] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[45] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[47] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[49] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[51] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[53] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[55] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[57] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[59] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[61] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[63] >> 16) & 0xffffu) == 0x8000u));
        float _sysv_poll256_group_0_f32[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll256_group_0_f32[_pair * 2])[0]), "=f"((&_sysv_poll256_group_0_f32[_pair * 2])[1])
                : "r"(_sysv_poll256_group_0[_pair]));
        }
        #pragma unroll
        for (int peer = 1; peer < 8; peer++) {
            float _sysv_poll256_group_0_f32_0[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_sysv_poll256_group_0_f32_0[_pair * 2])[0]), "=f"((&_sysv_poll256_group_0_f32_0[_pair * 2])[1])
                    : "r"((_sysv_poll256_group_0 + peer * 8)[_pair]));
            }
            #pragma unroll
            for (int element_4 = 0; element_4 < 16; element_4++) {
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_sysv_poll256_group_0_f32[element_4] + _sysv_poll256_group_0_f32_0[element_4]);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                _sysv_poll256_group_0_f32[element_4] = _cvt_f32_3;
            }
        }
        uint32_t _sysv_poll256_group_0_f32_bf16[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_sysv_poll256_group_0_f32[_lp*2 + 0], _sysv_poll256_group_0_f32[_lp*2+1 + 0]));
            _sysv_poll256_group_0_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int peer_1 = 0; peer_1 < 8; peer_1++) {
            __nv_bfloat16* peer_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + peer_1]);
            asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload + owner_offset), "r"((_sysv_poll256_group_0_f32_bf16)[0]), "r"((_sysv_poll256_group_0_f32_bf16)[1]), "r"((_sysv_poll256_group_0_f32_bf16)[2]), "r"((_sysv_poll256_group_0_f32_bf16)[3]), "r"((_sysv_poll256_group_0_f32_bf16)[4]), "r"((_sysv_poll256_group_0_f32_bf16)[5]), "r"((_sysv_poll256_group_0_f32_bf16)[6]), "r"((_sysv_poll256_group_0_f32_bf16)[7]) : "memory");
        }
    }
    uint32_t _sysv_poll256_group_1[8];
    do {
        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_1[0]), "=r"(_sysv_poll256_group_1[1]), "=r"(_sysv_poll256_group_1[2]), "=r"(_sysv_poll256_group_1[3]), "=r"(_sysv_poll256_group_1[4]), "=r"(_sysv_poll256_group_1[5]), "=r"(_sysv_poll256_group_1[6]), "=r"(_sysv_poll256_group_1[7]) : "l"(local_payload + owner_offset) : "memory");
    } while ((((_sysv_poll256_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[1] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[3] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[5] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[7] >> 16) & 0xffffu) == 0x8000u));
    float _sysv_poll256_group_1_f32[16];
    #pragma unroll
    for (int _pair = 0; _pair < 8; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_sysv_poll256_group_1_f32[_pair * 2])[0]), "=f"((&_sysv_poll256_group_1_f32[_pair * 2])[1])
            : "r"(_sysv_poll256_group_1[_pair]));
    }
    uint32_t _sysv_poll256_group_1_f32_bf16[8];
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_sysv_poll256_group_1_f32[_lp*2 + 0], _sysv_poll256_group_1_f32[_lp*2+1 + 0]));
        _sysv_poll256_group_1_f32_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + ((long long)vector_index * 16)), "r"((_sysv_poll256_group_1_f32_bf16)[0]), "r"((_sysv_poll256_group_1_f32_bf16)[1]), "r"((_sysv_poll256_group_1_f32_bf16)[2]), "r"((_sysv_poll256_group_1_f32_bf16)[3]), "r"((_sysv_poll256_group_1_f32_bf16)[4]), "r"((_sysv_poll256_group_1_f32_bf16)[5]), "r"((_sysv_poll256_group_1_f32_bf16)[6]), "r"((_sysv_poll256_group_1_f32_bf16)[7]) : "memory");
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _lcv_p_4 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_lcv_p_4 != static_cast<int>(num_bids)) {}
                *reinterpret_cast<int*>(control + 2) = static_cast<int>((rotation + 1) % 3);
                *reinterpret_cast<int*>(control + 4) = static_cast<int>(tokens * 2560 * 8 * 2);
                *(reinterpret_cast<int*>(completion) + (0)) = 0;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
