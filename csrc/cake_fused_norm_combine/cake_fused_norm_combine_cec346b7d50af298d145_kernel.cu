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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SCRATCH_OFF 0
#define SMEM_SCRATCH_STAGE_BYTES 256
#define SMEM_SCRATCH_STRIDE 256
#define SMEM_EXCHANGE_OFF 256
#define SMEM_EXCHANGE_STAGE_BYTES 10240
#define SMEM_EXCHANGE_STRIDE 10240
#define SMEM_TOTAL 10496
#define THREADS 320

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_fused_norm_combine_cec346b7d50af298d145(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ collective_out, unsigned long long* __restrict__ workspace, int rank, int tokens, float epsilon)
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

    // Kernel setup ops
    float* scratch = reinterpret_cast<float*>(smem_raw + 0);
    const int scratch_addr = smem + 0;
    float* exchange = reinterpret_cast<float*>(smem_raw + 256);
    const int exchange_addr = smem + 256;

    // === Task calls (declaration order) ===
    if (tid < 80) {
        long long row = (long long)(bid / 8) * 5120 + (long long)(tid * 64);
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + row)));
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(residual + row)));
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(weight + (tid * 64))));
    }
    if (tid == 0) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(workspace + 16)));
        uint64_t _sysv_u32_0;
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(_sysv_u32_0) : "l"(workspace + 24) : "memory");
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<int*>(_sysv_u32_0))));
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int token = bid / 8;
    int target = bid % 8;
    int track = tid / 160;
    int local_tid = tid % 160;
    long long offset = ((long long)token * 2 + (long long)track) * 2560 + (long long)(local_tid * 16);
    float _vec_load_0[16];
    {
        const void* _v8p_0 = (const void*)(x + offset + (0));
        uint32_t _v8_0_0[8];
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
        (&_vec_load_0[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
        (&_vec_load_0[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
        (&_vec_load_0[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
        (&_vec_load_0[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
        (&_vec_load_0[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
        (&_vec_load_0[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
        (&_vec_load_0[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
        (&_vec_load_0[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
        (&_vec_load_0[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
        (&_vec_load_0[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
        (&_vec_load_0[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
        (&_vec_load_0[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
        (&_vec_load_0[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
        (&_vec_load_0[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
        (&_vec_load_0[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
        (&_vec_load_0[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
    }
    float _vec_load_1[16];
    {
        const void* _v8p_1 = (const void*)(residual + offset + (0));
        uint32_t _v8_1_0[8];
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
        (&_vec_load_1[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) << 16);
        (&_vec_load_1[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) & 0xffff0000u);
        (&_vec_load_1[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) << 16);
        (&_vec_load_1[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) & 0xffff0000u);
        (&_vec_load_1[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) << 16);
        (&_vec_load_1[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) & 0xffff0000u);
        (&_vec_load_1[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) << 16);
        (&_vec_load_1[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) & 0xffff0000u);
        (&_vec_load_1[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) << 16);
        (&_vec_load_1[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) & 0xffff0000u);
        (&_vec_load_1[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) << 16);
        (&_vec_load_1[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) & 0xffff0000u);
        (&_vec_load_1[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) << 16);
        (&_vec_load_1[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) & 0xffff0000u);
        (&_vec_load_1[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) << 16);
        (&_vec_load_1[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) & 0xffff0000u);
    }
    float _vec_load_2[16];
    {
        const void* _v8p_2 = (const void*)(weight + (track * 2560 + local_tid * 16) + (0));
        uint32_t _v8_2_0[8];
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
        (&_vec_load_2[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
        (&_vec_load_2[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
        (&_vec_load_2[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
        (&_vec_load_2[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
        (&_vec_load_2[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
        (&_vec_load_2[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
        (&_vec_load_2[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
        (&_vec_load_2[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
        (&_vec_load_2[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
        (&_vec_load_2[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
        (&_vec_load_2[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
        (&_vec_load_2[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
        (&_vec_load_2[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
        (&_vec_load_2[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
        (&_vec_load_2[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
        (&_vec_load_2[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
    }
    unsigned long long control_address = workspace[24];
    int* control_0 = reinterpret_cast<int*>(control_address);
    int control_current_word = control_0[2];
    int control_bytes_per_buffer_word = control_0[3];
    int control_clear_word = control_0[4];
    int control_current = control_current_word % 3;
    int control_dirty = (control_current_word + 2) % 3;
    __nv_bfloat16* local_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + rank]);
    long long data_base = (long long)control_current * (long long)control_bytes_per_buffer_word / 2;
    long long clear_base = (long long)control_dirty * (long long)control_bytes_per_buffer_word / 2;
    int total_vectors = tokens * 160;
    int vector_index = token * 160 + tid;
    __syncthreads();
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(control_0)) + (0))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    int local_warp = local_tid / 32;
    int scratch_row = track * 32;
    float propagated[16];
    float square_sum = 0.0f;
    #pragma unroll
    for (int element = 0; element < 16; element++) {
        propagated[element] = (float)(__nv_bfloat16)(_vec_load_0[element] + _vec_load_1[element]);
        square_sum = __fmaf_rn(propagated[element], propagated[element], square_sum);
    }
    unsigned int propagated_bf16[8];
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(propagated[_lp*2 + 0], propagated[_lp*2+1 + 0]));
        propagated_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    if (target == rank) {
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(residual_out + offset), "r"((propagated_bf16)[0]), "r"((propagated_bf16)[1]), "r"((propagated_bf16)[2]), "r"((propagated_bf16)[3]), "r"((propagated_bf16)[4]), "r"((propagated_bf16)[5]), "r"((propagated_bf16)[6]), "r"((propagated_bf16)[7]) : "memory");
    }
    float _warp_reduce_0 = square_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    if (lane == 0) {
        scratch[scratch_row + local_warp] = _warp_reduce_0;
    }
    __syncthreads();
    if (local_warp == 0) {
        float partial = ((lane < 5) ? scratch[scratch_row + lane] : 0.0f);
        float _warp_reduce_1 = partial;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        if (lane == 0) {
            scratch[scratch_row] = _warp_reduce_1;
        }
    }
    __syncthreads();
    float total_square_sum = scratch[scratch_row];
    float _fdiv_3 = __fdiv_rn(total_square_sum, (float)2560);
    float _rsqrt_0 = rsqrtf(_fdiv_3 + epsilon);
    float normalized[16];
    #pragma unroll
    for (int element_1 = 0; element_1 < 16; element_1++) {
        normalized[element_1] = (float)(__nv_bfloat16)(propagated[element_1] * _rsqrt_0 * _vec_load_2[element_1]);
    }
    unsigned int normalized_bf16[8];
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
        normalized_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    if (target == rank) {
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(norm_out + offset), "r"((normalized_bf16)[0]), "r"((normalized_bf16)[1]), "r"((normalized_bf16)[2]), "r"((normalized_bf16)[3]), "r"((normalized_bf16)[4]), "r"((normalized_bf16)[5]), "r"((normalized_bf16)[6]), "r"((normalized_bf16)[7]) : "memory");
    }
    if (track == 1) {
        #pragma unroll
        for (int element_2 = 0; element_2 < 16; element_2++) {
            exchange[element_2 * 160 + local_tid] = normalized[element_2];
        }
    }
    __syncthreads();
    float mean[16];
    #pragma unroll
    for (int element_3 = 0; element_3 < 16; element_3++) {
        mean[element_3] = 0.0f;
    }
    if (track == 0) {
        #pragma unroll
        for (int element_4 = 0; element_4 < 16; element_4++) {
            mean[element_4] = __fmaf_rn(normalized[element_4], 0.5f, mean[element_4]);
            float other_track = exchange[element_4 * 160 + local_tid];
            mean[element_4] = __fmaf_rn(other_track, 0.5f, mean[element_4]);
        }
    }
    float contribution[16];
    #pragma unroll
    for (int element_5 = 0; element_5 < 16; element_5++) {
        float _cvt_f32_0 = (float)(__nv_bfloat16)mean[element_5];
        contribution[element_5] = ((_cvt_f32_0 == 0.0f) ? 0.0f : _cvt_f32_0);
    }
    unsigned int contribution_bf16[8];
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contribution[_lp*2 + 0], contribution[_lp*2+1 + 0]));
        contribution_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    if (tid < 160) {
        long long publish_offset = data_base + ((long long)rank * (long long)total_vectors + (long long)vector_index) * 16;
        __nv_bfloat16* peer_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + target]);
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload + publish_offset), "r"((contribution_bf16)[0]), "r"((contribution_bf16)[1]), "r"((contribution_bf16)[2]), "r"((contribution_bf16)[3]), "r"((contribution_bf16)[4]), "r"((contribution_bf16)[5]), "r"((contribution_bf16)[6]), "r"((contribution_bf16)[7]) : "memory");
    }
    unsigned int sentinel[8];
    #pragma unroll
    for (int word = 0; word < 8; word++) {
        sentinel[word] = 2147516416;
    }
    int clear_start = bid * 320 + tid;
    #pragma unroll 1
    for (int clear_index = clear_start; clear_index < control_clear_word / 32; clear_index += num_bids * 320) {
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(local_payload + (clear_base + (long long)clear_index * 16)), "r"((sentinel)[0]), "r"((sentinel)[1]), "r"((sentinel)[2]), "r"((sentinel)[3]), "r"((sentinel)[4]), "r"((sentinel)[5]), "r"((sentinel)[6]), "r"((sentinel)[7]) : "memory");
    }
    if (target == rank) {
        if (tid < 160) {
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
            } while (
                (((_sysv_poll256_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[7] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[9] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[11] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[13] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[15] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[17] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[19] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[21] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[23] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[25] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[27] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[29] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[31] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[33] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[35] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[37] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[39] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[41] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[43] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[45] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[47] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[49] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[51] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[53] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[55] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[57] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[59] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[61] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_0[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[63] >> 16) & 0xffffu) == 0x8000u)
            );
            float _unpack_bf16_0[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_sysv_poll256_group_0[_pair]) << 16);
                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_sysv_poll256_group_0[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int peer = 1; peer < 8; peer++) {
                float _unpack_bf16_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((_sysv_poll256_group_0 + peer * 8)[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((_sysv_poll256_group_0 + peer * 8)[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int element_6 = 0; element_6 < 16; element_6++) {
                    _unpack_bf16_0[element_6] = (float)(__nv_bfloat16)(_unpack_bf16_0[element_6] + _unpack_bf16_1[element_6]);
                }
            }
            unsigned int _unpack_bf16_0_bf16[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_unpack_bf16_0[_lp*2 + 0], _unpack_bf16_0[_lp*2+1 + 0]));
                _unpack_bf16_0_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + ((long long)vector_index * 16)), "r"((_unpack_bf16_0_bf16)[0]), "r"((_unpack_bf16_0_bf16)[1]), "r"((_unpack_bf16_0_bf16)[2]), "r"((_unpack_bf16_0_bf16)[3]), "r"((_unpack_bf16_0_bf16)[4]), "r"((_unpack_bf16_0_bf16)[5]), "r"((_unpack_bf16_0_bf16)[6]), "r"((_unpack_bf16_0_bf16)[7]) : "memory");
        }
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _sysv_wait_p_4 = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned int*>(control_0)) + (0);
                while (*_sysv_wait_p_4 != static_cast<int>(num_bids)) {}
            }
            control_0[2] = (control_current_word + 1) % 3;
            control_0[4] = tokens * 2560 * 8 * 2;
            reinterpret_cast<unsigned int*>(control_0)[0] = 0;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
