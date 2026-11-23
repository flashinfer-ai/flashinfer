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
#define SMEM_PARTIAL_STAGE_OFF 0
#define SMEM_PARTIAL_STAGE_STAGE_BYTES 32768
#define SMEM_PARTIAL_STAGE_STRIDE 32768
#define SMEM_PARTIAL_META_STAGE_OFF 32768
#define SMEM_PARTIAL_META_STAGE_STAGE_BYTES 8192
#define SMEM_PARTIAL_META_STAGE_STRIDE 8192
#define SMEM_TOTAL 40960
#define THREADS 256

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



__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_blackwell_msa_5a7c4a2da9b6e9c14215(uint8_t* __restrict__ partial_o, __nv_bfloat16* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse)
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
    uint8_t* partial_stage = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int partial_stage_addr = smem + 0;
    unsigned int* partial_meta_stage = reinterpret_cast<unsigned int*>(smem_raw + 32768);
    const int partial_meta_stage_addr = smem + 32768;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int row_group = tid / 8;
    int lane_in_row = lane % 8;
    int source_lane_base = lane / 8 * 8;
    int row = blockIdx.x * 32 + row_group;
    int total_rows_out = total_q * num_q_heads;
    int row_valid = ((row < total_rows_out) ? 1 : 0);
    int split_count = 0;
    if (row_valid != 0) {
        int q_abs = row / num_q_heads;
        int q_head = row - q_abs * num_q_heads;
        int kv_head = q_head / qhead_per_kv;
        split_count = split_counts[q_abs * num_kv_heads + kv_head];
        if (split_count > topk) {
            split_count = topk;
        }
        if (split_count > 16) {
            split_count = 16;
        }
        if (split_count < 0) {
            split_count = 0;
        }
    }
    int split0 = lane_in_row;
    int split1 = lane_in_row + 8;
    float lse0 = -CUDART_INF_F;
    float lse1 = -CUDART_INF_F;
    if (row_valid != 0 && split0 < split_count) {
        lse0 = partial_lse[(long long)split0 * (long long)total_rows_out + (long long)row];
    }
    if (row_valid != 0 && split1 < split_count) {
        lse1 = partial_lse[(long long)split1 * (long long)total_rows_out + (long long)row];
    }
    float lse_max;
    asm("max.f32 %0, %1, %2;" : "=f"(lse_max) : "f"(lse0), "f"(lse1));
    float peer_max = __shfl_xor_sync(0xFFFFFFFF, lse_max, 4);
    if (peer_max > lse_max) {
        lse_max = peer_max;
    }
    float peer_max_0 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 2);
    if (peer_max_0 > lse_max) {
        lse_max = peer_max_0;
    }
    float peer_max_1 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 1);
    if (peer_max_1 > lse_max) {
        lse_max = peer_max_1;
    }
    float safe_lse_max = ((lse_max == -CUDART_INF_F) ? 0.0f : lse_max);
    float weight0 = 0.0f;
    float weight1 = 0.0f;
    if (split0 < split_count) {
        float _exp2_0 = approx_exp2((lse0 - safe_lse_max) * 1.442695f);
        weight0 = _exp2_0;
        if (lse0 == -CUDART_INF_F) {
            weight0 = 0.0f;
        }
    }
    if (split1 < split_count) {
        float _exp2_1 = approx_exp2((lse1 - safe_lse_max) * 1.442695f);
        weight1 = _exp2_1;
        if (lse1 == -CUDART_INF_F) {
            weight1 = 0.0f;
        }
    }
    float lse_sum = weight0 + weight1;
    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 4);
    lse_sum += _shfl_xor_sync_2;
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 2);
    lse_sum += _shfl_xor_sync_3;
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 1);
    lse_sum += _shfl_xor_sync_4;
    float _rcp_5 = approx_rcp(lse_sum);
    float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_5 : 0.0f);
    weight0 *= inv_lse_sum;
    weight1 *= inv_lse_sum;
    if (row_valid != 0 && lane_in_row == 0) {
        float _log2_6;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(lse_sum));
        float combined_lse = ((lse_sum > 0.0f) ? __fmaf_rn(_log2_6, 0.6931472f, safe_lse_max) : -CUDART_INF_F);
        if (return_softmax_lse != 0) {
            lse[row] = combined_lse;
        }
        if (return_temperature_lse != 0) {
            temperature_lse[row] = combined_lse;
        }
    }
    int combine_output_valid = row_valid;
    if (combine_output_valid != 0) {
        int col_segment = lane_in_row * 16;
        int scale_segment = lane_in_row / 2;
        unsigned int magic = 1258291200;
        float2 offset_pair = make_float2(-8.388736e+06f, -8.388736e+06f);
        float2 zero_pair = make_float2(0.0f, 0.0f);
        float2 accum_pair0 = zero_pair;
        float2 accum_pair1 = zero_pair;
        float2 accum_pair2 = zero_pair;
        float2 accum_pair3 = zero_pair;
        float2 accum_pair4 = zero_pair;
        float2 accum_pair5 = zero_pair;
        float2 accum_pair6 = zero_pair;
        float2 accum_pair7 = zero_pair;
        for (int preload_split = 0; preload_split < 8; preload_split++) {
            long long preload_split_row = (long long)preload_split * (long long)total_rows_out + (long long)row;
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"(partial_stage_addr + (unsigned int)(preload_split * 4096) + (unsigned int)(tid * 16)), "l"(partial_o + (preload_split_row * 128 + (long long)col_segment)), "r"((split_count > preload_split) ? 16 : 0));
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.ca.shared::cta.global.L2::128B [%1], [%2], 4;\n\t"
                "}"
                :: "r"((split_count > preload_split) ? 1 : 0), "r"(partial_meta_stage_addr + (unsigned int)(preload_split * 1024) + (unsigned int)(tid * 4)), "l"(partial_scale + (preload_split_row * 4 + (long long)(scale_segment / 2 * 2))));
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int split = 0; split < 16; split++) {
            float source_weight = ((split < 8) ? weight0 : weight1);
            float split_weight = __shfl_sync(0xFFFFFFFF, source_weight, source_lane_base + split % 8);
            asm volatile("cp.async.wait_group 7;");
            unsigned int packed_payload[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[(0) + 3]))
                : "r"(partial_stage_addr + (unsigned int)(split % 8 * 4096) + (unsigned int)(tid * 16)));
            if (split_count > split) {
                unsigned int packed_scale[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_scale[0])) : "r"(partial_meta_stage_addr + (unsigned int)(split % 8 * 1024) + (unsigned int)(tid * 4)));
                float _unpack_bf16_0[2];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_scale[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_scale[_pair]) & 0xffff0000u);
                }
                float dequant_scale = (((scale_segment & 1) == 0) ? _unpack_bf16_0[0] : _unpack_bf16_0[1]);
                float row_weight = split_weight * dequant_scale;
                float2 weight_pair = make_float2(row_weight, row_weight);
                uint32_t _prmt_0;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_0) : "r"(packed_payload[0]), "r"(magic));
                uint32_t _prmt_1;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_1) : "r"(packed_payload[0]), "r"(magic));
                uint32_t _prmt_2;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_2) : "r"(packed_payload[0]), "r"(magic));
                uint32_t _prmt_3;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_3) : "r"(packed_payload[0]), "r"(magic));
                accum_pair0 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_0), __uint_as_float(_prmt_1)), offset_pair), weight_pair, accum_pair0);
                accum_pair1 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_2), __uint_as_float(_prmt_3)), offset_pair), weight_pair, accum_pair1);
                uint32_t _prmt_4;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_4) : "r"(packed_payload[1]), "r"(magic));
                uint32_t _prmt_5;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_5) : "r"(packed_payload[1]), "r"(magic));
                uint32_t _prmt_6;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_6) : "r"(packed_payload[1]), "r"(magic));
                uint32_t _prmt_7;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_7) : "r"(packed_payload[1]), "r"(magic));
                accum_pair2 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_4), __uint_as_float(_prmt_5)), offset_pair), weight_pair, accum_pair2);
                accum_pair3 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_6), __uint_as_float(_prmt_7)), offset_pair), weight_pair, accum_pair3);
                uint32_t _prmt_8;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_8) : "r"(packed_payload[2]), "r"(magic));
                uint32_t _prmt_9;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_9) : "r"(packed_payload[2]), "r"(magic));
                uint32_t _prmt_10;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_10) : "r"(packed_payload[2]), "r"(magic));
                uint32_t _prmt_11;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_11) : "r"(packed_payload[2]), "r"(magic));
                accum_pair4 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_8), __uint_as_float(_prmt_9)), offset_pair), weight_pair, accum_pair4);
                accum_pair5 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_10), __uint_as_float(_prmt_11)), offset_pair), weight_pair, accum_pair5);
                uint32_t _prmt_12;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_12) : "r"(packed_payload[3]), "r"(magic));
                uint32_t _prmt_13;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_13) : "r"(packed_payload[3]), "r"(magic));
                uint32_t _prmt_14;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_14) : "r"(packed_payload[3]), "r"(magic));
                uint32_t _prmt_15;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_15) : "r"(packed_payload[3]), "r"(magic));
                accum_pair6 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_12), __uint_as_float(_prmt_13)), offset_pair), weight_pair, accum_pair6);
                accum_pair7 = fma_f32x2_rn_ftz(add_f32x2(make_float2(__uint_as_float(_prmt_14), __uint_as_float(_prmt_15)), offset_pair), weight_pair, accum_pair7);
            }
            int next_split = split + 8;
            int safe_next_split = ((next_split < 16) ? next_split : 0);
            long long next_split_row = (long long)safe_next_split * (long long)total_rows_out + (long long)row;
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"(partial_stage_addr + (unsigned int)(split % 8 * 4096) + (unsigned int)(tid * 16)), "l"(partial_o + (next_split_row * 128 + (long long)col_segment)), "r"((next_split < 16 && next_split < split_count) ? 16 : 0));
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.ca.shared::cta.global.L2::128B [%1], [%2], 4;\n\t"
                "}"
                :: "r"((next_split < 16 && next_split < split_count) ? 1 : 0), "r"(partial_meta_stage_addr + (unsigned int)(split % 8 * 1024) + (unsigned int)(tid * 4)), "l"(partial_scale + (next_split_row * 4 + (long long)(scale_segment / 2 * 2))));
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group 0;");
        float accum[16];
        accum[0] = accum_pair0.x;
        accum[1] = accum_pair0.y;
        accum[2] = accum_pair1.x;
        accum[3] = accum_pair1.y;
        accum[4] = accum_pair2.x;
        accum[5] = accum_pair2.y;
        accum[6] = accum_pair3.x;
        accum[7] = accum_pair3.y;
        accum[8] = accum_pair4.x;
        accum[9] = accum_pair4.y;
        accum[10] = accum_pair5.x;
        accum[11] = accum_pair5.y;
        accum[12] = accum_pair6.x;
        accum[13] = accum_pair6.y;
        accum[14] = accum_pair7.x;
        accum[15] = accum_pair7.y;
        {
            __nv_bfloat162 _pk[8];
            _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
            _pk[4] = __floats2bfloat162_rn(accum[0 + 8], accum[0 + 9]);
            _pk[5] = __floats2bfloat162_rn(accum[0 + 10], accum[0 + 11]);
            _pk[6] = __floats2bfloat162_rn(accum[0 + 12], accum[0 + 13]);
            _pk[7] = __floats2bfloat162_rn(accum[0 + 14], accum[0 + 15]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row * 128 + (long long)col_segment)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row * 128 + (long long)col_segment)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
        }
    }
}

} // extern "C"
