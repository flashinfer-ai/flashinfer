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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_blackwell_msa_device_common.cuh"

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

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_blackwell_msa_711d5eeebecfcde61503(uint8_t* __restrict__ partial_o, __nv_bfloat16* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse)
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
    uint8_t* partial_stage = reinterpret_cast<uint8_t*>(smem_raw + SMEM_PARTIAL_STAGE_OFF);
    const int partial_stage_addr = smem + SMEM_PARTIAL_STAGE_OFF;
    unsigned int* partial_meta_stage = reinterpret_cast<unsigned int*>(smem_raw + SMEM_PARTIAL_META_STAGE_OFF);
    const int partial_meta_stage_addr = smem + SMEM_PARTIAL_META_STAGE_OFF;

    // === Task calls (dependency order) ===
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
    float lse0 = -CAKE_INF;
    float lse1 = -CAKE_INF;
    if (row_valid != 0 && split0 < split_count) {
        lse0 = partial_lse[(long long)split0 * (long long)total_rows_out + (long long)row];
    }
    if (row_valid != 0 && split1 < split_count) {
        lse1 = partial_lse[(long long)split1 * (long long)total_rows_out + (long long)row];
    }
    float _max_0 = max_noftz(lse0, lse1);
    float lse_max = _max_0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 4);
    float peer_max = _shfl_xor_0;
    if (peer_max > lse_max) {
        lse_max = peer_max;
    }
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 2);
    float peer_max_0 = _shfl_xor_1;
    if (peer_max_0 > lse_max) {
        lse_max = peer_max_0;
    }
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 1);
    float peer_max_1 = _shfl_xor_2;
    if (peer_max_1 > lse_max) {
        lse_max = peer_max_1;
    }
    float safe_lse_max = ((lse_max == -CAKE_INF) ? 0.0f : lse_max);
    float weight0 = 0.0f;
    float weight1 = 0.0f;
    if (split0 < split_count) {
        float _exp2_0 = approx_exp2((lse0 - safe_lse_max) * 1.4426950408889634f);
        weight0 = _exp2_0;
        if (lse0 == -CAKE_INF) {
            weight0 = 0.0f;
        }
    }
    if (split1 < split_count) {
        float _exp2_1 = approx_exp2((lse1 - safe_lse_max) * 1.4426950408889634f);
        weight1 = _exp2_1;
        if (lse1 == -CAKE_INF) {
            weight1 = 0.0f;
        }
    }
    float lse_sum = weight0 + weight1;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 4);
    lse_sum += _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 2);
    lse_sum += _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 1);
    lse_sum += _shfl_xor_5;
    float _rcp_0 = approx_rcp(lse_sum);
    float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_0 : 0.0f);
    weight0 *= inv_lse_sum;
    weight1 *= inv_lse_sum;
    if (row_valid != 0 && lane_in_row == 0) {
        float _log2_0;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(lse_sum));
        float combined_lse = ((lse_sum > 0.0f) ? safe_lse_max + _log2_0 * 0.6931471805599453f : -CAKE_INF);
        if (return_softmax_lse != 0) {
            lse[row] = combined_lse;
        }
        if (return_temperature_lse != 0) {
            temperature_lse[row] = combined_lse;
        }
    }
    int combine_output_valid = row_valid;
    if (split_count == 1) {
        combine_output_valid = 0;
    }
    if (combine_output_valid != 0) {
        int col_segment = lane_in_row * 16;
        int scale_segment = lane_in_row / 2;
        unsigned int magic = 1258291200;
        float2 _f2_0 = make_float2(-8388736.0f, -8388736.0f);
        float2 offset_pair = _f2_0;
        float2 _f2_1 = make_float2(0.0f, 0.0f);
        float2 zero_pair = _f2_1;
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
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, source_weight, source_lane_base + split % 8);
            float split_weight = _shfl_0;
            asm volatile("cp.async.wait_group 7;");
            unsigned int packed_payload[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_payload[(0) + 3]))
                : "r"(partial_stage_addr + (unsigned int)(split % 8 * 4096) + (unsigned int)(tid * 16)));
            if (split_count > split) {
                unsigned int packed_scale[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_scale[0])) : "r"(partial_meta_stage_addr + (unsigned int)(split % 8 * 1024) + (unsigned int)(tid * 4)));
                float packed_scale_f32[2];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&packed_scale_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_scale[_pair]) << 16);
                    (&packed_scale_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_scale[_pair]) & 0xffff0000u);
                }
                float dequant_scale = (((scale_segment & 1) == 0) ? packed_scale_f32[0] : packed_scale_f32[1]);
                float row_weight = split_weight * dequant_scale;
                float2 _f2_2 = make_float2(row_weight, row_weight);
                float2 weight_pair = _f2_2;
                uint32_t _prmt_b32_0;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_b32_0) : "r"(packed_payload[0]), "r"(magic));
                uint32_t _prmt_b32_1;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_b32_1) : "r"(packed_payload[0]), "r"(magic));
                uint32_t _prmt_b32_2;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_b32_2) : "r"(packed_payload[0]), "r"(magic));
                uint32_t _prmt_b32_3;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_b32_3) : "r"(packed_payload[0]), "r"(magic));
                float2 _f2_3 = make_float2(__uint_as_float(_prmt_b32_0), __uint_as_float(_prmt_b32_1));
                accum_pair0 = fma_f32x2_rn_ftz(add_f32x2(_f2_3, offset_pair), weight_pair, accum_pair0);
                float2 _f2_4 = make_float2(__uint_as_float(_prmt_b32_2), __uint_as_float(_prmt_b32_3));
                accum_pair1 = fma_f32x2_rn_ftz(add_f32x2(_f2_4, offset_pair), weight_pair, accum_pair1);
                uint32_t _prmt_b32_4;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_b32_4) : "r"(packed_payload[1]), "r"(magic));
                uint32_t _prmt_b32_5;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_b32_5) : "r"(packed_payload[1]), "r"(magic));
                uint32_t _prmt_b32_6;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_b32_6) : "r"(packed_payload[1]), "r"(magic));
                uint32_t _prmt_b32_7;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_b32_7) : "r"(packed_payload[1]), "r"(magic));
                float2 _f2_5 = make_float2(__uint_as_float(_prmt_b32_4), __uint_as_float(_prmt_b32_5));
                accum_pair2 = fma_f32x2_rn_ftz(add_f32x2(_f2_5, offset_pair), weight_pair, accum_pair2);
                float2 _f2_6 = make_float2(__uint_as_float(_prmt_b32_6), __uint_as_float(_prmt_b32_7));
                accum_pair3 = fma_f32x2_rn_ftz(add_f32x2(_f2_6, offset_pair), weight_pair, accum_pair3);
                uint32_t _prmt_b32_8;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_b32_8) : "r"(packed_payload[2]), "r"(magic));
                uint32_t _prmt_b32_9;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_b32_9) : "r"(packed_payload[2]), "r"(magic));
                uint32_t _prmt_b32_10;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_b32_10) : "r"(packed_payload[2]), "r"(magic));
                uint32_t _prmt_b32_11;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_b32_11) : "r"(packed_payload[2]), "r"(magic));
                float2 _f2_7 = make_float2(__uint_as_float(_prmt_b32_8), __uint_as_float(_prmt_b32_9));
                accum_pair4 = fma_f32x2_rn_ftz(add_f32x2(_f2_7, offset_pair), weight_pair, accum_pair4);
                float2 _f2_8 = make_float2(__uint_as_float(_prmt_b32_10), __uint_as_float(_prmt_b32_11));
                accum_pair5 = fma_f32x2_rn_ftz(add_f32x2(_f2_8, offset_pair), weight_pair, accum_pair5);
                uint32_t _prmt_b32_12;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_b32_12) : "r"(packed_payload[3]), "r"(magic));
                uint32_t _prmt_b32_13;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_b32_13) : "r"(packed_payload[3]), "r"(magic));
                uint32_t _prmt_b32_14;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_b32_14) : "r"(packed_payload[3]), "r"(magic));
                uint32_t _prmt_b32_15;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_b32_15) : "r"(packed_payload[3]), "r"(magic));
                float2 _f2_9 = make_float2(__uint_as_float(_prmt_b32_12), __uint_as_float(_prmt_b32_13));
                accum_pair6 = fma_f32x2_rn_ftz(add_f32x2(_f2_9, offset_pair), weight_pair, accum_pair6);
                float2 _f2_10 = make_float2(__uint_as_float(_prmt_b32_14), __uint_as_float(_prmt_b32_15));
                accum_pair7 = fma_f32x2_rn_ftz(add_f32x2(_f2_10, offset_pair), weight_pair, accum_pair7);
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
