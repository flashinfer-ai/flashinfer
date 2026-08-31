typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) BlackwellMsaTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define BLACKWELL_MSA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PARTIAL_STAGE_OFF 0
#define SMEM_PARTIAL_STAGE_STAGE_BYTES 32768
#define SMEM_PARTIAL_STAGE_STRIDE 32768
#define SMEM_PARTIAL_META_STAGE_OFF 32768
#define SMEM_PARTIAL_META_STAGE_STAGE_BYTES 4096
#define SMEM_PARTIAL_META_STAGE_STRIDE 4096
#define SMEM_TOTAL 36864
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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_minimax_sparse_reverse_prefill_combine_topk16_fp8partial_bf16_sm100(uint8_t* __restrict__ partial_o, float* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse)
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
    uint8_t* partial_stage = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int partial_stage_addr = smem + 0;
    __nv_bfloat16* partial_meta_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int partial_meta_stage_addr = smem + 32768;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int row_group = tid / 8;
    int lane_in_row = lane % 8;
    int source_lane_base = lane / 8 * 8;
    int row = blockIdx.x * 32 + row_group;
    int total_rows_out = total_q * num_q_heads;
    long long partial_metadata_rows = (long long)topk * (long long)total_rows_out;
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
    float lse0 = -BLACKWELL_MSA_INF;
    float lse1 = -BLACKWELL_MSA_INF;
    if (row_valid != 0 && split0 < split_count) {
        long long split_row0 = (long long)split0 * (long long)total_rows_out + (long long)row;
        lse0 = partial_lse[split_row0];
    }
    if (row_valid != 0 && split1 < split_count) {
        long long split_row1 = (long long)split1 * (long long)total_rows_out + (long long)row;
        lse1 = partial_lse[split_row1];
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
    float safe_lse_max = ((lse_max == -BLACKWELL_MSA_INF) ? 0.0f : lse_max);
    float weight0 = 0.0f;
    float weight1 = 0.0f;
    if (split0 < split_count) {
        float _exp2_0 = approx_exp2((lse0 - safe_lse_max) * 1.4426950408889634f);
        weight0 = _exp2_0;
        if (lse0 == -BLACKWELL_MSA_INF) {
            weight0 = 0.0f;
        }
    }
    if (split1 < split_count) {
        float _exp2_1 = approx_exp2((lse1 - safe_lse_max) * 1.4426950408889634f);
        weight1 = _exp2_1;
        if (lse1 == -BLACKWELL_MSA_INF) {
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
    if (return_temperature_lse != 0) {
        float temperature0 = -BLACKWELL_MSA_INF;
        float temperature1 = -BLACKWELL_MSA_INF;
        if (row_valid != 0 && split0 < split_count) {
            long long temperature_row0 = (long long)split0 * (long long)total_rows_out + (long long)row;
            temperature0 = partial_temperature_lse[temperature_row0];
        }
        if (row_valid != 0 && split1 < split_count) {
            long long temperature_row1 = (long long)split1 * (long long)total_rows_out + (long long)row;
            temperature1 = partial_temperature_lse[temperature_row1];
        }
        float _max_1 = max_noftz(temperature0, temperature1);
        float temperature_max = _max_1;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, temperature_max, 4);
        float peer_temperature_max = _shfl_xor_6;
        if (peer_temperature_max > temperature_max) {
            temperature_max = peer_temperature_max;
        }
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, temperature_max, 2);
        float peer_temperature_max_0 = _shfl_xor_7;
        if (peer_temperature_max_0 > temperature_max) {
            temperature_max = peer_temperature_max_0;
        }
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, temperature_max, 1);
        float peer_temperature_max_1 = _shfl_xor_8;
        if (peer_temperature_max_1 > temperature_max) {
            temperature_max = peer_temperature_max_1;
        }
        float safe_temperature_max = ((temperature_max == -BLACKWELL_MSA_INF) ? 0.0f : temperature_max);
        float temperature_sum = 0.0f;
        if (split0 < split_count) {
            float _exp2_2 = approx_exp2((temperature0 - safe_temperature_max) * 1.4426950408889634f);
            float temperature_contribution0 = _exp2_2;
            if (temperature0 == -BLACKWELL_MSA_INF) {
                temperature_contribution0 = 0.0f;
            }
            temperature_sum += temperature_contribution0;
        }
        if (split1 < split_count) {
            float _exp2_3 = approx_exp2((temperature1 - safe_temperature_max) * 1.4426950408889634f);
            float temperature_contribution1 = _exp2_3;
            if (temperature1 == -BLACKWELL_MSA_INF) {
                temperature_contribution1 = 0.0f;
            }
            temperature_sum += temperature_contribution1;
        }
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, temperature_sum, 4);
        temperature_sum += _shfl_xor_9;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, temperature_sum, 2);
        temperature_sum += _shfl_xor_10;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, temperature_sum, 1);
        temperature_sum += _shfl_xor_11;
        if (row_valid != 0 && lane_in_row == 0) {
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(temperature_sum));
            temperature_lse[row] = ((temperature_sum > 0.0f) ? safe_temperature_max + _log2_0 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
        }
    }
    float center_sum = 0.0f;
    float scaled_weight0 = weight0;
    float scaled_weight1 = weight1;
    {
        float scale0 = 0.0f;
        float scale1 = 0.0f;
        float center0 = 0.0f;
        float center1 = 0.0f;
        if (row_valid != 0 && split0 < split_count) {
            long long split_row0_1 = (long long)split0 * (long long)total_rows_out + (long long)row;
            scale0 = partial_scale[split_row0_1];
            center0 = partial_scale[partial_metadata_rows + split_row0_1];
        }
        if (row_valid != 0 && split1 < split_count) {
            long long split_row1_1 = (long long)split1 * (long long)total_rows_out + (long long)row;
            scale1 = partial_scale[split_row1_1];
            center1 = partial_scale[partial_metadata_rows + split_row1_1];
        }
        center_sum = weight0 * center0 + weight1 * center1;
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, center_sum, 4);
        center_sum += _shfl_xor_12;
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, center_sum, 2);
        center_sum += _shfl_xor_13;
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, center_sum, 1);
        center_sum += _shfl_xor_14;
        scaled_weight0 *= scale0;
        scaled_weight1 *= scale1;
    }
    if (return_softmax_lse != 0 && row_valid != 0 && lane_in_row == 0) {
        float _log2_1;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(lse_sum));
        float combined_lse = ((lse_sum > 0.0f) ? safe_lse_max + _log2_1 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
        lse[row] = combined_lse;
    }
    int combine_output_valid = row_valid;
    if (combine_output_valid != 0) {
        int col_segment = lane_in_row * 16;
        float2 _f2_0 = make_float2(center_sum, center_sum);
        float2 initial_center_pair = _f2_0;
        float2 accum_pair0 = initial_center_pair;
        float2 accum_pair1 = initial_center_pair;
        float2 accum_pair2 = initial_center_pair;
        float2 accum_pair3 = initial_center_pair;
        float2 accum_pair4 = initial_center_pair;
        float2 accum_pair5 = initial_center_pair;
        float2 accum_pair6 = initial_center_pair;
        float2 accum_pair7 = initial_center_pair;
        float segment_center_sum = 0.0f;
        for (int preload_split = 0; preload_split < 8; preload_split++) {
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"(partial_stage_addr + (unsigned int)(preload_split * 4096) + (unsigned int)(tid * 16)), "l"(partial_o + (((long long)preload_split * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment)), "r"((split_count > preload_split) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int split = 0; split < 16; split++) {
            float source_weight = ((split < 8) ? scaled_weight0 : scaled_weight1);
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, source_weight, source_lane_base + split % 8);
            float split_weight = _shfl_0;
            asm volatile("cp.async.wait_group 7;");
            float _partial_stage_reg_0[16];
            {
                uint32_t _smem_addr_0 = (uint32_t)__cvta_generic_to_shared(smem_raw + (unsigned int)(split % 8 * 4096 + tid * 16));
                uint4 _fp8x16_0;
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_fp8x16_0.x), "=r"(_fp8x16_0.y), "=r"(_fp8x16_0.z), "=r"(_fp8x16_0.w)
                    : "r"(_smem_addr_0) : "memory");
                uint16_t _e4m3x2_0_0 = (uint16_t)((_fp8x16_0.x >> 0) & 0xFFFFu);
                uint32_t _f16x2_0_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_0) : "h"(_e4m3x2_0_0));
                uint16_t _h0_0 = (uint16_t)((_f16x2_0_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[0]) : "h"(_h0_0));
                uint16_t _h1_0 = (uint16_t)((_f16x2_0_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[1]) : "h"(_h1_0));
                uint16_t _e4m3x2_1_0 = (uint16_t)((_fp8x16_0.x >> 16) & 0xFFFFu);
                uint32_t _f16x2_1_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_0) : "h"(_e4m3x2_1_0));
                uint16_t _h2_0 = (uint16_t)((_f16x2_1_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[2]) : "h"(_h2_0));
                uint16_t _h3_0 = (uint16_t)((_f16x2_1_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[3]) : "h"(_h3_0));
                uint16_t _e4m3x2_2_0 = (uint16_t)((_fp8x16_0.y >> 0) & 0xFFFFu);
                uint32_t _f16x2_2_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_0) : "h"(_e4m3x2_2_0));
                uint16_t _h4_0 = (uint16_t)((_f16x2_2_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[4]) : "h"(_h4_0));
                uint16_t _h5_0 = (uint16_t)((_f16x2_2_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[5]) : "h"(_h5_0));
                uint16_t _e4m3x2_3_0 = (uint16_t)((_fp8x16_0.y >> 16) & 0xFFFFu);
                uint32_t _f16x2_3_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_0) : "h"(_e4m3x2_3_0));
                uint16_t _h6_0 = (uint16_t)((_f16x2_3_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[6]) : "h"(_h6_0));
                uint16_t _h7_0 = (uint16_t)((_f16x2_3_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[7]) : "h"(_h7_0));
                uint16_t _e4m3x2_4_0 = (uint16_t)((_fp8x16_0.z >> 0) & 0xFFFFu);
                uint32_t _f16x2_4_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4_0) : "h"(_e4m3x2_4_0));
                uint16_t _h8_0 = (uint16_t)((_f16x2_4_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[8]) : "h"(_h8_0));
                uint16_t _h9_0 = (uint16_t)((_f16x2_4_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[9]) : "h"(_h9_0));
                uint16_t _e4m3x2_5_0 = (uint16_t)((_fp8x16_0.z >> 16) & 0xFFFFu);
                uint32_t _f16x2_5_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5_0) : "h"(_e4m3x2_5_0));
                uint16_t _h10_0 = (uint16_t)((_f16x2_5_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[10]) : "h"(_h10_0));
                uint16_t _h11_0 = (uint16_t)((_f16x2_5_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[11]) : "h"(_h11_0));
                uint16_t _e4m3x2_6_0 = (uint16_t)((_fp8x16_0.w >> 0) & 0xFFFFu);
                uint32_t _f16x2_6_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6_0) : "h"(_e4m3x2_6_0));
                uint16_t _h12_0 = (uint16_t)((_f16x2_6_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[12]) : "h"(_h12_0));
                uint16_t _h13_0 = (uint16_t)((_f16x2_6_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[13]) : "h"(_h13_0));
                uint16_t _e4m3x2_7_0 = (uint16_t)((_fp8x16_0.w >> 16) & 0xFFFFu);
                uint32_t _f16x2_7_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_7_0) : "h"(_e4m3x2_7_0));
                uint16_t _h14_0 = (uint16_t)((_f16x2_7_0 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[14]) : "h"(_h14_0));
                uint16_t _h15_0 = (uint16_t)((_f16x2_7_0 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_partial_stage_reg_0[15]) : "h"(_h15_0));
            }
            if (split_count > split) {
                {
                    float2 _f2_10 = make_float2(split_weight, split_weight);
                    float2 split_weight_pair = _f2_10;
                    float2 _f2_11 = make_float2(_partial_stage_reg_0[0], _partial_stage_reg_0[1]);
                    accum_pair0 = fma_f32x2(_f2_11, split_weight_pair, accum_pair0);
                    float2 _f2_12 = make_float2(_partial_stage_reg_0[2], _partial_stage_reg_0[3]);
                    accum_pair1 = fma_f32x2(_f2_12, split_weight_pair, accum_pair1);
                    float2 _f2_13 = make_float2(_partial_stage_reg_0[4], _partial_stage_reg_0[5]);
                    accum_pair2 = fma_f32x2(_f2_13, split_weight_pair, accum_pair2);
                    float2 _f2_14 = make_float2(_partial_stage_reg_0[6], _partial_stage_reg_0[7]);
                    accum_pair3 = fma_f32x2(_f2_14, split_weight_pair, accum_pair3);
                    float2 _f2_15 = make_float2(_partial_stage_reg_0[8], _partial_stage_reg_0[9]);
                    accum_pair4 = fma_f32x2(_f2_15, split_weight_pair, accum_pair4);
                    float2 _f2_16 = make_float2(_partial_stage_reg_0[10], _partial_stage_reg_0[11]);
                    accum_pair5 = fma_f32x2(_f2_16, split_weight_pair, accum_pair5);
                    float2 _f2_17 = make_float2(_partial_stage_reg_0[12], _partial_stage_reg_0[13]);
                    accum_pair6 = fma_f32x2(_f2_17, split_weight_pair, accum_pair6);
                    float2 _f2_18 = make_float2(_partial_stage_reg_0[14], _partial_stage_reg_0[15]);
                    accum_pair7 = fma_f32x2(_f2_18, split_weight_pair, accum_pair7);
                }
            }
            int next_split = split + 8;
            int safe_next_split = ((next_split < 16) ? next_split : 0);
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"(partial_stage_addr + (unsigned int)(split % 8 * 4096) + (unsigned int)(tid * 16)), "l"(partial_o + (((long long)safe_next_split * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment)), "r"((next_split < 16 && next_split < split_count) ? 16 : 0));
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
