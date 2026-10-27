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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_KV_U8_OFF 0
#define SMEM_KV_U8_STAGE_BYTES 36864
#define SMEM_KV_U8_STRIDE 36864
#define SMEM_Q_FRAG_OFF 36864
#define SMEM_Q_FRAG_STAGE_BYTES 8192
#define SMEM_Q_FRAG_STRIDE 8192
#define SMEM_P_FRAG_OFF 45056
#define SMEM_P_FRAG_STAGE_BYTES 2048
#define SMEM_P_FRAG_STRIDE 2048
#define SMEM_ROW_MAX_S_OFF 47104
#define SMEM_ROW_MAX_S_STAGE_BYTES 256
#define SMEM_ROW_MAX_S_STRIDE 256
#define SMEM_ROW_SUM_S_OFF 47360
#define SMEM_ROW_SUM_S_STAGE_BYTES 256
#define SMEM_ROW_SUM_S_STRIDE 256
#define SMEM_VALID_S_OFF 47616
#define SMEM_VALID_S_STAGE_BYTES 16
#define SMEM_VALID_S_STRIDE 16
#define SMEM_ADDR_RING_OFF 47632
#define SMEM_ADDR_RING_STAGE_BYTES 2048
#define SMEM_ADDR_RING_STRIDE 2048
#define SMEM_LUT_S_OFF 49696
#define SMEM_LUT_S_STAGE_BYTES 16
#define SMEM_LUT_S_STRIDE 16
#define SMEM_SLOT_VALID_OFF 49680
#define SMEM_SLOT_VALID_STAGE_BYTES 16
#define SMEM_SLOT_VALID_STRIDE 16
#define SMEM_LSE_TAB_OFF 0
#define SMEM_LSE_TAB_STAGE_BYTES 36800
#define SMEM_LSE_TAB_STRIDE 36800
#define SMEM_MAX_TAB_OFF 36800
#define SMEM_MAX_TAB_STAGE_BYTES 64
#define SMEM_MAX_TAB_STRIDE 64
#define SMEM_TOTAL 49792
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_qsa_nvfp4_4f13e4710a51ac27ff6f(__nv_bfloat16* __restrict__ q, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int* __restrict__ indices, int* __restrict__ block_table, int* __restrict__ token_to_req, __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, float* partial_out, float* partial_lse, unsigned int* __restrict__ merge_counters, long long q_stride_row, long long q_stride_head, long long data_stride_block, long long data_stride_token, long long data_stride_head, long long sf_stride_block, long long sf_stride_token, long long sf_stride_head, long long indices_stride_row, long long table_stride_req, long long out_stride_row, long long out_stride_head, float score_scale, float output_scale, int num_rows, int num_requests, int num_blocks, CakeFastDivmod page_div, int table_width, int topk, int group_size, int num_qo_heads, int num_tiles, int num_splits)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* kv_u8 = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int kv_u8_addr = smem + 0;
    unsigned int* q_frag = reinterpret_cast<unsigned int*>(smem_raw + 36864);
    const int q_frag_addr = smem + 36864;
    unsigned int* p_frag = reinterpret_cast<unsigned int*>(smem_raw + 45056);
    const int p_frag_addr = smem + 45056;
    float* row_max_s = reinterpret_cast<float*>(smem_raw + 47104);
    const int row_max_s_addr = smem + 47104;
    float* row_sum_s = reinterpret_cast<float*>(smem_raw + 47360);
    const int row_sum_s_addr = smem + 47360;
    unsigned int* valid_s = reinterpret_cast<unsigned int*>(smem_raw + 47616);
    const int valid_s_addr = smem + 47616;
    unsigned int* addr_ring = reinterpret_cast<unsigned int*>(smem_raw + 47632);
    const int addr_ring_addr = smem + 47632;
    unsigned int* lut_s = reinterpret_cast<unsigned int*>(smem_raw + 49696);
    const int lut_s_addr = smem + 49696;
    unsigned int* slot_valid = reinterpret_cast<unsigned int*>(smem_raw + 49680);
    const int slot_valid_addr = smem + 49680;
    float* lse_tab = reinterpret_cast<float*>(smem_raw + 0);
    const int lse_tab_addr = smem + 0;
    float* max_tab = reinterpret_cast<float*>(smem_raw + 36800);
    const int max_tab_addr = smem + 36800;

    // === Task calls (declaration order) ===
    int row = blockIdx.x;
    int kv_head = blockIdx.y;
    int split = blockIdx.z;
    int tid_0 = tid;
    int warp_1 = tid_0 >> 5;
    int lane_1 = tid_0 & 31;
    int g = lane_1 >> 2;
    int t = lane_1 & 3;
    long long row_i64 = (long long)row;
    int request = token_to_req[row];
    bool request_ok = request >= 0 && request < num_requests;
    int _max_0 = ((request) > (0) ? (request) : (0));
    int safe_request = ((_max_0) < (num_requests - 1) ? (_max_0) : (num_requests - 1));
    long long safe_request_row = (long long)safe_request * table_stride_req;
    long long indices_row = row_i64 * indices_stride_row;
    int valid_count = indices[indices_row + (long long)topk];
    int _min_1 = ((valid_count) < (topk) ? (valid_count) : (topk));
    int tile_end = ((num_tiles) < ((_min_1 + 63) / 64) ? (num_tiles) : ((_min_1 + 63) / 64));
    int first_head = kv_head * group_size;
    int token = tid_0 & 63;
    int idx[3];
    int column = split * 64 + token;
    int safe_column = ((column) < (topk - 1) ? (column) : (topk - 1));
    int raw = indices[indices_row + (long long)safe_column];
    idx[0] = raw;
    int column_1 = (split + num_splits) * 64 + token;
    int safe_column_2 = ((column_1) < (topk - 1) ? (column_1) : (topk - 1));
    int raw_3 = indices[indices_row + (long long)safe_column_2];
    idx[1] = raw_3;
    int column_4 = (split + 2 * num_splits) * 64 + token;
    int safe_column_5 = ((column_4) < (topk - 1) ? (column_4) : (topk - 1));
    int raw_6 = indices[indices_row + (long long)safe_column_5];
    idx[2] = raw_6;
    long long q_row = row_i64 * q_stride_row;
    unsigned int zero_word = 0;
    int e = tid_0;
    int e_step = e >> 5;
    int e_lane = e & 31;
    int e_g = e_lane >> 2;
    int e_t = e_lane & 3;
    bool lo_ok = e_g < group_size;
    bool hi_ok = e_g + 8 < group_size;
    long long dim = q_row + (long long)(64 * e_t + 4 * e_step);
    long long lo_src = dim + (long long)(first_head + e_g) * q_stride_head;
    long long hi_src = lo_src + 8 * q_stride_head;
    unsigned int w0 = zero_word;
    unsigned int w1 = zero_word;
    unsigned int w2 = zero_word;
    unsigned int w3 = zero_word;
    if (lo_ok) {
        unsigned int _vec_load_0[2];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(q + lo_src + 0);
            uint2* _vdst_2 = reinterpret_cast<uint2*>(&_vec_load_0[0]);
            *_vdst_2 = _vld_2;
        }
        w0 = _vec_load_0[0];
        w2 = _vec_load_0[1];
    }
    if (hi_ok) {
        unsigned int _vec_load_1[2];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(q + hi_src + 0);
            uint2* _vdst_3 = reinterpret_cast<uint2*>(&_vec_load_1[0]);
            *_vdst_3 = _vld_3;
        }
        w1 = _vec_load_1[0];
        w3 = _vec_load_1[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step + e_lane))), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
    int e_7 = tid_0 + 128;
    int e_step_8 = e_7 >> 5;
    int e_lane_9 = e_7 & 31;
    int e_g_10 = e_lane_9 >> 2;
    int e_t_11 = e_lane_9 & 3;
    bool lo_ok_12 = e_g_10 < group_size;
    bool hi_ok_13 = e_g_10 + 8 < group_size;
    long long dim_14 = q_row + (long long)(64 * e_t_11 + 4 * e_step_8);
    long long lo_src_15 = dim_14 + (long long)(first_head + e_g_10) * q_stride_head;
    long long hi_src_16 = lo_src_15 + 8 * q_stride_head;
    unsigned int w0_17 = zero_word;
    unsigned int w1_18 = zero_word;
    unsigned int w2_19 = zero_word;
    unsigned int w3_20 = zero_word;
    if (lo_ok_12) {
        unsigned int _vec_load_2[2];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(q + lo_src_15 + 0);
            uint2* _vdst_4 = reinterpret_cast<uint2*>(&_vec_load_2[0]);
            *_vdst_4 = _vld_4;
        }
        w0_17 = _vec_load_2[0];
        w2_19 = _vec_load_2[1];
    }
    if (hi_ok_13) {
        unsigned int _vec_load_3[2];
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(q + hi_src_16 + 0);
            uint2* _vdst_5 = reinterpret_cast<uint2*>(&_vec_load_3[0]);
            *_vdst_5 = _vld_5;
        }
        w1_18 = _vec_load_3[0];
        w3_20 = _vec_load_3[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_8 + e_lane_9))), "r"(w0_17), "r"(w1_18), "r"(w2_19), "r"(w3_20) : "memory");
    int e_21 = tid_0 + 256;
    int e_step_22 = e_21 >> 5;
    int e_lane_23 = e_21 & 31;
    int e_g_24 = e_lane_23 >> 2;
    int e_t_25 = e_lane_23 & 3;
    bool lo_ok_26 = e_g_24 < group_size;
    bool hi_ok_27 = e_g_24 + 8 < group_size;
    long long dim_28 = q_row + (long long)(64 * e_t_25 + 4 * e_step_22);
    long long lo_src_29 = dim_28 + (long long)(first_head + e_g_24) * q_stride_head;
    long long hi_src_30 = lo_src_29 + 8 * q_stride_head;
    unsigned int w0_31 = zero_word;
    unsigned int w1_32 = zero_word;
    unsigned int w2_33 = zero_word;
    unsigned int w3_34 = zero_word;
    if (lo_ok_26) {
        unsigned int _vec_load_4[2];
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(q + lo_src_29 + 0);
            uint2* _vdst_6 = reinterpret_cast<uint2*>(&_vec_load_4[0]);
            *_vdst_6 = _vld_6;
        }
        w0_31 = _vec_load_4[0];
        w2_33 = _vec_load_4[1];
    }
    if (hi_ok_27) {
        unsigned int _vec_load_5[2];
        {
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(q + hi_src_30 + 0);
            uint2* _vdst_7 = reinterpret_cast<uint2*>(&_vec_load_5[0]);
            *_vdst_7 = _vld_7;
        }
        w1_32 = _vec_load_5[0];
        w3_34 = _vec_load_5[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_22 + e_lane_23))), "r"(w0_31), "r"(w1_32), "r"(w2_33), "r"(w3_34) : "memory");
    int e_35 = tid_0 + 384;
    int e_step_36 = e_35 >> 5;
    int e_lane_37 = e_35 & 31;
    int e_g_38 = e_lane_37 >> 2;
    int e_t_39 = e_lane_37 & 3;
    bool lo_ok_40 = e_g_38 < group_size;
    bool hi_ok_41 = e_g_38 + 8 < group_size;
    long long dim_42 = q_row + (long long)(64 * e_t_39 + 4 * e_step_36);
    long long lo_src_43 = dim_42 + (long long)(first_head + e_g_38) * q_stride_head;
    long long hi_src_44 = lo_src_43 + 8 * q_stride_head;
    unsigned int w0_45 = zero_word;
    unsigned int w1_46 = zero_word;
    unsigned int w2_47 = zero_word;
    unsigned int w3_48 = zero_word;
    if (lo_ok_40) {
        unsigned int _vec_load_6[2];
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(q + lo_src_43 + 0);
            uint2* _vdst_8 = reinterpret_cast<uint2*>(&_vec_load_6[0]);
            *_vdst_8 = _vld_8;
        }
        w0_45 = _vec_load_6[0];
        w2_47 = _vec_load_6[1];
    }
    if (hi_ok_41) {
        unsigned int _vec_load_7[2];
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(q + hi_src_44 + 0);
            uint2* _vdst_9 = reinterpret_cast<uint2*>(&_vec_load_7[0]);
            *_vdst_9 = _vld_9;
        }
        w1_46 = _vec_load_7[0];
        w3_48 = _vec_load_7[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_36 + e_lane_37))), "r"(w0_45), "r"(w1_46), "r"(w2_47), "r"(w3_48) : "memory");
    unsigned long long kv_head_u64 = (unsigned long long)kv_head;
    unsigned long long data_stride_block_u64 = (unsigned long long)data_stride_block;
    unsigned long long data_stride_token_u64 = (unsigned long long)data_stride_token;
    unsigned long long sf_stride_block_u64 = (unsigned long long)sf_stride_block;
    unsigned long long sf_stride_token_u64 = (unsigned long long)sf_stride_token;
    unsigned long long head_data = kv_head_u64 * (unsigned long long)data_stride_head;
    unsigned long long head_sf = kv_head_u64 * (unsigned long long)sf_stride_head;
    unsigned long long k_base = (unsigned long long)k_data + head_data;
    unsigned long long v_base = (unsigned long long)v_data + head_data;
    unsigned long long k_sf_base = (unsigned long long)k_sf + head_sf;
    unsigned long long v_sf_base = (unsigned long long)v_sf + head_sf;
    unsigned int kv_addr = kv_u8_addr;
#if __CUDA_ARCH__ == 900
    if (tid_0 == 0) {
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(lut_s_addr), "r"(2071689984), "r"(3229614080u) : "memory");
    }
    unsigned int lut_regs[2];
#else
#endif
    float o[32];
    o[0] = 0.0f;
    o[1] = 0.0f;
    o[2] = 0.0f;
    o[3] = 0.0f;
    o[4] = 0.0f;
    o[5] = 0.0f;
    o[6] = 0.0f;
    o[7] = 0.0f;
    o[8] = 0.0f;
    o[9] = 0.0f;
    o[10] = 0.0f;
    o[11] = 0.0f;
    o[12] = 0.0f;
    o[13] = 0.0f;
    o[14] = 0.0f;
    o[15] = 0.0f;
    o[16] = 0.0f;
    o[17] = 0.0f;
    o[18] = 0.0f;
    o[19] = 0.0f;
    o[20] = 0.0f;
    o[21] = 0.0f;
    o[22] = 0.0f;
    o[23] = 0.0f;
    o[24] = 0.0f;
    o[25] = 0.0f;
    o[26] = 0.0f;
    o[27] = 0.0f;
    o[28] = 0.0f;
    o[29] = 0.0f;
    o[30] = 0.0f;
    o[31] = 0.0f;
    float m[2];
    float l[2];
    m[0] = -1e+20f;
    l[0] = 0.0f;
    m[1] = -1e+20f;
    l[1] = 0.0f;
    float s[8];
    unsigned int ok[8];
    float row_max[2];
    float row_sum[2];
    float m_new[2];
    unsigned int sf_words[2];
    unsigned int sf0[2];
    unsigned int sf1[2];
    unsigned int raw0[4];
    unsigned int raw1[4];
    unsigned int d0[4];
    unsigned int d1[4];
    unsigned int a_frag[4];
    unsigned int b_lo[2];
    unsigned int b_hi[2];
    unsigned int p_a[4];
    unsigned int vw[4];
    unsigned int dp_lo[8];
    unsigned int dp_hi[8];
    unsigned int sfv[2];
    unsigned int b_pair[2];
    unsigned long long v_src[4];
    unsigned int v_bits[1];
    int phys[2];
    int local[2];
    int column_49 = split * 64 + token;
    bool in_range = column_49 < topk && tile_end > split;
    int logical = ((in_range) ? idx[0] : -1);
    idx[0] = logical;
    int column_50 = (split + num_splits) * 64 + token;
    bool in_range_51 = column_50 < topk && tile_end > split + num_splits;
    int logical_52 = ((in_range_51) ? idx[1] : -1);
    idx[1] = logical_52;
    int column_53 = (split + 2 * num_splits) * 64 + token;
    bool in_range_54 = column_53 < topk && tile_end > split + 2 * num_splits;
    int logical_55 = ((in_range_54) ? idx[2] : -1);
    idx[2] = logical_55;
    int safe_token = ((idx[0]) > (0) ? (idx[0]) : (0));
    uint32_t _fast_div_q_0 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token);
    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)safe_token) - _fast_div_q_0 * (uint32_t)(page_div.divisor);
    int page = (int)_fast_div_q_0;
    bool valid = request_ok && idx[0] >= 0 && page < table_width;
    int table_column = ((page) < (table_width - 1) ? (page) : (table_width - 1));
    int entry = ((valid) ? block_table[safe_request_row + (long long)table_column] : -1);
    phys[0] = entry;
    local[0] = (int)_fast_div_r_0;
    int safe_token_56 = ((idx[1]) > (0) ? (idx[1]) : (0));
    uint32_t _fast_div_q_1 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_56), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_56);
    uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)safe_token_56) - _fast_div_q_1 * (uint32_t)(page_div.divisor);
    int page_57 = (int)_fast_div_q_1;
    bool valid_58 = request_ok && idx[1] >= 0 && page_57 < table_width;
    int table_column_59 = ((page_57) < (table_width - 1) ? (page_57) : (table_width - 1));
    int entry_60 = ((valid_58) ? block_table[safe_request_row + (long long)table_column_59] : -1);
    phys[1] = entry_60;
    local[1] = (int)_fast_div_r_1;
    bool multi = tile_end > split + num_splits;
    if (multi) {
        if (warp_1 < 2) {
            bool valid_0 = phys[0] >= 0 && phys[0] < num_blocks;
            int _max_10 = ((phys[0]) > (0) ? (phys[0]) : (0));
            unsigned long long block = (unsigned long long)_max_10;
            unsigned long long local_u64 = (unsigned long long)local[0];
            unsigned long long data_off = block * data_stride_block_u64 + local_u64 * data_stride_token_u64;
            unsigned long long sf_off = block * sf_stride_block_u64 + local_u64 * sf_stride_token_u64;
            unsigned int w0_1 = (unsigned int)data_off;
            unsigned int w1_2 = (unsigned int)(data_off >> 32);
            unsigned int w2_3 = (unsigned int)sf_off;
            unsigned int w3_4 = (unsigned int)(sf_off >> 32);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(token * 16)), "r"(w0_1), "r"(w1_2), "r"(w2_3), "r"(w3_4) : "memory");
            unsigned int bits = __ballot_sync(0xFFFFFFFF, valid_0);
            if (lane_1 == 0) {
                slot_valid[warp_1] = bits;
            }
        }
        __syncthreads();
        if (tile_end > split) {
            unsigned int bits_1 = slot_valid[warp_1 >> 1] >> (unsigned int)(16 * (warp_1 & 1));
            if (tid_0 < 2) {
                valid_s[tid_0] = slot_valid[tid_0];
            }
            int col = lane_1 & 7;
            int sub = lane_1 >> 3;
            int token0 = 16 * warp_1 + sub;
            unsigned int entry_addr = addr_ring_addr + (unsigned int)(token0 * 16);
            unsigned int bits_sub = bits_1 >> (unsigned int)sub;
            unsigned long long col_off = (unsigned long long)(col * 16);
            unsigned int stage_base = kv_addr;
            unsigned int dst_k0 = ((stage_base + (unsigned int)(token0 * 144 + col * 16)) < (kv_addr + 36848) ? (stage_base + (unsigned int)(token0 * 144 + col * 16)) : (kv_addr + 36848));
            int sf_lane = lane_1 & 15;
            bool sf_is_v = lane_1 >= 16;
            int sf_token = 16 * warp_1 + sf_lane;
            unsigned int entries_0[4];
            unsigned int entries_1[4];
            unsigned int entries_2[4];
            unsigned int entries_3[4];
            unsigned int sf_entry[4];
            __int128_t _smem_raw_11;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(entry_addr));
            entries_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
            entries_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
            entries_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
            entries_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
            __int128_t _smem_raw_12;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(entry_addr + 64));
            entries_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
            entries_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
            entries_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
            entries_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
            __int128_t _smem_raw_13;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(entry_addr + 128));
            entries_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
            entries_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
            entries_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
            entries_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
            __int128_t _smem_raw_14;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(entry_addr + 192));
            entries_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
            entries_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
            entries_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
            entries_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
            __int128_t _smem_raw_15;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_15) : "r"(addr_ring_addr + (unsigned int)(sf_token * 16)));
            sf_entry[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[0];
            sf_entry[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[1];
            sf_entry[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[2];
            sf_entry[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[3];
            unsigned long long k_src_base = k_base + col_off;
            unsigned long long v_src_base = v_base + col_off;
            v_bits[0] = bits_sub;
            unsigned long long data_off_1 = (unsigned long long)entries_0[0] | (unsigned long long)entries_0[1] << 32;
            unsigned int src_size = (bits_sub & 1) << 4;
            v_src[0] = v_src_base + data_off_1;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0), "l"(k_src_base + data_off_1), "r"(src_size), "l"(0x12F0000000000000ULL));
            unsigned long long data_off_0 = (unsigned long long)entries_1[0] | (unsigned long long)entries_1[1] << 32;
            unsigned int src_size_1 = (bits_sub >> 4 & 1) << 4;
            v_src[1] = v_src_base + data_off_0;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0 + 576), "l"(k_src_base + data_off_0), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            unsigned long long data_off_2 = (unsigned long long)entries_2[0] | (unsigned long long)entries_2[1] << 32;
            unsigned int src_size_3 = (bits_sub >> 8 & 1) << 4;
            v_src[2] = v_src_base + data_off_2;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0 + 1152), "l"(k_src_base + data_off_2), "r"(src_size_3), "l"(0x12F0000000000000ULL));
            unsigned long long data_off_4 = (unsigned long long)entries_3[0] | (unsigned long long)entries_3[1] << 32;
            unsigned int src_size_5 = (bits_sub >> 12 & 1) << 4;
            v_src[3] = v_src_base + data_off_4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0 + 1728), "l"(k_src_base + data_off_4), "r"(src_size_5), "l"(0x12F0000000000000ULL));
            unsigned long long sf_off_1 = (unsigned long long)sf_entry[2] | (unsigned long long)sf_entry[3] << 32;
            unsigned long long sf_base = ((sf_is_v) ? v_sf_base : k_sf_base);
            int sf_side = ((sf_is_v) ? 9216 : 0);
            unsigned int sf_size = (bits_1 >> (unsigned int)sf_lane & 1) << 4;
            unsigned int sf_dst = stage_base + (unsigned int)(sf_side + sf_token * 144 + 128);
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(sf_dst), "l"(sf_base + sf_off_1), "r"(sf_size), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.commit_group;");
            int col_6 = lane_1 & 7;
            int sub_7 = lane_1 >> 3;
            int token0_8 = 16 * warp_1 + sub_7;
            unsigned int dst_v0 = ((kv_addr + (unsigned int)(9216 + token0_8 * 144 + col_6 * 16)) < (kv_addr + 36848) ? (kv_addr + (unsigned int)(9216 + token0_8 * 144 + col_6 * 16)) : (kv_addr + 36848));
            unsigned int src_size_9 = (v_bits[0] & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0), "l"(v_src[0]), "r"(src_size_9), "l"(0x12F0000000000000ULL));
            unsigned int src_size_10 = (v_bits[0] >> 4 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0 + 576), "l"(v_src[1]), "r"(src_size_10), "l"(0x12F0000000000000ULL));
            unsigned int src_size_11 = (v_bits[0] >> 8 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0 + 1152), "l"(v_src[2]), "r"(src_size_11), "l"(0x12F0000000000000ULL));
            unsigned int src_size_12 = (v_bits[0] >> 12 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0 + 1728), "l"(v_src[3]), "r"(src_size_12), "l"(0x12F0000000000000ULL));
        } else {
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.commit_group;");
        if (warp_1 < 2) {
            bool valid_0_1 = phys[1] >= 0 && phys[1] < num_blocks;
            int _max_16 = ((phys[1]) > (0) ? (phys[1]) : (0));
            unsigned long long block_1 = (unsigned long long)_max_16;
            unsigned long long local_u64_1 = (unsigned long long)local[1];
            unsigned long long data_off_3 = block_1 * data_stride_block_u64 + local_u64_1 * data_stride_token_u64;
            unsigned long long sf_off_2 = block_1 * sf_stride_block_u64 + local_u64_1 * sf_stride_token_u64;
            unsigned int w0_1_1 = (unsigned int)data_off_3;
            unsigned int w1_2_1 = (unsigned int)(data_off_3 >> 32);
            unsigned int w2_3_1 = (unsigned int)sf_off_2;
            unsigned int w3_4_1 = (unsigned int)(sf_off_2 >> 32);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(1024 + token * 16)), "r"(w0_1_1), "r"(w1_2_1), "r"(w2_3_1), "r"(w3_4_1) : "memory");
            unsigned int bits_2 = __ballot_sync(0xFFFFFFFF, valid_0_1);
            if (lane_1 == 0) {
                slot_valid[2 + warp_1] = bits_2;
            }
            idx[0] = idx[2];
        }
    } else {
        if (split < tile_end) {
            bool valid_0_2 = phys[0] >= 0 && phys[0] < num_blocks;
            int _max_17 = ((phys[0]) > (0) ? (phys[0]) : (0));
            unsigned long long block_2 = (unsigned long long)_max_17;
            unsigned long long local_u64_2 = (unsigned long long)local[0];
            bool hi = tid_0 >= 64;
            int token_1 = tid_0 & 63;
            unsigned int piece_bytes = ((hi) ? 64 : 0);
            unsigned long long data_off_5 = block_2 * data_stride_block_u64 + local_u64_2 * data_stride_token_u64 + (unsigned long long)piece_bytes;
            unsigned long long sf_off_3 = block_2 * sf_stride_block_u64 + local_u64_2 * sf_stride_token_u64;
            unsigned long long k_src = k_base + data_off_5;
            unsigned long long v_src_2 = v_base + data_off_5;
            unsigned long long sf_base_1 = ((hi) ? v_sf_base : k_sf_base);
            unsigned int row_dst = kv_addr + (unsigned int)(token_1 * 144);
            unsigned int dst_k = row_dst + piece_bytes;
            unsigned int sf_side_1 = ((hi) ? 9216 : 0);
            unsigned int sf_dst_1 = row_dst + sf_side_1 + 128;
            int nbytes = ((valid_0_2) ? 16 : 0);
            unsigned int src_size_2 = (unsigned int)nbytes;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k), "l"(k_src), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 16), "l"(k_src + 16), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 32), "l"(k_src + 32), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 48), "l"(k_src + 48), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(sf_dst_1), "l"(sf_base_1 + sf_off_3), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216), "l"(v_src_2), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216 + 16), "l"(v_src_2 + 16), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216 + 32), "l"(v_src_2 + 32), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216 + 48), "l"(v_src_2 + 48), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            unsigned int bits_3 = __ballot_sync(0xFFFFFFFF, valid_0_2);
            if (warp_1 < 2 && lane_1 == 0) {
                valid_s[warp_1] = bits_3;
            }
        } else {
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.commit_group;");
    }
    int stage = 0;
    int parity = 0;
    #pragma unroll 1
    for (int tile = split; tile < tile_end; tile += num_splits) {
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
#if __CUDA_ARCH__ == 900
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lut_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_regs[(0) + 1]))
            : "r"(lut_s_addr));
#else
#endif
        if (warp_1 < 2 && multi) {
            int column_0 = (tile + 3 * num_splits) * 64 + token;
            bool in_range_1 = column_0 < topk && tile_end > tile + 3 * num_splits;
            int safe_column_3 = ((column_0) < (topk - 1) ? (column_0) : (topk - 1));
            int logical_4 = ((in_range_1) ? indices[indices_row + (long long)safe_column_3] : -1);
            idx[1] = logical_4;
            int safe_token_5 = ((idx[0]) > (0) ? (idx[0]) : (0));
            uint32_t _fast_div_q_2 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_5), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_5);
            uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)safe_token_5) - _fast_div_q_2 * (uint32_t)(page_div.divisor);
            int page_6 = (int)_fast_div_q_2;
            bool valid_7 = request_ok && idx[0] >= 0 && page_6 < table_width;
            int table_column_8 = ((page_6) < (table_width - 1) ? (page_6) : (table_width - 1));
            int entry_9 = ((valid_7) ? block_table[safe_request_row + (long long)table_column_8] : -1);
            phys[0] = entry_9;
            local[0] = (int)_fast_div_r_2;
        }
        int issue_stage = stage + 1;
        if (issue_stage >= 2) {
            issue_stage = issue_stage - 2;
        }
        int issue_slot = parity + 1 & 1;
        bool issue_next = tile_end > tile + num_splits;
        if (issue_next) {
            unsigned int bits_4 = slot_valid[issue_slot * 2 + (warp_1 >> 1)] >> (unsigned int)(16 * (warp_1 & 1));
            if (tid_0 < 2) {
                valid_s[issue_stage * 2 + tid_0] = slot_valid[issue_slot * 2 + tid_0];
            }
            int col_1 = lane_1 & 7;
            int sub_1 = lane_1 >> 3;
            int token0_1 = 16 * warp_1 + sub_1;
            unsigned int entry_addr_1 = addr_ring_addr + (unsigned int)(issue_slot * 1024 + token0_1 * 16);
            unsigned int bits_sub_1 = bits_4 >> (unsigned int)sub_1;
            unsigned long long col_off_1 = (unsigned long long)(col_1 * 16);
            unsigned int stage_base_1 = kv_addr + (unsigned int)(issue_stage * 18432);
            unsigned int dst_k0_1 = ((stage_base_1 + (unsigned int)(token0_1 * 144 + col_1 * 16)) < (kv_addr + 36848) ? (stage_base_1 + (unsigned int)(token0_1 * 144 + col_1 * 16)) : (kv_addr + 36848));
            int sf_lane_1 = lane_1 & 15;
            bool sf_is_v_1 = lane_1 >= 16;
            int sf_token_1 = 16 * warp_1 + sf_lane_1;
            unsigned int entries_0_1[4];
            unsigned int entries_1_1[4];
            unsigned int entries_2_1[4];
            unsigned int entries_3_1[4];
            unsigned int sf_entry_1[4];
            __int128_t _smem_raw_18;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(entry_addr_1));
            entries_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
            entries_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
            entries_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
            entries_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
            __int128_t _smem_raw_19;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(entry_addr_1 + 64));
            entries_1_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
            entries_1_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
            entries_1_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
            entries_1_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
            __int128_t _smem_raw_20;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_20) : "r"(entry_addr_1 + 128));
            entries_2_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[0];
            entries_2_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[1];
            entries_2_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[2];
            entries_2_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[3];
            __int128_t _smem_raw_21;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_21) : "r"(entry_addr_1 + 192));
            entries_3_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[0];
            entries_3_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[1];
            entries_3_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[2];
            entries_3_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[3];
            __int128_t _smem_raw_22;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_22) : "r"(addr_ring_addr + (unsigned int)(issue_slot * 1024 + sf_token_1 * 16)));
            sf_entry_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[0];
            sf_entry_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[1];
            sf_entry_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[2];
            sf_entry_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[3];
            unsigned long long k_src_base_1 = k_base + col_off_1;
            unsigned long long v_src_base_1 = v_base + col_off_1;
            v_bits[0] = bits_sub_1;
            unsigned long long data_off_6 = (unsigned long long)entries_0_1[0] | (unsigned long long)entries_0_1[1] << 32;
            unsigned int src_size_4 = (bits_sub_1 & 1) << 4;
            v_src[0] = v_src_base_1 + data_off_6;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0_1), "l"(k_src_base_1 + data_off_6), "r"(src_size_4), "l"(0x12F0000000000000ULL));
            unsigned long long data_off_0_1 = (unsigned long long)entries_1_1[0] | (unsigned long long)entries_1_1[1] << 32;
            unsigned int src_size_1_1 = (bits_sub_1 >> 4 & 1) << 4;
            v_src[1] = v_src_base_1 + data_off_0_1;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0_1 + 576), "l"(k_src_base_1 + data_off_0_1), "r"(src_size_1_1), "l"(0x12F0000000000000ULL));
            unsigned long long data_off_2_1 = (unsigned long long)entries_2_1[0] | (unsigned long long)entries_2_1[1] << 32;
            unsigned int src_size_3_1 = (bits_sub_1 >> 8 & 1) << 4;
            v_src[2] = v_src_base_1 + data_off_2_1;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0_1 + 1152), "l"(k_src_base_1 + data_off_2_1), "r"(src_size_3_1), "l"(0x12F0000000000000ULL));
            unsigned long long data_off_4_1 = (unsigned long long)entries_3_1[0] | (unsigned long long)entries_3_1[1] << 32;
            unsigned int src_size_5_1 = (bits_sub_1 >> 12 & 1) << 4;
            v_src[3] = v_src_base_1 + data_off_4_1;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0_1 + 1728), "l"(k_src_base_1 + data_off_4_1), "r"(src_size_5_1), "l"(0x12F0000000000000ULL));
            unsigned long long sf_off_4 = (unsigned long long)sf_entry_1[2] | (unsigned long long)sf_entry_1[3] << 32;
            unsigned long long sf_base_2 = ((sf_is_v_1) ? v_sf_base : k_sf_base);
            int sf_side_2 = ((sf_is_v_1) ? 9216 : 0);
            unsigned int sf_size_1 = (bits_4 >> (unsigned int)sf_lane_1 & 1) << 4;
            unsigned int sf_dst_2 = stage_base_1 + (unsigned int)(sf_side_2 + sf_token_1 * 144 + 128);
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(sf_dst_2), "l"(sf_base_2 + sf_off_4), "r"(sf_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.commit_group;");
        } else {
            asm volatile("cp.async.commit_group;");
        }
        unsigned int stage_addr = kv_addr + (unsigned int)(stage * 18432);
        unsigned int k0_addr = stage_addr + (unsigned int)((16 * warp_1 + g) * 144);
        unsigned int k1_addr = k0_addr + 1152;
        unsigned int scale_col = (unsigned int)(128 + 4 * t);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_words[0])) : "r"(k0_addr + scale_col));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_words[1])) : "r"(k1_addr + scale_col));
        uint32_t _prmt_0;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_0) : "r"(sf_words[0]), "r"(sf_words[0]));
        sf0[0] = _prmt_0 & 2147516416u | (_prmt_0 & 2130738944) >> 4;
        sf0[1] = _prmt_0 << 8 & 2147516416u | (_prmt_0 << 8 & 2130738944) >> 4;
        uint32_t _prmt_1;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_1) : "r"(sf_words[1]), "r"(sf_words[1]));
        sf1[0] = _prmt_1 & 2147516416u | (_prmt_1 & 2130738944) >> 4;
        sf1[1] = _prmt_1 << 8 & 2147516416u | (_prmt_1 << 8 & 2130738944) >> 4;
        s[0] = 0.0f;
        s[1] = 0.0f;
        s[2] = 0.0f;
        s[3] = 0.0f;
        s[4] = 0.0f;
        s[5] = 0.0f;
        s[6] = 0.0f;
        s[7] = 0.0f;
        unsigned int data_col = (unsigned int)(32 * t);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 3]))
            : "r"(k0_addr + data_col));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 3]))
            : "r"(k1_addr + data_col));
        uint32_t _prmt_2;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_2) : "r"(raw0[0] & 2004318071), "r"(raw0[0] & 2004318071));
        uint32_t _prmt_3;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_3) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_3) : "r"(2071689984), "r"(2088533116), "r"(raw0[0] & 2004318071));
#endif
        uint32_t _prmt_4;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_4) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_2));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_4) : "r"(2071689984), "r"(2088533116), "r"(_prmt_2));
#endif
        uint32_t _prmt_5;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_5) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_5) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#endif
        uint32_t _prmt_6;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_6) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_2));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_6) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_2));
#endif
        uint32_t _prmt_7;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_7) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_8;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_8) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_23;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_23) : "r"(_prmt_5), "r"(_prmt_3 | _prmt_7 & 2155905152u));
        d0[0] = _prmt_23;
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_24) : "r"(_prmt_5), "r"(_prmt_3 | _prmt_7 & 2155905152u));
        d0[1] = _prmt_24;
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_25) : "r"(_prmt_6), "r"(_prmt_4 | _prmt_8 & 2155905152u));
        d0[2] = _prmt_25;
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_26) : "r"(_prmt_6), "r"(_prmt_4 | _prmt_8 & 2155905152u));
        d0[3] = _prmt_26;
        uint32_t _prmt_9;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_9) : "r"(raw1[0] & 2004318071), "r"(raw1[0] & 2004318071));
        uint32_t _prmt_10;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_10) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_10) : "r"(2071689984), "r"(2088533116), "r"(raw1[0] & 2004318071));
#endif
        uint32_t _prmt_11;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_11) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_9));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_11) : "r"(2071689984), "r"(2088533116), "r"(_prmt_9));
#endif
        uint32_t _prmt_12;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_12) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_12) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#endif
        uint32_t _prmt_13;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_13) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_9));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_13) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_9));
#endif
        uint32_t _prmt_14;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_14) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_15;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_15) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_27) : "r"(_prmt_12), "r"(_prmt_10 | _prmt_14 & 2155905152u));
        d1[0] = _prmt_27;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_28) : "r"(_prmt_12), "r"(_prmt_10 | _prmt_14 & 2155905152u));
        d1[1] = _prmt_28;
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_29) : "r"(_prmt_13), "r"(_prmt_11 | _prmt_15 & 2155905152u));
        d1[2] = _prmt_29;
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_30) : "r"(_prmt_13), "r"(_prmt_11 | _prmt_15 & 2155905152u));
        d1[3] = _prmt_30;
        uint32_t _prmt_16;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_16) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_17;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_17) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_0;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_0) : "r"(d0[0]), "r"(_prmt_16), "r"(2147516416u));
        d0[0] = _bf16x2_fma_0;
        uint32_t _bf16x2_fma_1;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_1) : "r"(d1[0]), "r"(_prmt_17), "r"(2147516416u));
        d1[0] = _bf16x2_fma_1;
        uint32_t _bf16x2_fma_2;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_2) : "r"(d0[1]), "r"(_prmt_16), "r"(2147516416u));
        d0[1] = _bf16x2_fma_2;
        uint32_t _bf16x2_fma_3;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_3) : "r"(d1[1]), "r"(_prmt_17), "r"(2147516416u));
        d1[1] = _bf16x2_fma_3;
        uint32_t _bf16x2_fma_4;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_4) : "r"(d0[2]), "r"(_prmt_16), "r"(2147516416u));
        d0[2] = _bf16x2_fma_4;
        uint32_t _bf16x2_fma_5;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_5) : "r"(d1[2]), "r"(_prmt_17), "r"(2147516416u));
        d1[2] = _bf16x2_fma_5;
        uint32_t _bf16x2_fma_6;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_6) : "r"(d0[3]), "r"(_prmt_16), "r"(2147516416u));
        d0[3] = _bf16x2_fma_6;
        uint32_t _bf16x2_fma_7;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_7) : "r"(d1[3]), "r"(_prmt_17), "r"(2147516416u));
        d1[3] = _bf16x2_fma_7;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * lane_1)));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (32 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        uint32_t _prmt_18;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_18) : "r"(raw0[1] & 2004318071), "r"(raw0[1] & 2004318071));
        uint32_t _prmt_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_19) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_19) : "r"(2071689984), "r"(2088533116), "r"(raw0[1] & 2004318071));
#endif
        uint32_t _prmt_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_20) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_18));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_20) : "r"(2071689984), "r"(2088533116), "r"(_prmt_18));
#endif
        uint32_t _prmt_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_21) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_21) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#endif
        uint32_t _prmt_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_22) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_18));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_22) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_18));
#endif
        uint32_t _prmt_23_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_23_1) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_24_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_24_1) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_31) : "r"(_prmt_21), "r"(_prmt_19 | _prmt_23_1 & 2155905152u));
        d0[0] = _prmt_31;
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_32) : "r"(_prmt_21), "r"(_prmt_19 | _prmt_23_1 & 2155905152u));
        d0[1] = _prmt_32;
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_33) : "r"(_prmt_22), "r"(_prmt_20 | _prmt_24_1 & 2155905152u));
        d0[2] = _prmt_33;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_34) : "r"(_prmt_22), "r"(_prmt_20 | _prmt_24_1 & 2155905152u));
        d0[3] = _prmt_34;
        uint32_t _prmt_25_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_25_1) : "r"(raw1[1] & 2004318071), "r"(raw1[1] & 2004318071));
        uint32_t _prmt_26_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_26_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_26_1) : "r"(2071689984), "r"(2088533116), "r"(raw1[1] & 2004318071));
#endif
        uint32_t _prmt_27_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_27_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_25_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_27_1) : "r"(2071689984), "r"(2088533116), "r"(_prmt_25_1));
#endif
        uint32_t _prmt_28_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28_1) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#endif
        uint32_t _prmt_29_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_29_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_25_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_29_1) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_25_1));
#endif
        uint32_t _prmt_30_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_30_1) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_31_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_31_1) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_35) : "r"(_prmt_28_1), "r"(_prmt_26_1 | _prmt_30_1 & 2155905152u));
        d1[0] = _prmt_35;
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_36) : "r"(_prmt_28_1), "r"(_prmt_26_1 | _prmt_30_1 & 2155905152u));
        d1[1] = _prmt_36;
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_37) : "r"(_prmt_29_1), "r"(_prmt_27_1 | _prmt_31_1 & 2155905152u));
        d1[2] = _prmt_37;
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_38) : "r"(_prmt_29_1), "r"(_prmt_27_1 | _prmt_31_1 & 2155905152u));
        d1[3] = _prmt_38;
        uint32_t _prmt_32_1;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_32_1) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_33_1;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_33_1) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_8;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_8) : "r"(d0[0]), "r"(_prmt_32_1), "r"(2147516416u));
        d0[0] = _bf16x2_fma_8;
        uint32_t _bf16x2_fma_9;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_9) : "r"(d1[0]), "r"(_prmt_33_1), "r"(2147516416u));
        d1[0] = _bf16x2_fma_9;
        uint32_t _bf16x2_fma_10;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_10) : "r"(d0[1]), "r"(_prmt_32_1), "r"(2147516416u));
        d0[1] = _bf16x2_fma_10;
        uint32_t _bf16x2_fma_11;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_11) : "r"(d1[1]), "r"(_prmt_33_1), "r"(2147516416u));
        d1[1] = _bf16x2_fma_11;
        uint32_t _bf16x2_fma_12;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_12) : "r"(d0[2]), "r"(_prmt_32_1), "r"(2147516416u));
        d0[2] = _bf16x2_fma_12;
        uint32_t _bf16x2_fma_13;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_13) : "r"(d1[2]), "r"(_prmt_33_1), "r"(2147516416u));
        d1[2] = _bf16x2_fma_13;
        uint32_t _bf16x2_fma_14;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_14) : "r"(d0[3]), "r"(_prmt_32_1), "r"(2147516416u));
        d0[3] = _bf16x2_fma_14;
        uint32_t _bf16x2_fma_15;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_15) : "r"(d1[3]), "r"(_prmt_33_1), "r"(2147516416u));
        d1[3] = _bf16x2_fma_15;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (64 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (96 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        uint32_t _prmt_34_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_34_1) : "r"(raw0[2] & 2004318071), "r"(raw0[2] & 2004318071));
        uint32_t _prmt_35_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_35_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_35_1) : "r"(2071689984), "r"(2088533116), "r"(raw0[2] & 2004318071));
#endif
        uint32_t _prmt_36_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_36_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_34_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_36_1) : "r"(2071689984), "r"(2088533116), "r"(_prmt_34_1));
#endif
        uint32_t _prmt_37_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37_1) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#endif
        uint32_t _prmt_38_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_34_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38_1) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_34_1));
#endif
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_39) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_40) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_39_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_39_1) : "r"(_prmt_37_1), "r"(_prmt_35_1 | _prmt_39 & 2155905152u));
        d0[0] = _prmt_39_1;
        uint32_t _prmt_40_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_40_1) : "r"(_prmt_37_1), "r"(_prmt_35_1 | _prmt_39 & 2155905152u));
        d0[1] = _prmt_40_1;
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_41) : "r"(_prmt_38_1), "r"(_prmt_36_1 | _prmt_40 & 2155905152u));
        d0[2] = _prmt_41;
        uint32_t _prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_42) : "r"(_prmt_38_1), "r"(_prmt_36_1 | _prmt_40 & 2155905152u));
        d0[3] = _prmt_42;
        uint32_t _prmt_41_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_41_1) : "r"(raw1[2] & 2004318071), "r"(raw1[2] & 2004318071));
        uint32_t _prmt_42_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_42_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_42_1) : "r"(2071689984), "r"(2088533116), "r"(raw1[2] & 2004318071));
#endif
        uint32_t _prmt_43;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_43) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_41_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_43) : "r"(2071689984), "r"(2088533116), "r"(_prmt_41_1));
#endif
        uint32_t _prmt_44;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_44) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_44) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#endif
        uint32_t _prmt_45;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_45) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_41_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_45) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_41_1));
#endif
        uint32_t _prmt_46;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_46) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_47;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_47) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_43_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_43_1) : "r"(_prmt_44), "r"(_prmt_42_1 | _prmt_46 & 2155905152u));
        d1[0] = _prmt_43_1;
        uint32_t _prmt_44_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_44_1) : "r"(_prmt_44), "r"(_prmt_42_1 | _prmt_46 & 2155905152u));
        d1[1] = _prmt_44_1;
        uint32_t _prmt_45_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_45_1) : "r"(_prmt_45), "r"(_prmt_43 | _prmt_47 & 2155905152u));
        d1[2] = _prmt_45_1;
        uint32_t _prmt_46_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_46_1) : "r"(_prmt_45), "r"(_prmt_43 | _prmt_47 & 2155905152u));
        d1[3] = _prmt_46_1;
        uint32_t _prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_48) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_49;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_49) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_16;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_16) : "r"(d0[0]), "r"(_prmt_48), "r"(2147516416u));
        d0[0] = _bf16x2_fma_16;
        uint32_t _bf16x2_fma_17;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_17) : "r"(d1[0]), "r"(_prmt_49), "r"(2147516416u));
        d1[0] = _bf16x2_fma_17;
        uint32_t _bf16x2_fma_18;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_18) : "r"(d0[1]), "r"(_prmt_48), "r"(2147516416u));
        d0[1] = _bf16x2_fma_18;
        uint32_t _bf16x2_fma_19;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_19) : "r"(d1[1]), "r"(_prmt_49), "r"(2147516416u));
        d1[1] = _bf16x2_fma_19;
        uint32_t _bf16x2_fma_20;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_20) : "r"(d0[2]), "r"(_prmt_48), "r"(2147516416u));
        d0[2] = _bf16x2_fma_20;
        uint32_t _bf16x2_fma_21;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_21) : "r"(d1[2]), "r"(_prmt_49), "r"(2147516416u));
        d1[2] = _bf16x2_fma_21;
        uint32_t _bf16x2_fma_22;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_22) : "r"(d0[3]), "r"(_prmt_48), "r"(2147516416u));
        d0[3] = _bf16x2_fma_22;
        uint32_t _bf16x2_fma_23;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_23) : "r"(d1[3]), "r"(_prmt_49), "r"(2147516416u));
        d1[3] = _bf16x2_fma_23;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (128 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (160 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        uint32_t _prmt_50;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_50) : "r"(raw0[3] & 2004318071), "r"(raw0[3] & 2004318071));
        uint32_t _prmt_51;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_51) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_51) : "r"(2071689984), "r"(2088533116), "r"(raw0[3] & 2004318071));
#endif
        uint32_t _prmt_52;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_52) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_50));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_52) : "r"(2071689984), "r"(2088533116), "r"(_prmt_50));
#endif
        uint32_t _prmt_53;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_53) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_53) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#endif
        uint32_t _prmt_54;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_54) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_50));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_54) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_50));
#endif
        uint32_t _prmt_55;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_55) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_56;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_56) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_47_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_47_1) : "r"(_prmt_53), "r"(_prmt_51 | _prmt_55 & 2155905152u));
        d0[0] = _prmt_47_1;
        uint32_t _prmt_48_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_48_1) : "r"(_prmt_53), "r"(_prmt_51 | _prmt_55 & 2155905152u));
        d0[1] = _prmt_48_1;
        uint32_t _prmt_49_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_49_1) : "r"(_prmt_54), "r"(_prmt_52 | _prmt_56 & 2155905152u));
        d0[2] = _prmt_49_1;
        uint32_t _prmt_50_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_50_1) : "r"(_prmt_54), "r"(_prmt_52 | _prmt_56 & 2155905152u));
        d0[3] = _prmt_50_1;
        uint32_t _prmt_57;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_57) : "r"(raw1[3] & 2004318071), "r"(raw1[3] & 2004318071));
        uint32_t _prmt_58;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_58) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_58) : "r"(2071689984), "r"(2088533116), "r"(raw1[3] & 2004318071));
#endif
        uint32_t _prmt_59;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_59) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_57));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_59) : "r"(2071689984), "r"(2088533116), "r"(_prmt_57));
#endif
        uint32_t _prmt_60;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_60) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_60) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#endif
        uint32_t _prmt_61;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_61) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_57));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_61) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_57));
#endif
        uint32_t _prmt_62;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_62) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_63;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_63) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_51_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_51_1) : "r"(_prmt_60), "r"(_prmt_58 | _prmt_62 & 2155905152u));
        d1[0] = _prmt_51_1;
        uint32_t _prmt_52_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_52_1) : "r"(_prmt_60), "r"(_prmt_58 | _prmt_62 & 2155905152u));
        d1[1] = _prmt_52_1;
        uint32_t _prmt_53_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_53_1) : "r"(_prmt_61), "r"(_prmt_59 | _prmt_63 & 2155905152u));
        d1[2] = _prmt_53_1;
        uint32_t _prmt_54_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_54_1) : "r"(_prmt_61), "r"(_prmt_59 | _prmt_63 & 2155905152u));
        d1[3] = _prmt_54_1;
        uint32_t _prmt_64;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_64) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_65;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_65) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_24;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_24) : "r"(d0[0]), "r"(_prmt_64), "r"(2147516416u));
        d0[0] = _bf16x2_fma_24;
        uint32_t _bf16x2_fma_25;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_25) : "r"(d1[0]), "r"(_prmt_65), "r"(2147516416u));
        d1[0] = _bf16x2_fma_25;
        uint32_t _bf16x2_fma_26;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_26) : "r"(d0[1]), "r"(_prmt_64), "r"(2147516416u));
        d0[1] = _bf16x2_fma_26;
        uint32_t _bf16x2_fma_27;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_27) : "r"(d1[1]), "r"(_prmt_65), "r"(2147516416u));
        d1[1] = _bf16x2_fma_27;
        uint32_t _bf16x2_fma_28;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_28) : "r"(d0[2]), "r"(_prmt_64), "r"(2147516416u));
        d0[2] = _bf16x2_fma_28;
        uint32_t _bf16x2_fma_29;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_29) : "r"(d1[2]), "r"(_prmt_65), "r"(2147516416u));
        d1[2] = _bf16x2_fma_29;
        uint32_t _bf16x2_fma_30;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_30) : "r"(d0[3]), "r"(_prmt_64), "r"(2147516416u));
        d0[3] = _bf16x2_fma_30;
        uint32_t _bf16x2_fma_31;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_31) : "r"(d1[3]), "r"(_prmt_65), "r"(2147516416u));
        d1[3] = _bf16x2_fma_31;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (192 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (224 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        unsigned int data_col_0 = (unsigned int)(32 * t + 16);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 3]))
            : "r"(k0_addr + data_col_0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 3]))
            : "r"(k1_addr + data_col_0));
        uint32_t _prmt_66;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_66) : "r"(raw0[0] & 2004318071), "r"(raw0[0] & 2004318071));
        uint32_t _prmt_67;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_67) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_67) : "r"(2071689984), "r"(2088533116), "r"(raw0[0] & 2004318071));
#endif
        uint32_t _prmt_68;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_68) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_66));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_68) : "r"(2071689984), "r"(2088533116), "r"(_prmt_66));
#endif
        uint32_t _prmt_69;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_69) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_69) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#endif
        uint32_t _prmt_70;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_70) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_66));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_70) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_66));
#endif
        uint32_t _prmt_71;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_71) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_72;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_72) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_55_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_55_1) : "r"(_prmt_69), "r"(_prmt_67 | _prmt_71 & 2155905152u));
        d0[0] = _prmt_55_1;
        uint32_t _prmt_56_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_56_1) : "r"(_prmt_69), "r"(_prmt_67 | _prmt_71 & 2155905152u));
        d0[1] = _prmt_56_1;
        uint32_t _prmt_57_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_57_1) : "r"(_prmt_70), "r"(_prmt_68 | _prmt_72 & 2155905152u));
        d0[2] = _prmt_57_1;
        uint32_t _prmt_58_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_58_1) : "r"(_prmt_70), "r"(_prmt_68 | _prmt_72 & 2155905152u));
        d0[3] = _prmt_58_1;
        uint32_t _prmt_73;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_73) : "r"(raw1[0] & 2004318071), "r"(raw1[0] & 2004318071));
        uint32_t _prmt_74;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(2071689984), "r"(2088533116), "r"(raw1[0] & 2004318071));
#endif
        uint32_t _prmt_75;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_73));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(2071689984), "r"(2088533116), "r"(_prmt_73));
#endif
        uint32_t _prmt_76;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_76) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_76) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#endif
        uint32_t _prmt_77;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_73));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_73));
#endif
        uint32_t _prmt_78;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_78) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_79;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_79) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_59_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_59_1) : "r"(_prmt_76), "r"(_prmt_74 | _prmt_78 & 2155905152u));
        d1[0] = _prmt_59_1;
        uint32_t _prmt_60_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_60_1) : "r"(_prmt_76), "r"(_prmt_74 | _prmt_78 & 2155905152u));
        d1[1] = _prmt_60_1;
        uint32_t _prmt_61_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_61_1) : "r"(_prmt_77), "r"(_prmt_75 | _prmt_79 & 2155905152u));
        d1[2] = _prmt_61_1;
        uint32_t _prmt_62_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_62_1) : "r"(_prmt_77), "r"(_prmt_75 | _prmt_79 & 2155905152u));
        d1[3] = _prmt_62_1;
        uint32_t _prmt_80;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_80) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_81;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_81) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_32;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_32) : "r"(d0[0]), "r"(_prmt_80), "r"(2147516416u));
        d0[0] = _bf16x2_fma_32;
        uint32_t _bf16x2_fma_33;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_33) : "r"(d1[0]), "r"(_prmt_81), "r"(2147516416u));
        d1[0] = _bf16x2_fma_33;
        uint32_t _bf16x2_fma_34;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_34) : "r"(d0[1]), "r"(_prmt_80), "r"(2147516416u));
        d0[1] = _bf16x2_fma_34;
        uint32_t _bf16x2_fma_35;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_35) : "r"(d1[1]), "r"(_prmt_81), "r"(2147516416u));
        d1[1] = _bf16x2_fma_35;
        uint32_t _bf16x2_fma_36;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_36) : "r"(d0[2]), "r"(_prmt_80), "r"(2147516416u));
        d0[2] = _bf16x2_fma_36;
        uint32_t _bf16x2_fma_37;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_37) : "r"(d1[2]), "r"(_prmt_81), "r"(2147516416u));
        d1[2] = _bf16x2_fma_37;
        uint32_t _bf16x2_fma_38;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_38) : "r"(d0[3]), "r"(_prmt_80), "r"(2147516416u));
        d0[3] = _bf16x2_fma_38;
        uint32_t _bf16x2_fma_39;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_39) : "r"(d1[3]), "r"(_prmt_81), "r"(2147516416u));
        d1[3] = _bf16x2_fma_39;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (256 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (288 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_82) : "r"(raw0[1] & 2004318071), "r"(raw0[1] & 2004318071));
        uint32_t _prmt_83;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_83) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_83) : "r"(2071689984), "r"(2088533116), "r"(raw0[1] & 2004318071));
#endif
        uint32_t _prmt_84;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_84) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_82));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_84) : "r"(2071689984), "r"(2088533116), "r"(_prmt_82));
#endif
        uint32_t _prmt_85;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_85) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_85) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#endif
        uint32_t _prmt_86;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_86) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_82));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_86) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_82));
#endif
        uint32_t _prmt_87;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_87) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_88;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_88) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_63_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_63_1) : "r"(_prmt_85), "r"(_prmt_83 | _prmt_87 & 2155905152u));
        d0[0] = _prmt_63_1;
        uint32_t _prmt_64_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_64_1) : "r"(_prmt_85), "r"(_prmt_83 | _prmt_87 & 2155905152u));
        d0[1] = _prmt_64_1;
        uint32_t _prmt_65_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_65_1) : "r"(_prmt_86), "r"(_prmt_84 | _prmt_88 & 2155905152u));
        d0[2] = _prmt_65_1;
        uint32_t _prmt_66_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_66_1) : "r"(_prmt_86), "r"(_prmt_84 | _prmt_88 & 2155905152u));
        d0[3] = _prmt_66_1;
        uint32_t _prmt_89;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_89) : "r"(raw1[1] & 2004318071), "r"(raw1[1] & 2004318071));
        uint32_t _prmt_90;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_90) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_90) : "r"(2071689984), "r"(2088533116), "r"(raw1[1] & 2004318071));
#endif
        uint32_t _prmt_91;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_91) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_89));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_91) : "r"(2071689984), "r"(2088533116), "r"(_prmt_89));
#endif
        uint32_t _prmt_92;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_92) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_92) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#endif
        uint32_t _prmt_93;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_93) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_89));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_93) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_89));
#endif
        uint32_t _prmt_94;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_94) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_95;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_95) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_67_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_67_1) : "r"(_prmt_92), "r"(_prmt_90 | _prmt_94 & 2155905152u));
        d1[0] = _prmt_67_1;
        uint32_t _prmt_68_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_68_1) : "r"(_prmt_92), "r"(_prmt_90 | _prmt_94 & 2155905152u));
        d1[1] = _prmt_68_1;
        uint32_t _prmt_69_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_69_1) : "r"(_prmt_93), "r"(_prmt_91 | _prmt_95 & 2155905152u));
        d1[2] = _prmt_69_1;
        uint32_t _prmt_70_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_70_1) : "r"(_prmt_93), "r"(_prmt_91 | _prmt_95 & 2155905152u));
        d1[3] = _prmt_70_1;
        uint32_t _prmt_96;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_96) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_97;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_97) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_40;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_40) : "r"(d0[0]), "r"(_prmt_96), "r"(2147516416u));
        d0[0] = _bf16x2_fma_40;
        uint32_t _bf16x2_fma_41;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_41) : "r"(d1[0]), "r"(_prmt_97), "r"(2147516416u));
        d1[0] = _bf16x2_fma_41;
        uint32_t _bf16x2_fma_42;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_42) : "r"(d0[1]), "r"(_prmt_96), "r"(2147516416u));
        d0[1] = _bf16x2_fma_42;
        uint32_t _bf16x2_fma_43;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_43) : "r"(d1[1]), "r"(_prmt_97), "r"(2147516416u));
        d1[1] = _bf16x2_fma_43;
        uint32_t _bf16x2_fma_44;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_44) : "r"(d0[2]), "r"(_prmt_96), "r"(2147516416u));
        d0[2] = _bf16x2_fma_44;
        uint32_t _bf16x2_fma_45;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_45) : "r"(d1[2]), "r"(_prmt_97), "r"(2147516416u));
        d1[2] = _bf16x2_fma_45;
        uint32_t _bf16x2_fma_46;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_46) : "r"(d0[3]), "r"(_prmt_96), "r"(2147516416u));
        d0[3] = _bf16x2_fma_46;
        uint32_t _bf16x2_fma_47;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_47) : "r"(d1[3]), "r"(_prmt_97), "r"(2147516416u));
        d1[3] = _bf16x2_fma_47;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (320 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (352 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        uint32_t _prmt_98;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_98) : "r"(raw0[2] & 2004318071), "r"(raw0[2] & 2004318071));
        uint32_t _prmt_99;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_99) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_99) : "r"(2071689984), "r"(2088533116), "r"(raw0[2] & 2004318071));
#endif
        uint32_t _prmt_100;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_100) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_98));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_100) : "r"(2071689984), "r"(2088533116), "r"(_prmt_98));
#endif
        uint32_t _prmt_101;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_101) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_101) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#endif
        uint32_t _prmt_102;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_102) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_98));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_102) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_98));
#endif
        uint32_t _prmt_103;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_103) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_104;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_104) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_71_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_71_1) : "r"(_prmt_101), "r"(_prmt_99 | _prmt_103 & 2155905152u));
        d0[0] = _prmt_71_1;
        uint32_t _prmt_72_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_72_1) : "r"(_prmt_101), "r"(_prmt_99 | _prmt_103 & 2155905152u));
        d0[1] = _prmt_72_1;
        uint32_t _prmt_73_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_73_1) : "r"(_prmt_102), "r"(_prmt_100 | _prmt_104 & 2155905152u));
        d0[2] = _prmt_73_1;
        uint32_t _prmt_74_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_74_1) : "r"(_prmt_102), "r"(_prmt_100 | _prmt_104 & 2155905152u));
        d0[3] = _prmt_74_1;
        uint32_t _prmt_105;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_105) : "r"(raw1[2] & 2004318071), "r"(raw1[2] & 2004318071));
        uint32_t _prmt_106;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_106) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_106) : "r"(2071689984), "r"(2088533116), "r"(raw1[2] & 2004318071));
#endif
        uint32_t _prmt_107;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_107) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_105));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_107) : "r"(2071689984), "r"(2088533116), "r"(_prmt_105));
#endif
        uint32_t _prmt_108;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_108) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_108) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#endif
        uint32_t _prmt_109;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_109) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_105));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_109) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_105));
#endif
        uint32_t _prmt_110;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_110) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_111;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_111) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_75_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_75_1) : "r"(_prmt_108), "r"(_prmt_106 | _prmt_110 & 2155905152u));
        d1[0] = _prmt_75_1;
        uint32_t _prmt_76_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_76_1) : "r"(_prmt_108), "r"(_prmt_106 | _prmt_110 & 2155905152u));
        d1[1] = _prmt_76_1;
        uint32_t _prmt_77_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_77_1) : "r"(_prmt_109), "r"(_prmt_107 | _prmt_111 & 2155905152u));
        d1[2] = _prmt_77_1;
        uint32_t _prmt_78_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_78_1) : "r"(_prmt_109), "r"(_prmt_107 | _prmt_111 & 2155905152u));
        d1[3] = _prmt_78_1;
        uint32_t _prmt_112;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_112) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_113;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_113) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_48;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_48) : "r"(d0[0]), "r"(_prmt_112), "r"(2147516416u));
        d0[0] = _bf16x2_fma_48;
        uint32_t _bf16x2_fma_49;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_49) : "r"(d1[0]), "r"(_prmt_113), "r"(2147516416u));
        d1[0] = _bf16x2_fma_49;
        uint32_t _bf16x2_fma_50;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_50) : "r"(d0[1]), "r"(_prmt_112), "r"(2147516416u));
        d0[1] = _bf16x2_fma_50;
        uint32_t _bf16x2_fma_51;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_51) : "r"(d1[1]), "r"(_prmt_113), "r"(2147516416u));
        d1[1] = _bf16x2_fma_51;
        uint32_t _bf16x2_fma_52;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_52) : "r"(d0[2]), "r"(_prmt_112), "r"(2147516416u));
        d0[2] = _bf16x2_fma_52;
        uint32_t _bf16x2_fma_53;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_53) : "r"(d1[2]), "r"(_prmt_113), "r"(2147516416u));
        d1[2] = _bf16x2_fma_53;
        uint32_t _bf16x2_fma_54;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_54) : "r"(d0[3]), "r"(_prmt_112), "r"(2147516416u));
        d0[3] = _bf16x2_fma_54;
        uint32_t _bf16x2_fma_55;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_55) : "r"(d1[3]), "r"(_prmt_113), "r"(2147516416u));
        d1[3] = _bf16x2_fma_55;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (384 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (416 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        uint32_t _prmt_114;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_114) : "r"(raw0[3] & 2004318071), "r"(raw0[3] & 2004318071));
        uint32_t _prmt_115;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_115) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_115) : "r"(2071689984), "r"(2088533116), "r"(raw0[3] & 2004318071));
#endif
        uint32_t _prmt_116;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_116) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_114));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_116) : "r"(2071689984), "r"(2088533116), "r"(_prmt_114));
#endif
        uint32_t _prmt_117;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_117) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_117) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#endif
        uint32_t _prmt_118;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_118) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_114));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_118) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_114));
#endif
        uint32_t _prmt_119;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_119) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_120;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_120) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_79_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_79_1) : "r"(_prmt_117), "r"(_prmt_115 | _prmt_119 & 2155905152u));
        d0[0] = _prmt_79_1;
        uint32_t _prmt_80_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_80_1) : "r"(_prmt_117), "r"(_prmt_115 | _prmt_119 & 2155905152u));
        d0[1] = _prmt_80_1;
        uint32_t _prmt_81_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_81_1) : "r"(_prmt_118), "r"(_prmt_116 | _prmt_120 & 2155905152u));
        d0[2] = _prmt_81_1;
        uint32_t _prmt_82_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_82_1) : "r"(_prmt_118), "r"(_prmt_116 | _prmt_120 & 2155905152u));
        d0[3] = _prmt_82_1;
        uint32_t _prmt_121;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_121) : "r"(raw1[3] & 2004318071), "r"(raw1[3] & 2004318071));
        uint32_t _prmt_122;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_122) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_122) : "r"(2071689984), "r"(2088533116), "r"(raw1[3] & 2004318071));
#endif
        uint32_t _prmt_123;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_123) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_121));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_123) : "r"(2071689984), "r"(2088533116), "r"(_prmt_121));
#endif
        uint32_t _prmt_124;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_124) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_124) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#endif
        uint32_t _prmt_125;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_125) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_121));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_125) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_121));
#endif
        uint32_t _prmt_126;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_126) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_127;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_127) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_83_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_83_1) : "r"(_prmt_124), "r"(_prmt_122 | _prmt_126 & 2155905152u));
        d1[0] = _prmt_83_1;
        uint32_t _prmt_84_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_84_1) : "r"(_prmt_124), "r"(_prmt_122 | _prmt_126 & 2155905152u));
        d1[1] = _prmt_84_1;
        uint32_t _prmt_85_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_85_1) : "r"(_prmt_125), "r"(_prmt_123 | _prmt_127 & 2155905152u));
        d1[2] = _prmt_85_1;
        uint32_t _prmt_86_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_86_1) : "r"(_prmt_125), "r"(_prmt_123 | _prmt_127 & 2155905152u));
        d1[3] = _prmt_86_1;
        uint32_t _prmt_128;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_128) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_129;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_129) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_56;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_56) : "r"(d0[0]), "r"(_prmt_128), "r"(2147516416u));
        d0[0] = _bf16x2_fma_56;
        uint32_t _bf16x2_fma_57;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_57) : "r"(d1[0]), "r"(_prmt_129), "r"(2147516416u));
        d1[0] = _bf16x2_fma_57;
        uint32_t _bf16x2_fma_58;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_58) : "r"(d0[1]), "r"(_prmt_128), "r"(2147516416u));
        d0[1] = _bf16x2_fma_58;
        uint32_t _bf16x2_fma_59;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_59) : "r"(d1[1]), "r"(_prmt_129), "r"(2147516416u));
        d1[1] = _bf16x2_fma_59;
        uint32_t _bf16x2_fma_60;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_60) : "r"(d0[2]), "r"(_prmt_128), "r"(2147516416u));
        d0[2] = _bf16x2_fma_60;
        uint32_t _bf16x2_fma_61;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_61) : "r"(d1[2]), "r"(_prmt_129), "r"(2147516416u));
        d1[2] = _bf16x2_fma_61;
        uint32_t _bf16x2_fma_62;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_62) : "r"(d0[3]), "r"(_prmt_128), "r"(2147516416u));
        d0[3] = _bf16x2_fma_62;
        uint32_t _bf16x2_fma_63;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_63) : "r"(d1[3]), "r"(_prmt_129), "r"(2147516416u));
        d1[3] = _bf16x2_fma_63;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (448 + lane_1))));
        b_lo[0] = d0[0];
        b_lo[1] = d0[1];
        b_hi[0] = d1[0];
        b_hi[1] = d1[1];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&a_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&a_frag[(0) + 3]))
            : "r"(q_frag_addr + (unsigned int)(16 * (480 + lane_1))));
        b_lo[0] = d0[2];
        b_lo[1] = d0[3];
        b_hi[0] = d1[2];
        b_hi[1] = d1[3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_lo[0]), "r"(b_lo[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_hi[0]), "r"(b_hi[1]));
        unsigned int valid_bits = valid_s[stage * 2 + (warp_1 >> 1)] >> (unsigned int)(16 * (warp_1 & 1));
        row_max[0] = -1e+20f;
        row_max[1] = -1e+20f;
        ok[0] = valid_bits >> (unsigned int)(2 * t) & 1;
        s[0] = ((ok[0] != 0) ? s[0] * score_scale : -1e+20f);
        row_max[0] = fmaxf(row_max[0], s[0]);
        ok[1] = valid_bits >> (unsigned int)(2 * t + 1) & 1;
        s[1] = ((ok[1] != 0) ? s[1] * score_scale : -1e+20f);
        row_max[0] = fmaxf(row_max[0], s[1]);
        ok[2] = valid_bits >> (unsigned int)(2 * t) & 1;
        s[2] = ((ok[2] != 0) ? s[2] * score_scale : -1e+20f);
        row_max[1] = fmaxf(row_max[1], s[2]);
        ok[3] = valid_bits >> (unsigned int)(2 * t + 1) & 1;
        s[3] = ((ok[3] != 0) ? s[3] * score_scale : -1e+20f);
        row_max[1] = fmaxf(row_max[1], s[3]);
        ok[4] = valid_bits >> (unsigned int)(8 + 2 * t) & 1;
        s[4] = ((ok[4] != 0) ? s[4] * score_scale : -1e+20f);
        row_max[0] = fmaxf(row_max[0], s[4]);
        ok[5] = valid_bits >> (unsigned int)(8 + 2 * t + 1) & 1;
        s[5] = ((ok[5] != 0) ? s[5] * score_scale : -1e+20f);
        row_max[0] = fmaxf(row_max[0], s[5]);
        ok[6] = valid_bits >> (unsigned int)(8 + 2 * t) & 1;
        s[6] = ((ok[6] != 0) ? s[6] * score_scale : -1e+20f);
        row_max[1] = fmaxf(row_max[1], s[6]);
        ok[7] = valid_bits >> (unsigned int)(8 + 2 * t + 1) & 1;
        s[7] = ((ok[7] != 0) ? s[7] * score_scale : -1e+20f);
        row_max[1] = fmaxf(row_max[1], s[7]);
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, row_max[0], 1);
        row_max[0] = fmaxf(row_max[0], _shfl_xor_sync_87);
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, row_max[0], 2);
        row_max[0] = fmaxf(row_max[0], _shfl_xor_sync_88);
        if (t == 0) {
            row_max_s[warp_1 * 16 + g] = row_max[0];
        }
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, row_max[1], 1);
        row_max[1] = fmaxf(row_max[1], _shfl_xor_sync_89);
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, row_max[1], 2);
        row_max[1] = fmaxf(row_max[1], _shfl_xor_sync_90);
        if (t == 0) {
            row_max_s[warp_1 * 16 + g + 8] = row_max[1];
        }
        __syncthreads();
        if (issue_next) {
            int col_2 = lane_1 & 7;
            int sub_2 = lane_1 >> 3;
            int token0_2 = 16 * warp_1 + sub_2;
            unsigned int dst_v0_1 = ((kv_addr + (unsigned int)(issue_stage * 18432 + 9216 + token0_2 * 144 + col_2 * 16)) < (kv_addr + 36848) ? (kv_addr + (unsigned int)(issue_stage * 18432 + 9216 + token0_2 * 144 + col_2 * 16)) : (kv_addr + 36848));
            unsigned int src_size_6 = (v_bits[0] & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1), "l"(v_src[0]), "r"(src_size_6), "l"(0x12F0000000000000ULL));
            unsigned int src_size_0 = (v_bits[0] >> 4 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1 + 576), "l"(v_src[1]), "r"(src_size_0), "l"(0x12F0000000000000ULL));
            unsigned int src_size_1_2 = (v_bits[0] >> 8 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1 + 1152), "l"(v_src[2]), "r"(src_size_1_2), "l"(0x12F0000000000000ULL));
            unsigned int src_size_2_1 = (v_bits[0] >> 12 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1 + 1728), "l"(v_src[3]), "r"(src_size_2_1), "l"(0x12F0000000000000ULL));
        }
        asm volatile("cp.async.commit_group;");
        m_new[0] = m[0];
        m_new[0] = fmaxf(m_new[0], row_max_s[g]);
        m_new[0] = fmaxf(m_new[0], row_max_s[16 + g]);
        m_new[0] = fmaxf(m_new[0], row_max_s[32 + g]);
        m_new[0] = fmaxf(m_new[0], row_max_s[48 + g]);
        row_sum[0] = 0.0f;
        m_new[1] = m[1];
        m_new[1] = fmaxf(m_new[1], row_max_s[g + 8]);
        m_new[1] = fmaxf(m_new[1], row_max_s[16 + g + 8]);
        m_new[1] = fmaxf(m_new[1], row_max_s[32 + g + 8]);
        m_new[1] = fmaxf(m_new[1], row_max_s[48 + g + 8]);
        row_sum[1] = 0.0f;
        float _exp2_91 = approx_exp2(s[0] - m_new[0]);
        s[0] = ((ok[0] != 0) ? _exp2_91 : 0.0f);
        row_sum[0] = row_sum[0] + s[0];
        float _exp2_92 = approx_exp2(s[1] - m_new[0]);
        s[1] = ((ok[1] != 0) ? _exp2_92 : 0.0f);
        row_sum[0] = row_sum[0] + s[1];
        float _exp2_93 = approx_exp2(s[2] - m_new[1]);
        s[2] = ((ok[2] != 0) ? _exp2_93 : 0.0f);
        row_sum[1] = row_sum[1] + s[2];
        float _exp2_94 = approx_exp2(s[3] - m_new[1]);
        s[3] = ((ok[3] != 0) ? _exp2_94 : 0.0f);
        row_sum[1] = row_sum[1] + s[3];
        float _exp2_95 = approx_exp2(s[4] - m_new[0]);
        s[4] = ((ok[4] != 0) ? _exp2_95 : 0.0f);
        row_sum[0] = row_sum[0] + s[4];
        float _exp2_96 = approx_exp2(s[5] - m_new[0]);
        s[5] = ((ok[5] != 0) ? _exp2_96 : 0.0f);
        row_sum[0] = row_sum[0] + s[5];
        float _exp2_97 = approx_exp2(s[6] - m_new[1]);
        s[6] = ((ok[6] != 0) ? _exp2_97 : 0.0f);
        row_sum[1] = row_sum[1] + s[6];
        float _exp2_98 = approx_exp2(s[7] - m_new[1]);
        s[7] = ((ok[7] != 0) ? _exp2_98 : 0.0f);
        row_sum[1] = row_sum[1] + s[7];
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 1);
        row_sum[0] = row_sum[0] + _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 2);
        row_sum[0] = row_sum[0] + _shfl_xor_sync_100;
        if (t == 0) {
            row_sum_s[warp_1 * 16 + g] = row_sum[0];
        }
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 1);
        row_sum[1] = row_sum[1] + _shfl_xor_sync_101;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 2);
        row_sum[1] = row_sum[1] + _shfl_xor_sync_102;
        if (t == 0) {
            row_sum_s[warp_1 * 16 + g + 8] = row_sum[1];
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(p_frag_addr + (unsigned int)(16 * (32 * warp_1 + lane_1))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[0], s[1])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[2], s[3])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[4], s[5])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[6], s[7])))) : "memory");
        asm volatile("cp.async.wait_group 2;");
        __syncthreads();
        float alpha = approx_exp2(m[0] - m_new[0]);
        float tile_sum = 0.0f;
        tile_sum = tile_sum + row_sum_s[g];
        tile_sum = tile_sum + row_sum_s[16 + g];
        tile_sum = tile_sum + row_sum_s[32 + g];
        tile_sum = tile_sum + row_sum_s[48 + g];
        l[0] = __fmaf_rn(l[0], alpha, tile_sum);
        m[0] = m_new[0];
        o[0] = o[0] * alpha;
        o[1] = o[1] * alpha;
        o[4] = o[4] * alpha;
        o[5] = o[5] * alpha;
        o[8] = o[8] * alpha;
        o[9] = o[9] * alpha;
        o[12] = o[12] * alpha;
        o[13] = o[13] * alpha;
        o[16] = o[16] * alpha;
        o[17] = o[17] * alpha;
        o[20] = o[20] * alpha;
        o[21] = o[21] * alpha;
        o[24] = o[24] * alpha;
        o[25] = o[25] * alpha;
        o[28] = o[28] * alpha;
        o[29] = o[29] * alpha;
        float alpha_1 = approx_exp2(m[1] - m_new[1]);
        float tile_sum_2 = 0.0f;
        tile_sum_2 = tile_sum_2 + row_sum_s[g + 8];
        tile_sum_2 = tile_sum_2 + row_sum_s[16 + g + 8];
        tile_sum_2 = tile_sum_2 + row_sum_s[32 + g + 8];
        tile_sum_2 = tile_sum_2 + row_sum_s[48 + g + 8];
        l[1] = __fmaf_rn(l[1], alpha_1, tile_sum_2);
        m[1] = m_new[1];
        o[2] = o[2] * alpha_1;
        o[3] = o[3] * alpha_1;
        o[6] = o[6] * alpha_1;
        o[7] = o[7] * alpha_1;
        o[10] = o[10] * alpha_1;
        o[11] = o[11] * alpha_1;
        o[14] = o[14] * alpha_1;
        o[15] = o[15] * alpha_1;
        o[18] = o[18] * alpha_1;
        o[19] = o[19] * alpha_1;
        o[22] = o[22] * alpha_1;
        o[23] = o[23] * alpha_1;
        o[26] = o[26] * alpha_1;
        o[27] = o[27] * alpha_1;
        o[30] = o[30] * alpha_1;
        o[31] = o[31] * alpha_1;
        int v_stage = stage * 18432 + 9216;
        int v_col = 32 * warp_1 + 4 * g;
        int v_scale_col = 128 + 4 * warp_1 + (g >> 1);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&p_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 3]))
            : "r"(p_frag_addr + (unsigned int)(16 * lane_1)));
#if __CUDA_ARCH__ == 900
#else
        unsigned int sf_packed = 0;
#endif
        int v_row = v_stage + 2 * t * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row + v_col)));
        unsigned int scale_byte = (unsigned int)kv_u8[v_row + v_scale_col];
#if __CUDA_ARCH__ == 900
#else
        sf_packed = sf_packed | scale_byte;
#endif
        int v_row_3 = v_stage + (2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_3 + v_col)));
        unsigned int scale_byte_4 = (unsigned int)kv_u8[v_row_3 + v_scale_col];
#if __CUDA_ARCH__ == 900
#else
        sf_packed = sf_packed | scale_byte_4 << 8;
#endif
        int v_row_5 = v_stage + (2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_5 + v_col)));
        unsigned int scale_byte_6 = (unsigned int)kv_u8[v_row_5 + v_scale_col];
#if __CUDA_ARCH__ == 900
#else
        sf_packed = sf_packed | scale_byte_6 << 16;
#endif
        int v_row_7 = v_stage + (2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_7 + v_col)));
        unsigned int scale_byte_8 = (unsigned int)kv_u8[v_row_7 + v_scale_col];
#if __CUDA_ARCH__ == 900
#else
        sf_packed = sf_packed | scale_byte_8 << 24;
#endif
        uint32_t _prmt_130;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_130) : "r"(scale_byte), "r"(scale_byte_4));
#else
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_130) : "r"(sf_packed), "r"(sf_packed));
#endif
        sfv[0] = _prmt_130 & 2147516416u | (_prmt_130 & 2130738944) >> 4;
#if __CUDA_ARCH__ == 900
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_131) : "r"(scale_byte_6), "r"(scale_byte_8));
        sfv[1] = _prmt_131 & 2147516416u | (_prmt_131 & 2130738944) >> 4;
#else
        sfv[1] = _prmt_130 << 8 & 2147516416u | (_prmt_130 << 8 & 2130738944) >> 4;
#endif
        uint32_t w_lo;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi) : "r"(vw[0]), "r"(vw[1]));
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_131) : "r"(w_lo & 2004318071), "r"(w_lo & 2004318071));
#endif
        uint32_t _prmt_132;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_132) : "r"(w_lo & 2004318071), "r"(w_lo & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_132) : "r"(2071689984), "r"(2088533116), "r"(w_lo & 2004318071));
#endif
        uint32_t _prmt_133;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_133) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_133) : "r"(2071689984), "r"(2088533116), "r"(_prmt_131));
#endif
        uint32_t _prmt_134;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_132));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo & 2004318071));
#endif
        uint32_t _prmt_135;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_135) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_135) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_131));
#endif
        uint32_t _prmt_136;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_136) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_132));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_136) : "r"(w_lo << 4), "r"(w_lo));
#endif
        uint32_t _prmt_137;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_137) : "r"(w_lo << 4), "r"(w_lo));
        uint32_t _prmt_138;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_138) : "r"(w_lo << 4), "r"(w_lo));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_137) : "r"(w_lo << 4), "r"(w_lo));
#endif
        uint32_t _prmt_103_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_103_1) : "r"(_prmt_135), "r"(_prmt_133 | _prmt_137 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_103_1) : "r"(_prmt_134), "r"(_prmt_132 | _prmt_136 & 2155905152u));
#endif
        dp_lo[0] = _prmt_103_1;
        uint32_t _prmt_104_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_104_1) : "r"(_prmt_135), "r"(_prmt_133 | _prmt_137 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_104_1) : "r"(_prmt_134), "r"(_prmt_132 | _prmt_136 & 2155905152u));
#endif
        dp_lo[1] = _prmt_104_1;
        uint32_t _prmt_105_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_105_1) : "r"(_prmt_136), "r"(_prmt_134 | _prmt_138 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_105_1) : "r"(_prmt_135), "r"(_prmt_133 | _prmt_137 & 2155905152u));
#endif
        dp_lo[2] = _prmt_105_1;
        uint32_t _prmt_106_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_106_1) : "r"(_prmt_136), "r"(_prmt_134 | _prmt_138 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_106_1) : "r"(_prmt_135), "r"(_prmt_133 | _prmt_137 & 2155905152u));
#endif
        dp_lo[3] = _prmt_106_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_138;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_138) : "r"(w_hi & 2004318071), "r"(w_hi & 2004318071));
#endif
        uint32_t _prmt_139;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_139) : "r"(w_hi & 2004318071), "r"(w_hi & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_139) : "r"(2071689984), "r"(2088533116), "r"(w_hi & 2004318071));
#endif
        uint32_t _prmt_140;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_140) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_140) : "r"(2071689984), "r"(2088533116), "r"(_prmt_138));
#endif
        uint32_t _prmt_141;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_141) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_139));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_141) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi & 2004318071));
#endif
        uint32_t _prmt_142;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_142) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_142) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_138));
#endif
        uint32_t _prmt_143;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_143) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_139));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_143) : "r"(w_hi << 4), "r"(w_hi));
#endif
        uint32_t _prmt_144;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_144) : "r"(w_hi << 4), "r"(w_hi));
        uint32_t _prmt_145;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_145) : "r"(w_hi << 4), "r"(w_hi));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_144) : "r"(w_hi << 4), "r"(w_hi));
#endif
        uint32_t _prmt_107_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_107_1) : "r"(_prmt_142), "r"(_prmt_140 | _prmt_144 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_107_1) : "r"(_prmt_141), "r"(_prmt_139 | _prmt_143 & 2155905152u));
#endif
        dp_hi[0] = _prmt_107_1;
        uint32_t _prmt_108_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_108_1) : "r"(_prmt_142), "r"(_prmt_140 | _prmt_144 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_108_1) : "r"(_prmt_141), "r"(_prmt_139 | _prmt_143 & 2155905152u));
#endif
        dp_hi[1] = _prmt_108_1;
        uint32_t _prmt_109_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_109_1) : "r"(_prmt_143), "r"(_prmt_141 | _prmt_145 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_109_1) : "r"(_prmt_142), "r"(_prmt_140 | _prmt_144 & 2155905152u));
#endif
        dp_hi[2] = _prmt_109_1;
        uint32_t _prmt_110_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_110_1) : "r"(_prmt_143), "r"(_prmt_141 | _prmt_145 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_110_1) : "r"(_prmt_142), "r"(_prmt_140 | _prmt_144 & 2155905152u));
#endif
        dp_hi[3] = _prmt_110_1;
        uint32_t _bf16x2_fma_64;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_64) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_64;
        uint32_t _bf16x2_fma_65;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_65) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_65;
        uint32_t _bf16x2_fma_66;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_66) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_66;
        uint32_t _bf16x2_fma_67;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_67) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_67;
        uint32_t _bf16x2_fma_68;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_68) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_68;
        uint32_t _bf16x2_fma_69;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_69) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_69;
        uint32_t _bf16x2_fma_70;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_70) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_70;
        uint32_t _bf16x2_fma_71;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_71) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_71;
        uint32_t w_lo_9;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_9) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_10;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_10) : "r"(vw[2]), "r"(vw[3]));
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_145;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_145) : "r"(w_lo_9 & 2004318071), "r"(w_lo_9 & 2004318071));
#endif
        uint32_t _prmt_146;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_146) : "r"(w_lo_9 & 2004318071), "r"(w_lo_9 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_146) : "r"(2071689984), "r"(2088533116), "r"(w_lo_9 & 2004318071));
#endif
        uint32_t _prmt_147;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_147) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_9 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_147) : "r"(2071689984), "r"(2088533116), "r"(_prmt_145));
#endif
        uint32_t _prmt_148;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_148) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_146));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_148) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_9 & 2004318071));
#endif
        uint32_t _prmt_149;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_149) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_9 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_149) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_145));
#endif
        uint32_t _prmt_150;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_150) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_146));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_150) : "r"(w_lo_9 << 4), "r"(w_lo_9));
#endif
        uint32_t _prmt_151;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_151) : "r"(w_lo_9 << 4), "r"(w_lo_9));
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_152) : "r"(w_lo_9 << 4), "r"(w_lo_9));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_151) : "r"(w_lo_9 << 4), "r"(w_lo_9));
#endif
        uint32_t _prmt_111_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_111_1) : "r"(_prmt_149), "r"(_prmt_147 | _prmt_151 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_111_1) : "r"(_prmt_148), "r"(_prmt_146 | _prmt_150 & 2155905152u));
#endif
        dp_lo[4] = _prmt_111_1;
        uint32_t _prmt_112_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_112_1) : "r"(_prmt_149), "r"(_prmt_147 | _prmt_151 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_112_1) : "r"(_prmt_148), "r"(_prmt_146 | _prmt_150 & 2155905152u));
#endif
        dp_lo[5] = _prmt_112_1;
        uint32_t _prmt_113_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_113_1) : "r"(_prmt_150), "r"(_prmt_148 | _prmt_152 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_113_1) : "r"(_prmt_149), "r"(_prmt_147 | _prmt_151 & 2155905152u));
#endif
        dp_lo[6] = _prmt_113_1;
        uint32_t _prmt_114_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_114_1) : "r"(_prmt_150), "r"(_prmt_148 | _prmt_152 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_114_1) : "r"(_prmt_149), "r"(_prmt_147 | _prmt_151 & 2155905152u));
#endif
        dp_lo[7] = _prmt_114_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_152) : "r"(w_hi_10 & 2004318071), "r"(w_hi_10 & 2004318071));
#endif
        uint32_t _prmt_153;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_153) : "r"(w_hi_10 & 2004318071), "r"(w_hi_10 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_153) : "r"(2071689984), "r"(2088533116), "r"(w_hi_10 & 2004318071));
#endif
        uint32_t _prmt_154;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_154) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_10 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_154) : "r"(2071689984), "r"(2088533116), "r"(_prmt_152));
#endif
        uint32_t _prmt_155;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_155) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_153));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_155) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_10 & 2004318071));
#endif
        uint32_t _prmt_156;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_156) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_10 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_156) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_152));
#endif
        uint32_t _prmt_157;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_157) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_153));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_157) : "r"(w_hi_10 << 4), "r"(w_hi_10));
#endif
        uint32_t _prmt_158;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_158) : "r"(w_hi_10 << 4), "r"(w_hi_10));
        uint32_t _prmt_159;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_159) : "r"(w_hi_10 << 4), "r"(w_hi_10));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_158) : "r"(w_hi_10 << 4), "r"(w_hi_10));
#endif
        uint32_t _prmt_115_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_115_1) : "r"(_prmt_156), "r"(_prmt_154 | _prmt_158 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_115_1) : "r"(_prmt_155), "r"(_prmt_153 | _prmt_157 & 2155905152u));
#endif
        dp_hi[4] = _prmt_115_1;
        uint32_t _prmt_116_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_116_1) : "r"(_prmt_156), "r"(_prmt_154 | _prmt_158 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_116_1) : "r"(_prmt_155), "r"(_prmt_153 | _prmt_157 & 2155905152u));
#endif
        dp_hi[5] = _prmt_116_1;
        uint32_t _prmt_117_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_117_1) : "r"(_prmt_157), "r"(_prmt_155 | _prmt_159 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_117_1) : "r"(_prmt_156), "r"(_prmt_154 | _prmt_158 & 2155905152u));
#endif
        dp_hi[6] = _prmt_117_1;
        uint32_t _prmt_118_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_118_1) : "r"(_prmt_157), "r"(_prmt_155 | _prmt_159 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_118_1) : "r"(_prmt_156), "r"(_prmt_154 | _prmt_158 & 2155905152u));
#endif
        dp_hi[7] = _prmt_118_1;
        uint32_t _bf16x2_fma_72;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_72) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_72;
        uint32_t _bf16x2_fma_73;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_73) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_73;
        uint32_t _bf16x2_fma_74;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_74) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_74;
        uint32_t _bf16x2_fma_75;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_75) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_75;
        uint32_t _bf16x2_fma_76;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_76) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_76;
        uint32_t _bf16x2_fma_77;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_77) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_77;
        uint32_t _bf16x2_fma_78;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_78) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_78;
        uint32_t _bf16x2_fma_79;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_79) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_79;
        b_pair[0] = dp_lo[0];
        b_pair[1] = dp_lo[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[1];
        b_pair[1] = dp_lo[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[2];
        b_pair[1] = dp_lo[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[3];
        b_pair[1] = dp_lo[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[0];
        b_pair[1] = dp_hi[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[1];
        b_pair[1] = dp_hi[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[2];
        b_pair[1] = dp_hi[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[3];
        b_pair[1] = dp_hi[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&p_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 3]))
            : "r"(p_frag_addr + (unsigned int)(16 * (32 + lane_1))));
#if __CUDA_ARCH__ == 900
        int v_row_11 = v_stage + (16 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_11 + v_col)));
        unsigned int scale_byte_12 = (unsigned int)kv_u8[v_row_11 + v_scale_col];
        int v_row_13 = v_stage + (16 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_13 + v_col)));
        unsigned int scale_byte_14 = (unsigned int)kv_u8[v_row_13 + v_scale_col];
        int v_row_15 = v_stage + (16 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_15 + v_col)));
        unsigned int scale_byte_16 = (unsigned int)kv_u8[v_row_15 + v_scale_col];
        int v_row_17 = v_stage + (16 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_17 + v_col)));
        unsigned int scale_byte_18 = (unsigned int)kv_u8[v_row_17 + v_scale_col];
#else
        unsigned int sf_packed_11 = 0;
        int v_row_12 = v_stage + (16 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_12 + v_col)));
        unsigned int scale_byte_13 = (unsigned int)kv_u8[v_row_12 + v_scale_col];
        sf_packed_11 = sf_packed_11 | scale_byte_13;
        int v_row_14 = v_stage + (16 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_14 + v_col)));
        unsigned int scale_byte_15 = (unsigned int)kv_u8[v_row_14 + v_scale_col];
        sf_packed_11 = sf_packed_11 | scale_byte_15 << 8;
        int v_row_16 = v_stage + (16 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_16 + v_col)));
        unsigned int scale_byte_17 = (unsigned int)kv_u8[v_row_16 + v_scale_col];
        sf_packed_11 = sf_packed_11 | scale_byte_17 << 16;
        int v_row_18 = v_stage + (16 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_18 + v_col)));
        unsigned int scale_byte_19 = (unsigned int)kv_u8[v_row_18 + v_scale_col];
        sf_packed_11 = sf_packed_11 | scale_byte_19 << 24;
        uint32_t _prmt_159;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_159) : "r"(sf_packed_11), "r"(sf_packed_11));
        sfv[0] = _prmt_159 & 2147516416u | (_prmt_159 & 2130738944) >> 4;
        sfv[1] = _prmt_159 << 8 & 2147516416u | (_prmt_159 << 8 & 2130738944) >> 4;
        uint32_t w_lo_20;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_20) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_21;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_21) : "r"(vw[0]), "r"(vw[1]));
#endif
        uint32_t _prmt_160;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_160) : "r"(scale_byte_12), "r"(scale_byte_14));
        sfv[0] = _prmt_160 & 2147516416u | (_prmt_160 & 2130738944) >> 4;
#else
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_160) : "r"(w_lo_20 & 2004318071), "r"(w_lo_20 & 2004318071));
#endif
        uint32_t _prmt_161;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_161) : "r"(scale_byte_16), "r"(scale_byte_18));
        sfv[1] = _prmt_161 & 2147516416u | (_prmt_161 & 2130738944) >> 4;
        uint32_t w_lo_19;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_19) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_20;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_20) : "r"(vw[0]), "r"(vw[1]));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_161) : "r"(2071689984), "r"(2088533116), "r"(w_lo_20 & 2004318071));
#endif
        uint32_t _prmt_162;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_162) : "r"(w_lo_19 & 2004318071), "r"(w_lo_19 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_162) : "r"(2071689984), "r"(2088533116), "r"(_prmt_160));
#endif
        uint32_t _prmt_163;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_163) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_19 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_163) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_20 & 2004318071));
#endif
        uint32_t _prmt_164;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_164) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_162));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_164) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_160));
#endif
        uint32_t _prmt_165;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_165) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_19 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_165) : "r"(w_lo_20 << 4), "r"(w_lo_20));
#endif
        uint32_t _prmt_166;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_166) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_162));
        uint32_t _prmt_167;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_167) : "r"(w_lo_19 << 4), "r"(w_lo_19));
        uint32_t _prmt_168;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_168) : "r"(w_lo_19 << 4), "r"(w_lo_19));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_166) : "r"(w_lo_20 << 4), "r"(w_lo_20));
#endif
        uint32_t _prmt_119_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_119_1) : "r"(_prmt_165), "r"(_prmt_163 | _prmt_167 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_119_1) : "r"(_prmt_163), "r"(_prmt_161 | _prmt_165 & 2155905152u));
#endif
        dp_lo[0] = _prmt_119_1;
        uint32_t _prmt_120_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_120_1) : "r"(_prmt_165), "r"(_prmt_163 | _prmt_167 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_120_1) : "r"(_prmt_163), "r"(_prmt_161 | _prmt_165 & 2155905152u));
#endif
        dp_lo[1] = _prmt_120_1;
        uint32_t _prmt_121_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_121_1) : "r"(_prmt_166), "r"(_prmt_164 | _prmt_168 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_121_1) : "r"(_prmt_164), "r"(_prmt_162 | _prmt_166 & 2155905152u));
#endif
        dp_lo[2] = _prmt_121_1;
        uint32_t _prmt_122_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_122_1) : "r"(_prmt_166), "r"(_prmt_164 | _prmt_168 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_122_1) : "r"(_prmt_164), "r"(_prmt_162 | _prmt_166 & 2155905152u));
#endif
        dp_lo[3] = _prmt_122_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_167;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_167) : "r"(w_hi_21 & 2004318071), "r"(w_hi_21 & 2004318071));
        uint32_t _prmt_168;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_168) : "r"(2071689984), "r"(2088533116), "r"(w_hi_21 & 2004318071));
#endif
        uint32_t _prmt_169;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_169) : "r"(w_hi_20 & 2004318071), "r"(w_hi_20 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_169) : "r"(2071689984), "r"(2088533116), "r"(_prmt_167));
#endif
        uint32_t _prmt_170;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_170) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_20 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_170) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_21 & 2004318071));
#endif
        uint32_t _prmt_171;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_171) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_169));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_171) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_167));
#endif
        uint32_t _prmt_172;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_172) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_20 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_172) : "r"(w_hi_21 << 4), "r"(w_hi_21));
#endif
        uint32_t _prmt_173;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_173) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_169));
        uint32_t _prmt_174;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_174) : "r"(w_hi_20 << 4), "r"(w_hi_20));
        uint32_t _prmt_175;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_175) : "r"(w_hi_20 << 4), "r"(w_hi_20));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_173) : "r"(w_hi_21 << 4), "r"(w_hi_21));
#endif
        uint32_t _prmt_123_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_123_1) : "r"(_prmt_172), "r"(_prmt_170 | _prmt_174 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_123_1) : "r"(_prmt_170), "r"(_prmt_168 | _prmt_172 & 2155905152u));
#endif
        dp_hi[0] = _prmt_123_1;
        uint32_t _prmt_124_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_124_1) : "r"(_prmt_172), "r"(_prmt_170 | _prmt_174 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_124_1) : "r"(_prmt_170), "r"(_prmt_168 | _prmt_172 & 2155905152u));
#endif
        dp_hi[1] = _prmt_124_1;
        uint32_t _prmt_125_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_125_1) : "r"(_prmt_173), "r"(_prmt_171 | _prmt_175 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_125_1) : "r"(_prmt_171), "r"(_prmt_169 | _prmt_173 & 2155905152u));
#endif
        dp_hi[2] = _prmt_125_1;
        uint32_t _prmt_126_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_126_1) : "r"(_prmt_173), "r"(_prmt_171 | _prmt_175 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_126_1) : "r"(_prmt_171), "r"(_prmt_169 | _prmt_173 & 2155905152u));
#endif
        dp_hi[3] = _prmt_126_1;
        uint32_t _bf16x2_fma_80;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_80) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_80;
        uint32_t _bf16x2_fma_81;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_81) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_81;
        uint32_t _bf16x2_fma_82;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_82) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_82;
        uint32_t _bf16x2_fma_83;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_83) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_83;
        uint32_t _bf16x2_fma_84;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_84) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_84;
        uint32_t _bf16x2_fma_85;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_85) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_85;
        uint32_t _bf16x2_fma_86;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_86) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_86;
        uint32_t _bf16x2_fma_87;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_87) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_87;
#if __CUDA_ARCH__ == 900
        uint32_t w_lo_21;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_21) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_22;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_22) : "r"(vw[2]), "r"(vw[3]));
#else
        uint32_t w_lo_22;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_22) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_23;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_23) : "r"(vw[2]), "r"(vw[3]));
        uint32_t _prmt_174;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_174) : "r"(w_lo_22 & 2004318071), "r"(w_lo_22 & 2004318071));
        uint32_t _prmt_175;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_175) : "r"(2071689984), "r"(2088533116), "r"(w_lo_22 & 2004318071));
#endif
        uint32_t _prmt_176;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_176) : "r"(w_lo_21 & 2004318071), "r"(w_lo_21 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_176) : "r"(2071689984), "r"(2088533116), "r"(_prmt_174));
#endif
        uint32_t _prmt_177;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_177) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_21 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_177) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_22 & 2004318071));
#endif
        uint32_t _prmt_178;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_178) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_176));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_178) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_174));
#endif
        uint32_t _prmt_179;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_179) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_21 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_179) : "r"(w_lo_22 << 4), "r"(w_lo_22));
#endif
        uint32_t _prmt_180;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_180) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_176));
        uint32_t _prmt_181;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_181) : "r"(w_lo_21 << 4), "r"(w_lo_21));
        uint32_t _prmt_182;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_182) : "r"(w_lo_21 << 4), "r"(w_lo_21));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_180) : "r"(w_lo_22 << 4), "r"(w_lo_22));
#endif
        uint32_t _prmt_127_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_127_1) : "r"(_prmt_179), "r"(_prmt_177 | _prmt_181 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_127_1) : "r"(_prmt_177), "r"(_prmt_175 | _prmt_179 & 2155905152u));
#endif
        dp_lo[4] = _prmt_127_1;
        uint32_t _prmt_128_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_128_1) : "r"(_prmt_179), "r"(_prmt_177 | _prmt_181 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_128_1) : "r"(_prmt_177), "r"(_prmt_175 | _prmt_179 & 2155905152u));
#endif
        dp_lo[5] = _prmt_128_1;
        uint32_t _prmt_129_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_129_1) : "r"(_prmt_180), "r"(_prmt_178 | _prmt_182 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_129_1) : "r"(_prmt_178), "r"(_prmt_176 | _prmt_180 & 2155905152u));
#endif
        dp_lo[6] = _prmt_129_1;
        uint32_t _prmt_130_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_130_1) : "r"(_prmt_180), "r"(_prmt_178 | _prmt_182 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_130_1) : "r"(_prmt_178), "r"(_prmt_176 | _prmt_180 & 2155905152u));
#endif
        dp_lo[7] = _prmt_130_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_181;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_181) : "r"(w_hi_23 & 2004318071), "r"(w_hi_23 & 2004318071));
        uint32_t _prmt_182;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_182) : "r"(2071689984), "r"(2088533116), "r"(w_hi_23 & 2004318071));
#endif
        uint32_t _prmt_183;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_183) : "r"(w_hi_22 & 2004318071), "r"(w_hi_22 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_183) : "r"(2071689984), "r"(2088533116), "r"(_prmt_181));
#endif
        uint32_t _prmt_184;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_184) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_22 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_184) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_23 & 2004318071));
#endif
        uint32_t _prmt_185;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_185) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_183));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_185) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_181));
#endif
        uint32_t _prmt_186;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_186) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_22 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_186) : "r"(w_hi_23 << 4), "r"(w_hi_23));
#endif
        uint32_t _prmt_187;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_187) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_183));
        uint32_t _prmt_188;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_188) : "r"(w_hi_22 << 4), "r"(w_hi_22));
        uint32_t _prmt_189;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_189) : "r"(w_hi_22 << 4), "r"(w_hi_22));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_187) : "r"(w_hi_23 << 4), "r"(w_hi_23));
#endif
        uint32_t _prmt_131_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_131_1) : "r"(_prmt_186), "r"(_prmt_184 | _prmt_188 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_131_1) : "r"(_prmt_184), "r"(_prmt_182 | _prmt_186 & 2155905152u));
#endif
        dp_hi[4] = _prmt_131_1;
        uint32_t _prmt_132_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_132_1) : "r"(_prmt_186), "r"(_prmt_184 | _prmt_188 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_132_1) : "r"(_prmt_184), "r"(_prmt_182 | _prmt_186 & 2155905152u));
#endif
        dp_hi[5] = _prmt_132_1;
        uint32_t _prmt_133_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_133_1) : "r"(_prmt_187), "r"(_prmt_185 | _prmt_189 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_133_1) : "r"(_prmt_185), "r"(_prmt_183 | _prmt_187 & 2155905152u));
#endif
        dp_hi[6] = _prmt_133_1;
        uint32_t _prmt_134_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_134_1) : "r"(_prmt_187), "r"(_prmt_185 | _prmt_189 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_134_1) : "r"(_prmt_185), "r"(_prmt_183 | _prmt_187 & 2155905152u));
#endif
        dp_hi[7] = _prmt_134_1;
        uint32_t _bf16x2_fma_88;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_88) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_88;
        uint32_t _bf16x2_fma_89;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_89) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_89;
        uint32_t _bf16x2_fma_90;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_90) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_90;
        uint32_t _bf16x2_fma_91;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_91) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_91;
        uint32_t _bf16x2_fma_92;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_92) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_92;
        uint32_t _bf16x2_fma_93;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_93) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_93;
        uint32_t _bf16x2_fma_94;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_94) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_94;
        uint32_t _bf16x2_fma_95;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_95) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_95;
        b_pair[0] = dp_lo[0];
        b_pair[1] = dp_lo[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[1];
        b_pair[1] = dp_lo[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[2];
        b_pair[1] = dp_lo[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[3];
        b_pair[1] = dp_lo[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[0];
        b_pair[1] = dp_hi[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[1];
        b_pair[1] = dp_hi[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[2];
        b_pair[1] = dp_hi[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[3];
        b_pair[1] = dp_hi[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&p_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 3]))
            : "r"(p_frag_addr + (unsigned int)(16 * (64 + lane_1))));
#if __CUDA_ARCH__ == 900
        int v_row_23 = v_stage + (32 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_23 + v_col)));
        unsigned int scale_byte_24 = (unsigned int)kv_u8[v_row_23 + v_scale_col];
        int v_row_25 = v_stage + (32 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_25 + v_col)));
#else
        unsigned int sf_packed_24 = 0;
        int v_row_25 = v_stage + (32 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_25 + v_col)));
#endif
        unsigned int scale_byte_26 = (unsigned int)kv_u8[v_row_25 + v_scale_col];
#if __CUDA_ARCH__ == 900
        int v_row_27 = v_stage + (32 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_27 + v_col)));
#else
        sf_packed_24 = sf_packed_24 | scale_byte_26;
        int v_row_27 = v_stage + (32 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_27 + v_col)));
#endif
        unsigned int scale_byte_28 = (unsigned int)kv_u8[v_row_27 + v_scale_col];
#if __CUDA_ARCH__ == 900
        int v_row_29 = v_stage + (32 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_29 + v_col)));
#else
        sf_packed_24 = sf_packed_24 | scale_byte_28 << 8;
        int v_row_29 = v_stage + (32 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_29 + v_col)));
#endif
        unsigned int scale_byte_30 = (unsigned int)kv_u8[v_row_29 + v_scale_col];
#if __CUDA_ARCH__ == 900
#else
        sf_packed_24 = sf_packed_24 | scale_byte_30 << 16;
        int v_row_31 = v_stage + (32 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_31 + v_col)));
        unsigned int scale_byte_32 = (unsigned int)kv_u8[v_row_31 + v_scale_col];
        sf_packed_24 = sf_packed_24 | scale_byte_32 << 24;
        uint32_t _prmt_188;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_188) : "r"(sf_packed_24), "r"(sf_packed_24));
        sfv[0] = _prmt_188 & 2147516416u | (_prmt_188 & 2130738944) >> 4;
        sfv[1] = _prmt_188 << 8 & 2147516416u | (_prmt_188 << 8 & 2130738944) >> 4;
        uint32_t w_lo_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_33) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_34) : "r"(vw[0]), "r"(vw[1]));
        uint32_t _prmt_189;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_189) : "r"(w_lo_33 & 2004318071), "r"(w_lo_33 & 2004318071));
#endif
        uint32_t _prmt_190;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_190) : "r"(scale_byte_24), "r"(scale_byte_26));
        sfv[0] = _prmt_190 & 2147516416u | (_prmt_190 & 2130738944) >> 4;
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_190) : "r"(2071689984), "r"(2088533116), "r"(w_lo_33 & 2004318071));
#endif
        uint32_t _prmt_191;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_191) : "r"(scale_byte_28), "r"(scale_byte_30));
        sfv[1] = _prmt_191 & 2147516416u | (_prmt_191 & 2130738944) >> 4;
        uint32_t w_lo_31;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_31) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_32;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_32) : "r"(vw[0]), "r"(vw[1]));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_191) : "r"(2071689984), "r"(2088533116), "r"(_prmt_189));
#endif
        uint32_t _prmt_192;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_192) : "r"(w_lo_31 & 2004318071), "r"(w_lo_31 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_192) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_33 & 2004318071));
#endif
        uint32_t _prmt_193;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_193) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_31 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_193) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_189));
#endif
        uint32_t _prmt_194;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_194) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_192));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_194) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#endif
        uint32_t _prmt_195;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_195) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_31 & 2004318071));
        uint32_t _prmt_196;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_196) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_192));
        uint32_t _prmt_197;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_197) : "r"(w_lo_31 << 4), "r"(w_lo_31));
        uint32_t _prmt_198;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_198) : "r"(w_lo_31 << 4), "r"(w_lo_31));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_195) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#endif
        uint32_t _prmt_135_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_135_1) : "r"(_prmt_195), "r"(_prmt_193 | _prmt_197 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_135_1) : "r"(_prmt_192), "r"(_prmt_190 | _prmt_194 & 2155905152u));
#endif
        dp_lo[0] = _prmt_135_1;
        uint32_t _prmt_136_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_136_1) : "r"(_prmt_195), "r"(_prmt_193 | _prmt_197 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_136_1) : "r"(_prmt_192), "r"(_prmt_190 | _prmt_194 & 2155905152u));
#endif
        dp_lo[1] = _prmt_136_1;
        uint32_t _prmt_137_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_137_1) : "r"(_prmt_196), "r"(_prmt_194 | _prmt_198 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_137_1) : "r"(_prmt_193), "r"(_prmt_191 | _prmt_195 & 2155905152u));
#endif
        dp_lo[2] = _prmt_137_1;
        uint32_t _prmt_138_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_138_1) : "r"(_prmt_196), "r"(_prmt_194 | _prmt_198 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_138_1) : "r"(_prmt_193), "r"(_prmt_191 | _prmt_195 & 2155905152u));
#endif
        dp_lo[3] = _prmt_138_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_196;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_196) : "r"(w_hi_34 & 2004318071), "r"(w_hi_34 & 2004318071));
        uint32_t _prmt_197;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_197) : "r"(2071689984), "r"(2088533116), "r"(w_hi_34 & 2004318071));
        uint32_t _prmt_198;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_198) : "r"(2071689984), "r"(2088533116), "r"(_prmt_196));
#endif
        uint32_t _prmt_199;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_199) : "r"(w_hi_32 & 2004318071), "r"(w_hi_32 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_199) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_34 & 2004318071));
#endif
        uint32_t _prmt_200;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_200) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_32 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_200) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_196));
#endif
        uint32_t _prmt_201;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_201) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_199));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_201) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#endif
        uint32_t _prmt_202;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_202) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_32 & 2004318071));
        uint32_t _prmt_203;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_203) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_199));
        uint32_t _prmt_204;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_204) : "r"(w_hi_32 << 4), "r"(w_hi_32));
        uint32_t _prmt_205;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_205) : "r"(w_hi_32 << 4), "r"(w_hi_32));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_202) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#endif
        uint32_t _prmt_139_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_139_1) : "r"(_prmt_202), "r"(_prmt_200 | _prmt_204 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_139_1) : "r"(_prmt_199), "r"(_prmt_197 | _prmt_201 & 2155905152u));
#endif
        dp_hi[0] = _prmt_139_1;
        uint32_t _prmt_140_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_140_1) : "r"(_prmt_202), "r"(_prmt_200 | _prmt_204 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_140_1) : "r"(_prmt_199), "r"(_prmt_197 | _prmt_201 & 2155905152u));
#endif
        dp_hi[1] = _prmt_140_1;
        uint32_t _prmt_141_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_141_1) : "r"(_prmt_203), "r"(_prmt_201 | _prmt_205 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_141_1) : "r"(_prmt_200), "r"(_prmt_198 | _prmt_202 & 2155905152u));
#endif
        dp_hi[2] = _prmt_141_1;
        uint32_t _prmt_142_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_142_1) : "r"(_prmt_203), "r"(_prmt_201 | _prmt_205 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_142_1) : "r"(_prmt_200), "r"(_prmt_198 | _prmt_202 & 2155905152u));
#endif
        dp_hi[3] = _prmt_142_1;
        uint32_t _bf16x2_fma_96;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_96) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_96;
        uint32_t _bf16x2_fma_97;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_97) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_97;
        uint32_t _bf16x2_fma_98;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_98) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_98;
        uint32_t _bf16x2_fma_99;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_99) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_99;
        uint32_t _bf16x2_fma_100;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_100) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_100;
        uint32_t _bf16x2_fma_101;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_101) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_101;
        uint32_t _bf16x2_fma_102;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_102) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_102;
        uint32_t _bf16x2_fma_103;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_103) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_103;
#if __CUDA_ARCH__ == 900
        uint32_t w_lo_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_33) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_34) : "r"(vw[2]), "r"(vw[3]));
#else
        uint32_t w_lo_35;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_35) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_36;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_36) : "r"(vw[2]), "r"(vw[3]));
        uint32_t _prmt_203;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_203) : "r"(w_lo_35 & 2004318071), "r"(w_lo_35 & 2004318071));
        uint32_t _prmt_204;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_204) : "r"(2071689984), "r"(2088533116), "r"(w_lo_35 & 2004318071));
        uint32_t _prmt_205;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_205) : "r"(2071689984), "r"(2088533116), "r"(_prmt_203));
#endif
        uint32_t _prmt_206;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_206) : "r"(w_lo_33 & 2004318071), "r"(w_lo_33 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_206) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_35 & 2004318071));
#endif
        uint32_t _prmt_207;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_207) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_33 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_207) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_203));
#endif
        uint32_t _prmt_208;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_208) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_206));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_208) : "r"(w_lo_35 << 4), "r"(w_lo_35));
#endif
        uint32_t _prmt_209;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_209) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_33 & 2004318071));
        uint32_t _prmt_210;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_210) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_206));
        uint32_t _prmt_211;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_211) : "r"(w_lo_33 << 4), "r"(w_lo_33));
        uint32_t _prmt_212;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_212) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_209) : "r"(w_lo_35 << 4), "r"(w_lo_35));
#endif
        uint32_t _prmt_143_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_143_1) : "r"(_prmt_209), "r"(_prmt_207 | _prmt_211 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_143_1) : "r"(_prmt_206), "r"(_prmt_204 | _prmt_208 & 2155905152u));
#endif
        dp_lo[4] = _prmt_143_1;
        uint32_t _prmt_144_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_144_1) : "r"(_prmt_209), "r"(_prmt_207 | _prmt_211 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_144_1) : "r"(_prmt_206), "r"(_prmt_204 | _prmt_208 & 2155905152u));
#endif
        dp_lo[5] = _prmt_144_1;
        uint32_t _prmt_145_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_145_1) : "r"(_prmt_210), "r"(_prmt_208 | _prmt_212 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_145_1) : "r"(_prmt_207), "r"(_prmt_205 | _prmt_209 & 2155905152u));
#endif
        dp_lo[6] = _prmt_145_1;
        uint32_t _prmt_146_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_146_1) : "r"(_prmt_210), "r"(_prmt_208 | _prmt_212 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_146_1) : "r"(_prmt_207), "r"(_prmt_205 | _prmt_209 & 2155905152u));
#endif
        dp_lo[7] = _prmt_146_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_210;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_210) : "r"(w_hi_36 & 2004318071), "r"(w_hi_36 & 2004318071));
        uint32_t _prmt_211;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_211) : "r"(2071689984), "r"(2088533116), "r"(w_hi_36 & 2004318071));
        uint32_t _prmt_212;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_212) : "r"(2071689984), "r"(2088533116), "r"(_prmt_210));
#endif
        uint32_t _prmt_213;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_213) : "r"(w_hi_34 & 2004318071), "r"(w_hi_34 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_213) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_36 & 2004318071));
#endif
        uint32_t _prmt_214;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_214) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_34 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_214) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_210));
#endif
        uint32_t _prmt_215;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_215) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_213));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_215) : "r"(w_hi_36 << 4), "r"(w_hi_36));
#endif
        uint32_t _prmt_216;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_216) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_34 & 2004318071));
        uint32_t _prmt_217;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_217) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_213));
        uint32_t _prmt_218;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_218) : "r"(w_hi_34 << 4), "r"(w_hi_34));
        uint32_t _prmt_219;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_219) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_216) : "r"(w_hi_36 << 4), "r"(w_hi_36));
#endif
        uint32_t _prmt_147_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_147_1) : "r"(_prmt_216), "r"(_prmt_214 | _prmt_218 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_147_1) : "r"(_prmt_213), "r"(_prmt_211 | _prmt_215 & 2155905152u));
#endif
        dp_hi[4] = _prmt_147_1;
        uint32_t _prmt_148_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_148_1) : "r"(_prmt_216), "r"(_prmt_214 | _prmt_218 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_148_1) : "r"(_prmt_213), "r"(_prmt_211 | _prmt_215 & 2155905152u));
#endif
        dp_hi[5] = _prmt_148_1;
        uint32_t _prmt_149_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_149_1) : "r"(_prmt_217), "r"(_prmt_215 | _prmt_219 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_149_1) : "r"(_prmt_214), "r"(_prmt_212 | _prmt_216 & 2155905152u));
#endif
        dp_hi[6] = _prmt_149_1;
        uint32_t _prmt_150_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_150_1) : "r"(_prmt_217), "r"(_prmt_215 | _prmt_219 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_150_1) : "r"(_prmt_214), "r"(_prmt_212 | _prmt_216 & 2155905152u));
#endif
        dp_hi[7] = _prmt_150_1;
        uint32_t _bf16x2_fma_104;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_104) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_104;
        uint32_t _bf16x2_fma_105;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_105) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_105;
        uint32_t _bf16x2_fma_106;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_106) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_106;
        uint32_t _bf16x2_fma_107;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_107) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_107;
        uint32_t _bf16x2_fma_108;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_108) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_108;
        uint32_t _bf16x2_fma_109;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_109) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_109;
        uint32_t _bf16x2_fma_110;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_110) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_110;
        uint32_t _bf16x2_fma_111;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_111) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_111;
        b_pair[0] = dp_lo[0];
        b_pair[1] = dp_lo[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[1];
        b_pair[1] = dp_lo[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[2];
        b_pair[1] = dp_lo[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[3];
        b_pair[1] = dp_lo[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[0];
        b_pair[1] = dp_hi[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[1];
        b_pair[1] = dp_hi[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[2];
        b_pair[1] = dp_hi[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[3];
        b_pair[1] = dp_hi[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&p_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_a[(0) + 3]))
            : "r"(p_frag_addr + (unsigned int)(16 * (96 + lane_1))));
#if __CUDA_ARCH__ == 900
        int v_row_35 = v_stage + (48 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_35 + v_col)));
        unsigned int scale_byte_36 = (unsigned int)kv_u8[v_row_35 + v_scale_col];
        int v_row_37 = v_stage + (48 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_37 + v_col)));
        unsigned int scale_byte_38 = (unsigned int)kv_u8[v_row_37 + v_scale_col];
        int v_row_39 = v_stage + (48 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_39 + v_col)));
        unsigned int scale_byte_40 = (unsigned int)kv_u8[v_row_39 + v_scale_col];
        int v_row_41 = v_stage + (48 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_41 + v_col)));
        unsigned int scale_byte_42 = (unsigned int)kv_u8[v_row_41 + v_scale_col];
#else
        unsigned int sf_packed_37 = 0;
        int v_row_38 = v_stage + (48 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_38 + v_col)));
        unsigned int scale_byte_39 = (unsigned int)kv_u8[v_row_38 + v_scale_col];
        sf_packed_37 = sf_packed_37 | scale_byte_39;
        int v_row_40 = v_stage + (48 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_40 + v_col)));
        unsigned int scale_byte_41 = (unsigned int)kv_u8[v_row_40 + v_scale_col];
        sf_packed_37 = sf_packed_37 | scale_byte_41 << 8;
        int v_row_42 = v_stage + (48 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_42 + v_col)));
        unsigned int scale_byte_43 = (unsigned int)kv_u8[v_row_42 + v_scale_col];
        sf_packed_37 = sf_packed_37 | scale_byte_43 << 16;
        int v_row_44 = v_stage + (48 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_44 + v_col)));
        unsigned int scale_byte_45 = (unsigned int)kv_u8[v_row_44 + v_scale_col];
        sf_packed_37 = sf_packed_37 | scale_byte_45 << 24;
        uint32_t _prmt_217;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_217) : "r"(sf_packed_37), "r"(sf_packed_37));
        sfv[0] = _prmt_217 & 2147516416u | (_prmt_217 & 2130738944) >> 4;
        sfv[1] = _prmt_217 << 8 & 2147516416u | (_prmt_217 << 8 & 2130738944) >> 4;
        uint32_t w_lo_46;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_46) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_47;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_47) : "r"(vw[0]), "r"(vw[1]));
        uint32_t _prmt_218;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_218) : "r"(w_lo_46 & 2004318071), "r"(w_lo_46 & 2004318071));
        uint32_t _prmt_219;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_219) : "r"(2071689984), "r"(2088533116), "r"(w_lo_46 & 2004318071));
#endif
        uint32_t _prmt_220;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_220) : "r"(scale_byte_36), "r"(scale_byte_38));
        sfv[0] = _prmt_220 & 2147516416u | (_prmt_220 & 2130738944) >> 4;
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_220) : "r"(2071689984), "r"(2088533116), "r"(_prmt_218));
#endif
        uint32_t _prmt_221;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_221) : "r"(scale_byte_40), "r"(scale_byte_42));
        sfv[1] = _prmt_221 & 2147516416u | (_prmt_221 & 2130738944) >> 4;
        uint32_t w_lo_43;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_43) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_44;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_44) : "r"(vw[0]), "r"(vw[1]));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_221) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_46 & 2004318071));
#endif
        uint32_t _prmt_222;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_222) : "r"(w_lo_43 & 2004318071), "r"(w_lo_43 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_222) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_218));
#endif
        uint32_t _prmt_223;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_223) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_43 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_223) : "r"(w_lo_46 << 4), "r"(w_lo_46));
#endif
        uint32_t _prmt_224;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_224) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_222));
        uint32_t _prmt_225;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_225) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_43 & 2004318071));
        uint32_t _prmt_226;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_226) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_222));
        uint32_t _prmt_227;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_227) : "r"(w_lo_43 << 4), "r"(w_lo_43));
        uint32_t _prmt_228;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_228) : "r"(w_lo_43 << 4), "r"(w_lo_43));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_224) : "r"(w_lo_46 << 4), "r"(w_lo_46));
#endif
        uint32_t _prmt_151_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_151_1) : "r"(_prmt_225), "r"(_prmt_223 | _prmt_227 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_151_1) : "r"(_prmt_221), "r"(_prmt_219 | _prmt_223 & 2155905152u));
#endif
        dp_lo[0] = _prmt_151_1;
        uint32_t _prmt_152_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_152_1) : "r"(_prmt_225), "r"(_prmt_223 | _prmt_227 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_152_1) : "r"(_prmt_221), "r"(_prmt_219 | _prmt_223 & 2155905152u));
#endif
        dp_lo[1] = _prmt_152_1;
        uint32_t _prmt_153_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_153_1) : "r"(_prmt_226), "r"(_prmt_224 | _prmt_228 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_153_1) : "r"(_prmt_222), "r"(_prmt_220 | _prmt_224 & 2155905152u));
#endif
        dp_lo[2] = _prmt_153_1;
        uint32_t _prmt_154_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_154_1) : "r"(_prmt_226), "r"(_prmt_224 | _prmt_228 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_154_1) : "r"(_prmt_222), "r"(_prmt_220 | _prmt_224 & 2155905152u));
#endif
        dp_lo[3] = _prmt_154_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_225;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_225) : "r"(w_hi_47 & 2004318071), "r"(w_hi_47 & 2004318071));
        uint32_t _prmt_226;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_226) : "r"(2071689984), "r"(2088533116), "r"(w_hi_47 & 2004318071));
        uint32_t _prmt_227;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_227) : "r"(2071689984), "r"(2088533116), "r"(_prmt_225));
        uint32_t _prmt_228;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_228) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_47 & 2004318071));
#endif
        uint32_t _prmt_229;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_229) : "r"(w_hi_44 & 2004318071), "r"(w_hi_44 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_229) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_225));
#endif
        uint32_t _prmt_230;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_230) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_44 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_230) : "r"(w_hi_47 << 4), "r"(w_hi_47));
#endif
        uint32_t _prmt_231;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_231) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_229));
        uint32_t _prmt_232;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_232) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_44 & 2004318071));
        uint32_t _prmt_233;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_233) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_229));
        uint32_t _prmt_234;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_234) : "r"(w_hi_44 << 4), "r"(w_hi_44));
        uint32_t _prmt_235;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_235) : "r"(w_hi_44 << 4), "r"(w_hi_44));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_231) : "r"(w_hi_47 << 4), "r"(w_hi_47));
#endif
        uint32_t _prmt_155_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_155_1) : "r"(_prmt_232), "r"(_prmt_230 | _prmt_234 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_155_1) : "r"(_prmt_228), "r"(_prmt_226 | _prmt_230 & 2155905152u));
#endif
        dp_hi[0] = _prmt_155_1;
        uint32_t _prmt_156_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_156_1) : "r"(_prmt_232), "r"(_prmt_230 | _prmt_234 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_156_1) : "r"(_prmt_228), "r"(_prmt_226 | _prmt_230 & 2155905152u));
#endif
        dp_hi[1] = _prmt_156_1;
        uint32_t _prmt_157_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_157_1) : "r"(_prmt_233), "r"(_prmt_231 | _prmt_235 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_157_1) : "r"(_prmt_229), "r"(_prmt_227 | _prmt_231 & 2155905152u));
#endif
        dp_hi[2] = _prmt_157_1;
        uint32_t _prmt_158_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_158_1) : "r"(_prmt_233), "r"(_prmt_231 | _prmt_235 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_158_1) : "r"(_prmt_229), "r"(_prmt_227 | _prmt_231 & 2155905152u));
#endif
        dp_hi[3] = _prmt_158_1;
        uint32_t _bf16x2_fma_112;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_112) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_112;
        uint32_t _bf16x2_fma_113;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_113) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_113;
        uint32_t _bf16x2_fma_114;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_114) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_114;
        uint32_t _bf16x2_fma_115;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_115) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_115;
        uint32_t _bf16x2_fma_116;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_116) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_116;
        uint32_t _bf16x2_fma_117;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_117) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_117;
        uint32_t _bf16x2_fma_118;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_118) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_118;
        uint32_t _bf16x2_fma_119;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_119) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_119;
#if __CUDA_ARCH__ == 900
        uint32_t w_lo_45;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_45) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_46;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_46) : "r"(vw[2]), "r"(vw[3]));
#else
        uint32_t w_lo_48;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_48) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_49;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_49) : "r"(vw[2]), "r"(vw[3]));
        uint32_t _prmt_232;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_232) : "r"(w_lo_48 & 2004318071), "r"(w_lo_48 & 2004318071));
        uint32_t _prmt_233;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_233) : "r"(2071689984), "r"(2088533116), "r"(w_lo_48 & 2004318071));
        uint32_t _prmt_234;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_234) : "r"(2071689984), "r"(2088533116), "r"(_prmt_232));
        uint32_t _prmt_235;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_235) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_48 & 2004318071));
#endif
        uint32_t _prmt_236;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_236) : "r"(w_lo_45 & 2004318071), "r"(w_lo_45 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_236) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_232));
#endif
        uint32_t _prmt_237;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_237) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_45 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_237) : "r"(w_lo_48 << 4), "r"(w_lo_48));
#endif
        uint32_t _prmt_238;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_238) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_236));
        uint32_t _prmt_239;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_239) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_45 & 2004318071));
        uint32_t _prmt_240;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_240) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_236));
        uint32_t _prmt_241;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_241) : "r"(w_lo_45 << 4), "r"(w_lo_45));
        uint32_t _prmt_242;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_242) : "r"(w_lo_45 << 4), "r"(w_lo_45));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_238) : "r"(w_lo_48 << 4), "r"(w_lo_48));
#endif
        uint32_t _prmt_159_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_159_1) : "r"(_prmt_239), "r"(_prmt_237 | _prmt_241 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_159_1) : "r"(_prmt_235), "r"(_prmt_233 | _prmt_237 & 2155905152u));
#endif
        dp_lo[4] = _prmt_159_1;
        uint32_t _prmt_160_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_160_1) : "r"(_prmt_239), "r"(_prmt_237 | _prmt_241 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_160_1) : "r"(_prmt_235), "r"(_prmt_233 | _prmt_237 & 2155905152u));
#endif
        dp_lo[5] = _prmt_160_1;
        uint32_t _prmt_161_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_161_1) : "r"(_prmt_240), "r"(_prmt_238 | _prmt_242 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_161_1) : "r"(_prmt_236), "r"(_prmt_234 | _prmt_238 & 2155905152u));
#endif
        dp_lo[6] = _prmt_161_1;
        uint32_t _prmt_162_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_162_1) : "r"(_prmt_240), "r"(_prmt_238 | _prmt_242 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_162_1) : "r"(_prmt_236), "r"(_prmt_234 | _prmt_238 & 2155905152u));
#endif
        dp_lo[7] = _prmt_162_1;
#if __CUDA_ARCH__ == 900
#else
        uint32_t _prmt_239;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_239) : "r"(w_hi_49 & 2004318071), "r"(w_hi_49 & 2004318071));
        uint32_t _prmt_240;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_240) : "r"(2071689984), "r"(2088533116), "r"(w_hi_49 & 2004318071));
        uint32_t _prmt_241;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_241) : "r"(2071689984), "r"(2088533116), "r"(_prmt_239));
        uint32_t _prmt_242;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_242) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_49 & 2004318071));
#endif
        uint32_t _prmt_243;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_243) : "r"(w_hi_46 & 2004318071), "r"(w_hi_46 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_243) : "r"(3229614080u), "r"(3229630464u), "r"(_prmt_239));
#endif
        uint32_t _prmt_244;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_244) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_46 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_244) : "r"(w_hi_49 << 4), "r"(w_hi_49));
#endif
        uint32_t _prmt_245;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_245) : "r"(lut_regs[0]), "r"(2088533116), "r"(_prmt_243));
        uint32_t _prmt_246;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_246) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_46 & 2004318071));
        uint32_t _prmt_247;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_247) : "r"(lut_regs[1]), "r"(3229630464u), "r"(_prmt_243));
        uint32_t _prmt_248;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_248) : "r"(w_hi_46 << 4), "r"(w_hi_46));
        uint32_t _prmt_249;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_249) : "r"(w_hi_46 << 4), "r"(w_hi_46));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_245) : "r"(w_hi_49 << 4), "r"(w_hi_49));
#endif
        uint32_t _prmt_163_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_163_1) : "r"(_prmt_246), "r"(_prmt_244 | _prmt_248 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_163_1) : "r"(_prmt_242), "r"(_prmt_240 | _prmt_244 & 2155905152u));
#endif
        dp_hi[4] = _prmt_163_1;
        uint32_t _prmt_164_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_164_1) : "r"(_prmt_246), "r"(_prmt_244 | _prmt_248 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_164_1) : "r"(_prmt_242), "r"(_prmt_240 | _prmt_244 & 2155905152u));
#endif
        dp_hi[5] = _prmt_164_1;
        uint32_t _prmt_165_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_165_1) : "r"(_prmt_247), "r"(_prmt_245 | _prmt_249 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_165_1) : "r"(_prmt_243), "r"(_prmt_241 | _prmt_245 & 2155905152u));
#endif
        dp_hi[6] = _prmt_165_1;
        uint32_t _prmt_166_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_166_1) : "r"(_prmt_247), "r"(_prmt_245 | _prmt_249 & 2155905152u));
#else
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_166_1) : "r"(_prmt_243), "r"(_prmt_241 | _prmt_245 & 2155905152u));
#endif
        dp_hi[7] = _prmt_166_1;
        uint32_t _bf16x2_fma_120;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_120) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_120;
        uint32_t _bf16x2_fma_121;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_121) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_121;
        uint32_t _bf16x2_fma_122;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_122) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_122;
        uint32_t _bf16x2_fma_123;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_123) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_123;
        uint32_t _bf16x2_fma_124;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_124) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_124;
        uint32_t _bf16x2_fma_125;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_125) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_125;
        uint32_t _bf16x2_fma_126;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_126) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_126;
        uint32_t _bf16x2_fma_127;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_127) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_127;
        b_pair[0] = dp_lo[0];
        b_pair[1] = dp_lo[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[1];
        b_pair[1] = dp_lo[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[2];
        b_pair[1] = dp_lo[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_lo[3];
        b_pair[1] = dp_lo[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[0];
        b_pair[1] = dp_hi[4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[1];
        b_pair[1] = dp_hi[5];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[2];
        b_pair[1] = dp_hi[6];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        b_pair[0] = dp_hi[3];
        b_pair[1] = dp_hi[7];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
            : "r"(p_a[0]), "r"(p_a[1]), "r"(p_a[2]), "r"(p_a[3]), "r"(b_pair[0]), "r"(b_pair[1]));
        if (warp_1 < 2 && multi) {
            int write_slot = parity + 2 & 1;
            bool valid_0_3 = phys[0] >= 0 && phys[0] < num_blocks;
            int _max_167 = ((phys[0]) > (0) ? (phys[0]) : (0));
            unsigned long long block_3 = (unsigned long long)_max_167;
            unsigned long long local_u64_3 = (unsigned long long)local[0];
            unsigned long long data_off_7 = block_3 * data_stride_block_u64 + local_u64_3 * data_stride_token_u64;
            unsigned long long sf_off_5 = block_3 * sf_stride_block_u64 + local_u64_3 * sf_stride_token_u64;
            unsigned int w0_1_2 = (unsigned int)data_off_7;
            unsigned int w1_2_2 = (unsigned int)(data_off_7 >> 32);
            unsigned int w2_3_2 = (unsigned int)sf_off_5;
            unsigned int w3_4_2 = (unsigned int)(sf_off_5 >> 32);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(write_slot * 1024 + token * 16)), "r"(w0_1_2), "r"(w1_2_2), "r"(w2_3_2), "r"(w3_4_2) : "memory");
            unsigned int bits_5 = __ballot_sync(0xFFFFFFFF, valid_0_3);
            if (lane_1 == 0) {
                slot_valid[write_slot * 2 + warp_1] = bits_5;
            }
            idx[0] = idx[1];
        }
        stage = stage + 1;
        if (stage >= 2) {
            stage = 0;
        }
        parity = parity ^ 1;
    }
    int dim0 = 64 * warp_1 + 16 * t;
    float v[16];
    float gated[8];
    int head = g;
    if (head < group_size) {
        bool has_values = l[0] > 0.0f;
        float _rcp_168 = approx_rcp(fmaxf(l[0], 1e-20f));
        float inv = output_scale * _rcp_168;
        v[0] = o[0];
        v[1] = o[4];
        v[8] = o[1];
        v[9] = o[5];
        v[2] = o[8];
        v[3] = o[12];
        v[10] = o[9];
        v[11] = o[13];
        v[4] = o[16];
        v[5] = o[20];
        v[12] = o[17];
        v[13] = o[21];
        v[6] = o[24];
        v[7] = o[28];
        v[14] = o[25];
        v[15] = o[29];
        v[0] = ((has_values) ? v[0] * inv : 0.0f);
        v[1] = ((has_values) ? v[1] * inv : 0.0f);
        v[2] = ((has_values) ? v[2] * inv : 0.0f);
        v[3] = ((has_values) ? v[3] * inv : 0.0f);
        v[4] = ((has_values) ? v[4] * inv : 0.0f);
        v[5] = ((has_values) ? v[5] * inv : 0.0f);
        v[6] = ((has_values) ? v[6] * inv : 0.0f);
        v[7] = ((has_values) ? v[7] * inv : 0.0f);
        v[8] = ((has_values) ? v[8] * inv : 0.0f);
        v[9] = ((has_values) ? v[9] * inv : 0.0f);
        v[10] = ((has_values) ? v[10] * inv : 0.0f);
        v[11] = ((has_values) ? v[11] * inv : 0.0f);
        v[12] = ((has_values) ? v[12] * inv : 0.0f);
        v[13] = ((has_values) ? v[13] * inv : 0.0f);
        v[14] = ((has_values) ? v[14] * inv : 0.0f);
        v[15] = ((has_values) ? v[15] * inv : 0.0f);
        long long qo_head = (long long)(first_head + head);
        if (num_splits == 1) {
            long long gate_off = (row_i64 * (long long)num_qo_heads + qo_head) * 256 + (long long)dim0;
            long long out_off = row_i64 * out_stride_row + qo_head * out_stride_head + (long long)dim0;
            float _vec_load_8[8];
            {
                const uint4* _vptr_169 = reinterpret_cast<const uint4*>(gate + gate_off + 0);
                uint4 _vld_169[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_169[_blk] = _vptr_169[_blk];
                    uint32_t* _vpairs_169 = reinterpret_cast<uint32_t*>(&_vld_169[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) << 16);
                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn = (float)(__nv_bfloat16)v[0];
            float _exp2_170 = approx_exp2((-_vec_load_8[0]) * 1.442695f);
            float sigmoid = approx_rcp(1.0f + _exp2_170);
            gated[0] = attn * sigmoid;
            float attn_0 = (float)(__nv_bfloat16)v[1];
            float _exp2_171 = approx_exp2((-_vec_load_8[1]) * 1.442695f);
            float sigmoid_1 = approx_rcp(1.0f + _exp2_171);
            gated[1] = attn_0 * sigmoid_1;
            float attn_2 = (float)(__nv_bfloat16)v[2];
            float _exp2_172 = approx_exp2((-_vec_load_8[2]) * 1.442695f);
            float sigmoid_3 = approx_rcp(1.0f + _exp2_172);
            gated[2] = attn_2 * sigmoid_3;
            float attn_4 = (float)(__nv_bfloat16)v[3];
            float _exp2_173 = approx_exp2((-_vec_load_8[3]) * 1.442695f);
            float sigmoid_5 = approx_rcp(1.0f + _exp2_173);
            gated[3] = attn_4 * sigmoid_5;
            float attn_6 = (float)(__nv_bfloat16)v[4];
            float _exp2_174 = approx_exp2((-_vec_load_8[4]) * 1.442695f);
            float sigmoid_7 = approx_rcp(1.0f + _exp2_174);
            gated[4] = attn_6 * sigmoid_7;
            float attn_8 = (float)(__nv_bfloat16)v[5];
            float _exp2_175 = approx_exp2((-_vec_load_8[5]) * 1.442695f);
            float sigmoid_9 = approx_rcp(1.0f + _exp2_175);
            gated[5] = attn_8 * sigmoid_9;
            float attn_10 = (float)(__nv_bfloat16)v[6];
            float _exp2_176 = approx_exp2((-_vec_load_8[6]) * 1.442695f);
            float sigmoid_11 = approx_rcp(1.0f + _exp2_176);
            gated[6] = attn_10 * sigmoid_11;
            float attn_12 = (float)(__nv_bfloat16)v[7];
            float _exp2_177 = approx_exp2((-_vec_load_8[7]) * 1.442695f);
            float sigmoid_13 = approx_rcp(1.0f + _exp2_177);
            gated[7] = attn_12 * sigmoid_13;
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(out + out_off)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            float _vec_load_9[8];
            {
                const uint4* _vptr_178 = reinterpret_cast<const uint4*>(gate + (gate_off + 8) + 0);
                uint4 _vld_178[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_178[_blk] = _vptr_178[_blk];
                    uint32_t* _vpairs_178 = reinterpret_cast<uint32_t*>(&_vld_178[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_178[_pair]) << 16);
                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_178[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_14 = (float)(__nv_bfloat16)v[8];
            float _exp2_179 = approx_exp2((-_vec_load_9[0]) * 1.442695f);
            float sigmoid_15 = approx_rcp(1.0f + _exp2_179);
            gated[0] = attn_14 * sigmoid_15;
            float attn_16 = (float)(__nv_bfloat16)v[9];
            float _exp2_180 = approx_exp2((-_vec_load_9[1]) * 1.442695f);
            float sigmoid_17 = approx_rcp(1.0f + _exp2_180);
            gated[1] = attn_16 * sigmoid_17;
            float attn_18 = (float)(__nv_bfloat16)v[10];
            float _exp2_181 = approx_exp2((-_vec_load_9[2]) * 1.442695f);
            float sigmoid_19 = approx_rcp(1.0f + _exp2_181);
            gated[2] = attn_18 * sigmoid_19;
            float attn_20 = (float)(__nv_bfloat16)v[11];
            float _exp2_182 = approx_exp2((-_vec_load_9[3]) * 1.442695f);
            float sigmoid_21 = approx_rcp(1.0f + _exp2_182);
            gated[3] = attn_20 * sigmoid_21;
            float attn_22 = (float)(__nv_bfloat16)v[12];
            float _exp2_183 = approx_exp2((-_vec_load_9[4]) * 1.442695f);
            float sigmoid_23 = approx_rcp(1.0f + _exp2_183);
            gated[4] = attn_22 * sigmoid_23;
            float attn_24 = (float)(__nv_bfloat16)v[13];
            float _exp2_184 = approx_exp2((-_vec_load_9[5]) * 1.442695f);
            float sigmoid_25 = approx_rcp(1.0f + _exp2_184);
            gated[5] = attn_24 * sigmoid_25;
            float attn_26 = (float)(__nv_bfloat16)v[14];
            float _exp2_185 = approx_exp2((-_vec_load_9[6]) * 1.442695f);
            float sigmoid_27 = approx_rcp(1.0f + _exp2_185);
            gated[6] = attn_26 * sigmoid_27;
            float attn_28 = (float)(__nv_bfloat16)v[15];
            float _exp2_186 = approx_exp2((-_vec_load_9[7]) * 1.442695f);
            float sigmoid_29 = approx_rcp(1.0f + _exp2_186);
            gated[7] = attn_28 * sigmoid_29;
            unsigned int _packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(out + (out_off + 8))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        } else {
            long long partial_row = ((long long)split * (long long)num_rows + row_i64) * (long long)num_qo_heads + qo_head;
            long long partial_off = partial_row * 256 + (long long)dim0;
            {
                float4 _v4 = make_float4(v[0 + 0], v[0 + 1], v[0 + 2], v[0 + 3]);
                *reinterpret_cast<float4*>((partial_out + partial_off) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(v[4 + 0], v[4 + 1], v[4 + 2], v[4 + 3]);
                *reinterpret_cast<float4*>((partial_out + (partial_off + 4)) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(v[8 + 0], v[8 + 1], v[8 + 2], v[8 + 3]);
                *reinterpret_cast<float4*>((partial_out + (partial_off + 8)) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(v[12 + 0], v[12 + 1], v[12 + 2], v[12 + 3]);
                *reinterpret_cast<float4*>((partial_out + (partial_off + 12)) + 0) = _v4;
            }
            if (warp_1 == 0 && t == 0) {
                float _log2_187;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_187) : "f"(l[0]));
                *(reinterpret_cast<float*>(partial_lse + partial_row) + (0)) = ((has_values) ? m[0] + _log2_187 : -CUDART_INF_F);
            }
        }
    }
    int head_61 = g + 8;
    if (head_61 < group_size) {
        bool has_values_1 = l[1] > 0.0f;
        float _rcp_188 = approx_rcp(fmaxf(l[1], 1e-20f));
        float inv_1 = output_scale * _rcp_188;
        v[0] = o[2];
        v[1] = o[6];
        v[8] = o[3];
        v[9] = o[7];
        v[2] = o[10];
        v[3] = o[14];
        v[10] = o[11];
        v[11] = o[15];
        v[4] = o[18];
        v[5] = o[22];
        v[12] = o[19];
        v[13] = o[23];
        v[6] = o[26];
        v[7] = o[30];
        v[14] = o[27];
        v[15] = o[31];
        v[0] = ((has_values_1) ? v[0] * inv_1 : 0.0f);
        v[1] = ((has_values_1) ? v[1] * inv_1 : 0.0f);
        v[2] = ((has_values_1) ? v[2] * inv_1 : 0.0f);
        v[3] = ((has_values_1) ? v[3] * inv_1 : 0.0f);
        v[4] = ((has_values_1) ? v[4] * inv_1 : 0.0f);
        v[5] = ((has_values_1) ? v[5] * inv_1 : 0.0f);
        v[6] = ((has_values_1) ? v[6] * inv_1 : 0.0f);
        v[7] = ((has_values_1) ? v[7] * inv_1 : 0.0f);
        v[8] = ((has_values_1) ? v[8] * inv_1 : 0.0f);
        v[9] = ((has_values_1) ? v[9] * inv_1 : 0.0f);
        v[10] = ((has_values_1) ? v[10] * inv_1 : 0.0f);
        v[11] = ((has_values_1) ? v[11] * inv_1 : 0.0f);
        v[12] = ((has_values_1) ? v[12] * inv_1 : 0.0f);
        v[13] = ((has_values_1) ? v[13] * inv_1 : 0.0f);
        v[14] = ((has_values_1) ? v[14] * inv_1 : 0.0f);
        v[15] = ((has_values_1) ? v[15] * inv_1 : 0.0f);
        long long qo_head_1 = (long long)(first_head + head_61);
        if (num_splits == 1) {
            long long gate_off_1 = (row_i64 * (long long)num_qo_heads + qo_head_1) * 256 + (long long)dim0;
            long long out_off_1 = row_i64 * out_stride_row + qo_head_1 * out_stride_head + (long long)dim0;
            float _vec_load_10[8];
            {
                const uint4* _vptr_189 = reinterpret_cast<const uint4*>(gate + gate_off_1 + 0);
                uint4 _vld_189[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_189[_blk] = _vptr_189[_blk];
                    uint32_t* _vpairs_189 = reinterpret_cast<uint32_t*>(&_vld_189[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_189[_pair]) << 16);
                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_189[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_1 = (float)(__nv_bfloat16)v[0];
            float _exp2_190 = approx_exp2((-_vec_load_10[0]) * 1.442695f);
            float sigmoid_2 = approx_rcp(1.0f + _exp2_190);
            gated[0] = attn_1 * sigmoid_2;
            float attn_0_1 = (float)(__nv_bfloat16)v[1];
            float _exp2_191 = approx_exp2((-_vec_load_10[1]) * 1.442695f);
            float sigmoid_1_1 = approx_rcp(1.0f + _exp2_191);
            gated[1] = attn_0_1 * sigmoid_1_1;
            float attn_2_1 = (float)(__nv_bfloat16)v[2];
            float _exp2_192 = approx_exp2((-_vec_load_10[2]) * 1.442695f);
            float sigmoid_3_1 = approx_rcp(1.0f + _exp2_192);
            gated[2] = attn_2_1 * sigmoid_3_1;
            float attn_4_1 = (float)(__nv_bfloat16)v[3];
            float _exp2_193 = approx_exp2((-_vec_load_10[3]) * 1.442695f);
            float sigmoid_5_1 = approx_rcp(1.0f + _exp2_193);
            gated[3] = attn_4_1 * sigmoid_5_1;
            float attn_6_1 = (float)(__nv_bfloat16)v[4];
            float _exp2_194 = approx_exp2((-_vec_load_10[4]) * 1.442695f);
            float sigmoid_7_1 = approx_rcp(1.0f + _exp2_194);
            gated[4] = attn_6_1 * sigmoid_7_1;
            float attn_8_1 = (float)(__nv_bfloat16)v[5];
            float _exp2_195 = approx_exp2((-_vec_load_10[5]) * 1.442695f);
            float sigmoid_9_1 = approx_rcp(1.0f + _exp2_195);
            gated[5] = attn_8_1 * sigmoid_9_1;
            float attn_10_1 = (float)(__nv_bfloat16)v[6];
            float _exp2_196 = approx_exp2((-_vec_load_10[6]) * 1.442695f);
            float sigmoid_11_1 = approx_rcp(1.0f + _exp2_196);
            gated[6] = attn_10_1 * sigmoid_11_1;
            float attn_12_1 = (float)(__nv_bfloat16)v[7];
            float _exp2_197 = approx_exp2((-_vec_load_10[7]) * 1.442695f);
            float sigmoid_13_1 = approx_rcp(1.0f + _exp2_197);
            gated[7] = attn_12_1 * sigmoid_13_1;
            unsigned int _packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(out + out_off_1)[0] = reinterpret_cast<int4*>(_packed_2)[0];
            float _vec_load_11[8];
            {
                const uint4* _vptr_198 = reinterpret_cast<const uint4*>(gate + (gate_off_1 + 8) + 0);
                uint4 _vld_198[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_198[_blk] = _vptr_198[_blk];
                    uint32_t* _vpairs_198 = reinterpret_cast<uint32_t*>(&_vld_198[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_198[_pair]) << 16);
                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_198[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_14_1 = (float)(__nv_bfloat16)v[8];
            float _exp2_199 = approx_exp2((-_vec_load_11[0]) * 1.442695f);
            float sigmoid_15_1 = approx_rcp(1.0f + _exp2_199);
            gated[0] = attn_14_1 * sigmoid_15_1;
            float attn_16_1 = (float)(__nv_bfloat16)v[9];
            float _exp2_200 = approx_exp2((-_vec_load_11[1]) * 1.442695f);
            float sigmoid_17_1 = approx_rcp(1.0f + _exp2_200);
            gated[1] = attn_16_1 * sigmoid_17_1;
            float attn_18_1 = (float)(__nv_bfloat16)v[10];
            float _exp2_201 = approx_exp2((-_vec_load_11[2]) * 1.442695f);
            float sigmoid_19_1 = approx_rcp(1.0f + _exp2_201);
            gated[2] = attn_18_1 * sigmoid_19_1;
            float attn_20_1 = (float)(__nv_bfloat16)v[11];
            float _exp2_202 = approx_exp2((-_vec_load_11[3]) * 1.442695f);
            float sigmoid_21_1 = approx_rcp(1.0f + _exp2_202);
            gated[3] = attn_20_1 * sigmoid_21_1;
            float attn_22_1 = (float)(__nv_bfloat16)v[12];
            float _exp2_203 = approx_exp2((-_vec_load_11[4]) * 1.442695f);
            float sigmoid_23_1 = approx_rcp(1.0f + _exp2_203);
            gated[4] = attn_22_1 * sigmoid_23_1;
            float attn_24_1 = (float)(__nv_bfloat16)v[13];
            float _exp2_204 = approx_exp2((-_vec_load_11[5]) * 1.442695f);
            float sigmoid_25_1 = approx_rcp(1.0f + _exp2_204);
            gated[5] = attn_24_1 * sigmoid_25_1;
            float attn_26_1 = (float)(__nv_bfloat16)v[14];
            float _exp2_205 = approx_exp2((-_vec_load_11[6]) * 1.442695f);
            float sigmoid_27_1 = approx_rcp(1.0f + _exp2_205);
            gated[6] = attn_26_1 * sigmoid_27_1;
            float attn_28_1 = (float)(__nv_bfloat16)v[15];
            float _exp2_206 = approx_exp2((-_vec_load_11[7]) * 1.442695f);
            float sigmoid_29_1 = approx_rcp(1.0f + _exp2_206);
            gated[7] = attn_28_1 * sigmoid_29_1;
            unsigned int _packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(out + (out_off_1 + 8))[0] = reinterpret_cast<int4*>(_packed_3)[0];
        } else {
            long long partial_row_1 = ((long long)split * (long long)num_rows + row_i64) * (long long)num_qo_heads + qo_head_1;
            long long partial_off_1 = partial_row_1 * 256 + (long long)dim0;
            {
                float4 _v4 = make_float4(v[0 + 0], v[0 + 1], v[0 + 2], v[0 + 3]);
                *reinterpret_cast<float4*>((partial_out + partial_off_1) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(v[4 + 0], v[4 + 1], v[4 + 2], v[4 + 3]);
                *reinterpret_cast<float4*>((partial_out + (partial_off_1 + 4)) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(v[8 + 0], v[8 + 1], v[8 + 2], v[8 + 3]);
                *reinterpret_cast<float4*>((partial_out + (partial_off_1 + 8)) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(v[12 + 0], v[12 + 1], v[12 + 2], v[12 + 3]);
                *reinterpret_cast<float4*>((partial_out + (partial_off_1 + 12)) + 0) = _v4;
            }
            if (warp_1 == 0 && t == 0) {
                float _log2_207;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_207) : "f"(l[1]));
                *(reinterpret_cast<float*>(partial_lse + partial_row_1) + (0)) = ((has_values_1) ? m[1] + _log2_207 : -CUDART_INF_F);
            }
        }
    }
    if ((unsigned long long)merge_counters != 0) {
        if (num_splits > 1) {
            __syncthreads();
            if (tid_0 == 0) {
                int cidx = row * gridDim.y + kv_head;
                unsigned int flip = 2147483648;
                unsigned int nb = 1;
                if (split == 0) {
                    nb = flip - (unsigned int)(num_splits - 1);
                }
                unsigned int old;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(old) : "l"(&merge_counters[cidx]), "r"(static_cast<uint32_t>(nb)) : "memory");
                unsigned int target = (old ^ flip) & flip;
                {
                    unsigned int _poll_v_208;
                    unsigned long long _poll_t0_208;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_poll_t0_208) :: "memory");
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_poll_v_208) : "l"((reinterpret_cast<unsigned int*>(merge_counters) + (cidx))) : "memory");
                    while (((_poll_v_208 ^ static_cast<unsigned int>(target)) & static_cast<unsigned int>(2147483648)) != 0) {
                        unsigned long long _poll_now_208;
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_poll_now_208) :: "memory");
                        if (_poll_now_208 - _poll_t0_208 > 4000000000ull) {
                            asm volatile("trap;" ::: "memory");
                        }
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_poll_v_208) : "l"((reinterpret_cast<unsigned int*>(merge_counters) + (cidx))) : "memory");
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            int pairs_total = group_size * 128;
            int chunk = (pairs_total + num_splits - 1) / num_splits;
            int begin = split * chunk;
            int end = ((begin + chunk) < (pairs_total) ? (begin + chunk) : (pairs_total));
            int p_first = begin + tid_0;
            long long split_pitch = (long long)num_rows * (long long)num_qo_heads;
            long long row_base = row_i64 * (long long)num_qo_heads + (long long)first_head;
            int head_first = begin >> 7;
            int n_heads = 0;
            if (end > begin) {
                n_heads = (end - 1 >> 7) - head_first + 1;
            }
            if (num_splits > 16) {
                int n_lse = n_heads * num_splits;
                #pragma unroll 1
                for (int i = tid_0; i < n_lse; i += 128) {
                    int h = i / num_splits;
                    int s_0 = i - h * num_splits;
                    lse_tab[i] = partial_lse[(long long)s_0 * split_pitch + row_base + (long long)(head_first + h)];
                }
                __syncthreads();
                #pragma unroll 1
                for (int h_1 = warp_1; h_1 < n_heads; h_1 += 4) {
                    float hmax = -CUDART_INF_F;
                    #pragma unroll 1
                    for (int s_1 = lane_1; s_1 < num_splits; s_1 += 32) {
                        hmax = fmaxf(hmax, lse_tab[h_1 * num_splits + s_1]);
                    }
                    float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, hmax, 16);
                    hmax = fmaxf(hmax, _shfl_xor_sync_209);
                    float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, hmax, 8);
                    hmax = fmaxf(hmax, _shfl_xor_sync_210);
                    float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, hmax, 4);
                    hmax = fmaxf(hmax, _shfl_xor_sync_211);
                    float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, hmax, 2);
                    hmax = fmaxf(hmax, _shfl_xor_sync_212);
                    float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, hmax, 1);
                    hmax = fmaxf(hmax, _shfl_xor_sync_213);
                    if (lane_1 == 0) {
                        max_tab[h_1] = hmax;
                    }
                }
                __syncthreads();
                float lse[16];
                float p0[16];
                float p1[16];
                float acc0[1];
                float acc1[1];
                float den[1];
                float lse_max[1];
                int valid_0_4[1];
                int tab[1];
                long long dim_1[1];
                long long qo_head_2[1];
                long long row_head[1];
                float gate0[1];
                float gate1[1];
                unsigned int word[1];
                long long nqo = (long long)num_qo_heads;
                #pragma unroll 1
                for (int p = p_first; p < end; p += 128) {
                    int pk = ((p) < (end - 1) ? (p) : (end - 1));
                    valid_0_4[0] = ((end > p) ? 1 : 0);
                    int head_0 = pk >> 7;
                    dim_1[0] = (long long)((pk & 127) * 2);
                    qo_head_2[0] = (long long)(first_head + head_0);
                    row_head[0] = row_i64 * nqo + qo_head_2[0];
                    acc0[0] = 0.0f;
                    acc1[0] = 0.0f;
                    den[0] = 0.0f;
                    tab[0] = (head_0 - head_first) * num_splits;
                    lse_max[0] = max_tab[head_0 - head_first];
                    float _vec_load_12[2];
                    {
                        uint32_t _bf16x2_bits_214;
                        _bf16x2_bits_214 = *reinterpret_cast<const uint32_t*>(gate + (row_head[0] * 256 + dim_1[0]) + 0);
                        (&_vec_load_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_214) << 16);
                        (&_vec_load_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_214) & 0xffff0000u);
                    }
                    gate0[0] = _vec_load_12[0];
                    gate1[0] = _vec_load_12[1];
                    #pragma unroll 1
                    for (int base = 0; base < num_splits; base += 16) {
                        if (base < num_splits) {
                            long long part_row = (long long)base * split_pitch + row_head[0];
                            float _vec_load_13[2];
                            {
                                float2 _v2_215 = *reinterpret_cast<const float2*>(partial_out + (part_row * 256 + dim_1[0]) + 0);
                                _vec_load_13[0] = _v2_215.x;
                                _vec_load_13[0 + 1] = _v2_215.y;
                            }
                            p0[0] = _vec_load_13[0];
                            p1[0] = _vec_load_13[1];
                        }
                        if (base + 1 < num_splits) {
                            long long part_row_1 = (long long)(base + 1) * split_pitch + row_head[0];
                            float _vec_load_14[2];
                            {
                                float2 _v2_216 = *reinterpret_cast<const float2*>(partial_out + (part_row_1 * 256 + dim_1[0]) + 0);
                                _vec_load_14[0] = _v2_216.x;
                                _vec_load_14[0 + 1] = _v2_216.y;
                            }
                            p0[1] = _vec_load_14[0];
                            p1[1] = _vec_load_14[1];
                        }
                        if (base + 2 < num_splits) {
                            long long part_row_2 = (long long)(base + 2) * split_pitch + row_head[0];
                            float _vec_load_15[2];
                            {
                                float2 _v2_217 = *reinterpret_cast<const float2*>(partial_out + (part_row_2 * 256 + dim_1[0]) + 0);
                                _vec_load_15[0] = _v2_217.x;
                                _vec_load_15[0 + 1] = _v2_217.y;
                            }
                            p0[2] = _vec_load_15[0];
                            p1[2] = _vec_load_15[1];
                        }
                        if (base + 3 < num_splits) {
                            long long part_row_3 = (long long)(base + 3) * split_pitch + row_head[0];
                            float _vec_load_16[2];
                            {
                                float2 _v2_218 = *reinterpret_cast<const float2*>(partial_out + (part_row_3 * 256 + dim_1[0]) + 0);
                                _vec_load_16[0] = _v2_218.x;
                                _vec_load_16[0 + 1] = _v2_218.y;
                            }
                            p0[3] = _vec_load_16[0];
                            p1[3] = _vec_load_16[1];
                        }
                        if (base + 4 < num_splits) {
                            long long part_row_4 = (long long)(base + 4) * split_pitch + row_head[0];
                            float _vec_load_17[2];
                            {
                                float2 _v2_219 = *reinterpret_cast<const float2*>(partial_out + (part_row_4 * 256 + dim_1[0]) + 0);
                                _vec_load_17[0] = _v2_219.x;
                                _vec_load_17[0 + 1] = _v2_219.y;
                            }
                            p0[4] = _vec_load_17[0];
                            p1[4] = _vec_load_17[1];
                        }
                        if (base + 5 < num_splits) {
                            long long part_row_5 = (long long)(base + 5) * split_pitch + row_head[0];
                            float _vec_load_18[2];
                            {
                                float2 _v2_220 = *reinterpret_cast<const float2*>(partial_out + (part_row_5 * 256 + dim_1[0]) + 0);
                                _vec_load_18[0] = _v2_220.x;
                                _vec_load_18[0 + 1] = _v2_220.y;
                            }
                            p0[5] = _vec_load_18[0];
                            p1[5] = _vec_load_18[1];
                        }
                        if (base + 6 < num_splits) {
                            long long part_row_6 = (long long)(base + 6) * split_pitch + row_head[0];
                            float _vec_load_19[2];
                            {
                                float2 _v2_221 = *reinterpret_cast<const float2*>(partial_out + (part_row_6 * 256 + dim_1[0]) + 0);
                                _vec_load_19[0] = _v2_221.x;
                                _vec_load_19[0 + 1] = _v2_221.y;
                            }
                            p0[6] = _vec_load_19[0];
                            p1[6] = _vec_load_19[1];
                        }
                        if (base + 7 < num_splits) {
                            long long part_row_7 = (long long)(base + 7) * split_pitch + row_head[0];
                            float _vec_load_20[2];
                            {
                                float2 _v2_222 = *reinterpret_cast<const float2*>(partial_out + (part_row_7 * 256 + dim_1[0]) + 0);
                                _vec_load_20[0] = _v2_222.x;
                                _vec_load_20[0 + 1] = _v2_222.y;
                            }
                            p0[7] = _vec_load_20[0];
                            p1[7] = _vec_load_20[1];
                        }
                        if (base + 8 < num_splits) {
                            long long part_row_8 = (long long)(base + 8) * split_pitch + row_head[0];
                            float _vec_load_21[2];
                            {
                                float2 _v2_223 = *reinterpret_cast<const float2*>(partial_out + (part_row_8 * 256 + dim_1[0]) + 0);
                                _vec_load_21[0] = _v2_223.x;
                                _vec_load_21[0 + 1] = _v2_223.y;
                            }
                            p0[8] = _vec_load_21[0];
                            p1[8] = _vec_load_21[1];
                        }
                        if (base + 9 < num_splits) {
                            long long part_row_9 = (long long)(base + 9) * split_pitch + row_head[0];
                            float _vec_load_22[2];
                            {
                                float2 _v2_224 = *reinterpret_cast<const float2*>(partial_out + (part_row_9 * 256 + dim_1[0]) + 0);
                                _vec_load_22[0] = _v2_224.x;
                                _vec_load_22[0 + 1] = _v2_224.y;
                            }
                            p0[9] = _vec_load_22[0];
                            p1[9] = _vec_load_22[1];
                        }
                        if (base + 10 < num_splits) {
                            long long part_row_10 = (long long)(base + 10) * split_pitch + row_head[0];
                            float _vec_load_23[2];
                            {
                                float2 _v2_225 = *reinterpret_cast<const float2*>(partial_out + (part_row_10 * 256 + dim_1[0]) + 0);
                                _vec_load_23[0] = _v2_225.x;
                                _vec_load_23[0 + 1] = _v2_225.y;
                            }
                            p0[10] = _vec_load_23[0];
                            p1[10] = _vec_load_23[1];
                        }
                        if (base + 11 < num_splits) {
                            long long part_row_11 = (long long)(base + 11) * split_pitch + row_head[0];
                            float _vec_load_24[2];
                            {
                                float2 _v2_226 = *reinterpret_cast<const float2*>(partial_out + (part_row_11 * 256 + dim_1[0]) + 0);
                                _vec_load_24[0] = _v2_226.x;
                                _vec_load_24[0 + 1] = _v2_226.y;
                            }
                            p0[11] = _vec_load_24[0];
                            p1[11] = _vec_load_24[1];
                        }
                        if (base + 12 < num_splits) {
                            long long part_row_12 = (long long)(base + 12) * split_pitch + row_head[0];
                            float _vec_load_25[2];
                            {
                                float2 _v2_227 = *reinterpret_cast<const float2*>(partial_out + (part_row_12 * 256 + dim_1[0]) + 0);
                                _vec_load_25[0] = _v2_227.x;
                                _vec_load_25[0 + 1] = _v2_227.y;
                            }
                            p0[12] = _vec_load_25[0];
                            p1[12] = _vec_load_25[1];
                        }
                        if (base + 13 < num_splits) {
                            long long part_row_13 = (long long)(base + 13) * split_pitch + row_head[0];
                            float _vec_load_26[2];
                            {
                                float2 _v2_228 = *reinterpret_cast<const float2*>(partial_out + (part_row_13 * 256 + dim_1[0]) + 0);
                                _vec_load_26[0] = _v2_228.x;
                                _vec_load_26[0 + 1] = _v2_228.y;
                            }
                            p0[13] = _vec_load_26[0];
                            p1[13] = _vec_load_26[1];
                        }
                        if (base + 14 < num_splits) {
                            long long part_row_14 = (long long)(base + 14) * split_pitch + row_head[0];
                            float _vec_load_27[2];
                            {
                                float2 _v2_229 = *reinterpret_cast<const float2*>(partial_out + (part_row_14 * 256 + dim_1[0]) + 0);
                                _vec_load_27[0] = _v2_229.x;
                                _vec_load_27[0 + 1] = _v2_229.y;
                            }
                            p0[14] = _vec_load_27[0];
                            p1[14] = _vec_load_27[1];
                        }
                        if (base + 15 < num_splits) {
                            long long part_row_15 = (long long)(base + 15) * split_pitch + row_head[0];
                            float _vec_load_28[2];
                            {
                                float2 _v2_230 = *reinterpret_cast<const float2*>(partial_out + (part_row_15 * 256 + dim_1[0]) + 0);
                                _vec_load_28[0] = _v2_230.x;
                                _vec_load_28[0 + 1] = _v2_230.y;
                            }
                            p0[15] = _vec_load_28[0];
                            p1[15] = _vec_load_28[1];
                        }
                        bool live = base < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live) {
                            float weight = approx_exp2(lse_tab[tab[0] + base] - lse_max[0]);
                            den[0] = den[0] + weight;
                            acc0[0] = __fmaf_rn(weight, p0[0], acc0[0]);
                            acc1[0] = __fmaf_rn(weight, p1[0], acc1[0]);
                        }
                        bool live_0 = base + 1 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_0) {
                            float weight_1 = approx_exp2(lse_tab[tab[0] + base + 1] - lse_max[0]);
                            den[0] = den[0] + weight_1;
                            acc0[0] = __fmaf_rn(weight_1, p0[1], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1, p1[1], acc1[0]);
                        }
                        bool live_1 = base + 2 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_1) {
                            float weight_2 = approx_exp2(lse_tab[tab[0] + base + 2] - lse_max[0]);
                            den[0] = den[0] + weight_2;
                            acc0[0] = __fmaf_rn(weight_2, p0[2], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_2, p1[2], acc1[0]);
                        }
                        bool live_2 = base + 3 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_2) {
                            float weight_3 = approx_exp2(lse_tab[tab[0] + base + 3] - lse_max[0]);
                            den[0] = den[0] + weight_3;
                            acc0[0] = __fmaf_rn(weight_3, p0[3], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_3, p1[3], acc1[0]);
                        }
                        bool live_3 = base + 4 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_3) {
                            float weight_4 = approx_exp2(lse_tab[tab[0] + base + 4] - lse_max[0]);
                            den[0] = den[0] + weight_4;
                            acc0[0] = __fmaf_rn(weight_4, p0[4], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_4, p1[4], acc1[0]);
                        }
                        bool live_4 = base + 5 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_4) {
                            float weight_5 = approx_exp2(lse_tab[tab[0] + base + 5] - lse_max[0]);
                            den[0] = den[0] + weight_5;
                            acc0[0] = __fmaf_rn(weight_5, p0[5], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_5, p1[5], acc1[0]);
                        }
                        bool live_5 = base + 6 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_5) {
                            float weight_6 = approx_exp2(lse_tab[tab[0] + base + 6] - lse_max[0]);
                            den[0] = den[0] + weight_6;
                            acc0[0] = __fmaf_rn(weight_6, p0[6], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_6, p1[6], acc1[0]);
                        }
                        bool live_6 = base + 7 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_6) {
                            float weight_7 = approx_exp2(lse_tab[tab[0] + base + 7] - lse_max[0]);
                            den[0] = den[0] + weight_7;
                            acc0[0] = __fmaf_rn(weight_7, p0[7], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_7, p1[7], acc1[0]);
                        }
                        bool live_7 = base + 8 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_7) {
                            float weight_8 = approx_exp2(lse_tab[tab[0] + base + 8] - lse_max[0]);
                            den[0] = den[0] + weight_8;
                            acc0[0] = __fmaf_rn(weight_8, p0[8], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_8, p1[8], acc1[0]);
                        }
                        bool live_8 = base + 9 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_8) {
                            float weight_9 = approx_exp2(lse_tab[tab[0] + base + 9] - lse_max[0]);
                            den[0] = den[0] + weight_9;
                            acc0[0] = __fmaf_rn(weight_9, p0[9], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_9, p1[9], acc1[0]);
                        }
                        bool live_9 = base + 10 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_9) {
                            float weight_10 = approx_exp2(lse_tab[tab[0] + base + 10] - lse_max[0]);
                            den[0] = den[0] + weight_10;
                            acc0[0] = __fmaf_rn(weight_10, p0[10], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_10, p1[10], acc1[0]);
                        }
                        bool live_10 = base + 11 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_10) {
                            float weight_11 = approx_exp2(lse_tab[tab[0] + base + 11] - lse_max[0]);
                            den[0] = den[0] + weight_11;
                            acc0[0] = __fmaf_rn(weight_11, p0[11], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_11, p1[11], acc1[0]);
                        }
                        bool live_11 = base + 12 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_11) {
                            float weight_12 = approx_exp2(lse_tab[tab[0] + base + 12] - lse_max[0]);
                            den[0] = den[0] + weight_12;
                            acc0[0] = __fmaf_rn(weight_12, p0[12], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_12, p1[12], acc1[0]);
                        }
                        bool live_12 = base + 13 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_12) {
                            float weight_13 = approx_exp2(lse_tab[tab[0] + base + 13] - lse_max[0]);
                            den[0] = den[0] + weight_13;
                            acc0[0] = __fmaf_rn(weight_13, p0[13], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_13, p1[13], acc1[0]);
                        }
                        bool live_13 = base + 14 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_13) {
                            float weight_14 = approx_exp2(lse_tab[tab[0] + base + 14] - lse_max[0]);
                            den[0] = den[0] + weight_14;
                            acc0[0] = __fmaf_rn(weight_14, p0[14], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_14, p1[14], acc1[0]);
                        }
                        bool live_14 = base + 15 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_14) {
                            float weight_15 = approx_exp2(lse_tab[tab[0] + base + 15] - lse_max[0]);
                            den[0] = den[0] + weight_15;
                            acc0[0] = __fmaf_rn(weight_15, p0[15], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_15, p1[15], acc1[0]);
                        }
                    }
                    if (valid_0_4[0] != 0) {
                        float inv_2 = 0.0f;
                        if (den[0] > 0.0f) {
                            float _rcp_231 = approx_rcp(den[0]);
                            inv_2 = _rcp_231;
                        }
                        float attn0 = (float)(__nv_bfloat16)(acc0[0] * inv_2);
                        float attn1 = (float)(__nv_bfloat16)(acc1[0] * inv_2);
                        float _exp2_233 = approx_exp2((-gate0[0]) * 1.442695f);
                        float _rcp_232 = approx_rcp(1.0f + _exp2_233);
                        float result0 = attn0 * _rcp_232;
                        float _exp2_235 = approx_exp2((-gate1[0]) * 1.442695f);
                        float _rcp_234 = approx_rcp(1.0f + _exp2_235);
                        float result1 = attn1 * _rcp_234;
                        __nv_bfloat162 _bit_cast_src_236 = __float22bfloat162_rn(make_float2(result0, result1));
                        word[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_236)[0];
                        long long out_offset = row_i64 * out_stride_row + qo_head_2[0] * out_stride_head + dim_1[0];
                        *(reinterpret_cast<unsigned int*>(out + out_offset) + (0)) = word[0];
                    }
                }
            } else if (num_splits > 8) {
                float lse_1[16];
                float p0_1[16];
                float p1_1[16];
                float acc0_1[1];
                float acc1_1[1];
                float den_1[1];
                float lse_max_1[1];
                int valid_0_5[1];
                int tab_1[1];
                long long dim_1_1[1];
                long long qo_head_3[1];
                long long row_head_1[1];
                float gate0_1[1];
                float gate1_1[1];
                unsigned int word_1[1];
                long long nqo_1 = (long long)num_qo_heads;
                #pragma unroll 1
                for (int p_1 = p_first; p_1 < end; p_1 += 128) {
                    int pk_1 = ((p_1) < (end - 1) ? (p_1) : (end - 1));
                    valid_0_5[0] = ((end > p_1) ? 1 : 0);
                    int head_0_1 = pk_1 >> 7;
                    dim_1_1[0] = (long long)((pk_1 & 127) * 2);
                    qo_head_3[0] = (long long)(first_head + head_0_1);
                    row_head_1[0] = row_i64 * nqo_1 + qo_head_3[0];
                    acc0_1[0] = 0.0f;
                    acc1_1[0] = 0.0f;
                    den_1[0] = 0.0f;
                    lse_max_1[0] = -CUDART_INF_F;
                    float _vec_load_29[2];
                    {
                        uint32_t _bf16x2_bits_237;
                        _bf16x2_bits_237 = *reinterpret_cast<const uint32_t*>(gate + (row_head_1[0] * 256 + dim_1_1[0]) + 0);
                        (&_vec_load_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_237) << 16);
                        (&_vec_load_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_237) & 0xffff0000u);
                    }
                    gate0_1[0] = _vec_load_29[0];
                    gate1_1[0] = _vec_load_29[1];
                    #pragma unroll 1
                    for (int base_1 = 0; base_1 < num_splits; base_1 += 16) {
                        lse_1[0] = -CUDART_INF_F;
                        if (base_1 < num_splits) {
                            long long part_row_16 = (long long)base_1 * split_pitch + row_head_1[0];
                            lse_1[0] = partial_lse[part_row_16];
                            float _vec_load_30[2];
                            {
                                float2 _v2_238 = *reinterpret_cast<const float2*>(partial_out + (part_row_16 * 256 + dim_1_1[0]) + 0);
                                _vec_load_30[0] = _v2_238.x;
                                _vec_load_30[0 + 1] = _v2_238.y;
                            }
                            p0_1[0] = _vec_load_30[0];
                            p1_1[0] = _vec_load_30[1];
                        }
                        lse_1[1] = -CUDART_INF_F;
                        if (base_1 + 1 < num_splits) {
                            long long part_row_17 = (long long)(base_1 + 1) * split_pitch + row_head_1[0];
                            lse_1[1] = partial_lse[part_row_17];
                            float _vec_load_31[2];
                            {
                                float2 _v2_239 = *reinterpret_cast<const float2*>(partial_out + (part_row_17 * 256 + dim_1_1[0]) + 0);
                                _vec_load_31[0] = _v2_239.x;
                                _vec_load_31[0 + 1] = _v2_239.y;
                            }
                            p0_1[1] = _vec_load_31[0];
                            p1_1[1] = _vec_load_31[1];
                        }
                        lse_1[2] = -CUDART_INF_F;
                        if (base_1 + 2 < num_splits) {
                            long long part_row_18 = (long long)(base_1 + 2) * split_pitch + row_head_1[0];
                            lse_1[2] = partial_lse[part_row_18];
                            float _vec_load_32[2];
                            {
                                float2 _v2_240 = *reinterpret_cast<const float2*>(partial_out + (part_row_18 * 256 + dim_1_1[0]) + 0);
                                _vec_load_32[0] = _v2_240.x;
                                _vec_load_32[0 + 1] = _v2_240.y;
                            }
                            p0_1[2] = _vec_load_32[0];
                            p1_1[2] = _vec_load_32[1];
                        }
                        lse_1[3] = -CUDART_INF_F;
                        if (base_1 + 3 < num_splits) {
                            long long part_row_19 = (long long)(base_1 + 3) * split_pitch + row_head_1[0];
                            lse_1[3] = partial_lse[part_row_19];
                            float _vec_load_33[2];
                            {
                                float2 _v2_241 = *reinterpret_cast<const float2*>(partial_out + (part_row_19 * 256 + dim_1_1[0]) + 0);
                                _vec_load_33[0] = _v2_241.x;
                                _vec_load_33[0 + 1] = _v2_241.y;
                            }
                            p0_1[3] = _vec_load_33[0];
                            p1_1[3] = _vec_load_33[1];
                        }
                        lse_1[4] = -CUDART_INF_F;
                        if (base_1 + 4 < num_splits) {
                            long long part_row_20 = (long long)(base_1 + 4) * split_pitch + row_head_1[0];
                            lse_1[4] = partial_lse[part_row_20];
                            float _vec_load_34[2];
                            {
                                float2 _v2_242 = *reinterpret_cast<const float2*>(partial_out + (part_row_20 * 256 + dim_1_1[0]) + 0);
                                _vec_load_34[0] = _v2_242.x;
                                _vec_load_34[0 + 1] = _v2_242.y;
                            }
                            p0_1[4] = _vec_load_34[0];
                            p1_1[4] = _vec_load_34[1];
                        }
                        lse_1[5] = -CUDART_INF_F;
                        if (base_1 + 5 < num_splits) {
                            long long part_row_21 = (long long)(base_1 + 5) * split_pitch + row_head_1[0];
                            lse_1[5] = partial_lse[part_row_21];
                            float _vec_load_35[2];
                            {
                                float2 _v2_243 = *reinterpret_cast<const float2*>(partial_out + (part_row_21 * 256 + dim_1_1[0]) + 0);
                                _vec_load_35[0] = _v2_243.x;
                                _vec_load_35[0 + 1] = _v2_243.y;
                            }
                            p0_1[5] = _vec_load_35[0];
                            p1_1[5] = _vec_load_35[1];
                        }
                        lse_1[6] = -CUDART_INF_F;
                        if (base_1 + 6 < num_splits) {
                            long long part_row_22 = (long long)(base_1 + 6) * split_pitch + row_head_1[0];
                            lse_1[6] = partial_lse[part_row_22];
                            float _vec_load_36[2];
                            {
                                float2 _v2_244 = *reinterpret_cast<const float2*>(partial_out + (part_row_22 * 256 + dim_1_1[0]) + 0);
                                _vec_load_36[0] = _v2_244.x;
                                _vec_load_36[0 + 1] = _v2_244.y;
                            }
                            p0_1[6] = _vec_load_36[0];
                            p1_1[6] = _vec_load_36[1];
                        }
                        lse_1[7] = -CUDART_INF_F;
                        if (base_1 + 7 < num_splits) {
                            long long part_row_23 = (long long)(base_1 + 7) * split_pitch + row_head_1[0];
                            lse_1[7] = partial_lse[part_row_23];
                            float _vec_load_37[2];
                            {
                                float2 _v2_245 = *reinterpret_cast<const float2*>(partial_out + (part_row_23 * 256 + dim_1_1[0]) + 0);
                                _vec_load_37[0] = _v2_245.x;
                                _vec_load_37[0 + 1] = _v2_245.y;
                            }
                            p0_1[7] = _vec_load_37[0];
                            p1_1[7] = _vec_load_37[1];
                        }
                        lse_1[8] = -CUDART_INF_F;
                        if (base_1 + 8 < num_splits) {
                            long long part_row_24 = (long long)(base_1 + 8) * split_pitch + row_head_1[0];
                            lse_1[8] = partial_lse[part_row_24];
                            float _vec_load_38[2];
                            {
                                float2 _v2_246 = *reinterpret_cast<const float2*>(partial_out + (part_row_24 * 256 + dim_1_1[0]) + 0);
                                _vec_load_38[0] = _v2_246.x;
                                _vec_load_38[0 + 1] = _v2_246.y;
                            }
                            p0_1[8] = _vec_load_38[0];
                            p1_1[8] = _vec_load_38[1];
                        }
                        lse_1[9] = -CUDART_INF_F;
                        if (base_1 + 9 < num_splits) {
                            long long part_row_25 = (long long)(base_1 + 9) * split_pitch + row_head_1[0];
                            lse_1[9] = partial_lse[part_row_25];
                            float _vec_load_39[2];
                            {
                                float2 _v2_247 = *reinterpret_cast<const float2*>(partial_out + (part_row_25 * 256 + dim_1_1[0]) + 0);
                                _vec_load_39[0] = _v2_247.x;
                                _vec_load_39[0 + 1] = _v2_247.y;
                            }
                            p0_1[9] = _vec_load_39[0];
                            p1_1[9] = _vec_load_39[1];
                        }
                        lse_1[10] = -CUDART_INF_F;
                        if (base_1 + 10 < num_splits) {
                            long long part_row_26 = (long long)(base_1 + 10) * split_pitch + row_head_1[0];
                            lse_1[10] = partial_lse[part_row_26];
                            float _vec_load_40[2];
                            {
                                float2 _v2_248 = *reinterpret_cast<const float2*>(partial_out + (part_row_26 * 256 + dim_1_1[0]) + 0);
                                _vec_load_40[0] = _v2_248.x;
                                _vec_load_40[0 + 1] = _v2_248.y;
                            }
                            p0_1[10] = _vec_load_40[0];
                            p1_1[10] = _vec_load_40[1];
                        }
                        lse_1[11] = -CUDART_INF_F;
                        if (base_1 + 11 < num_splits) {
                            long long part_row_27 = (long long)(base_1 + 11) * split_pitch + row_head_1[0];
                            lse_1[11] = partial_lse[part_row_27];
                            float _vec_load_41[2];
                            {
                                float2 _v2_249 = *reinterpret_cast<const float2*>(partial_out + (part_row_27 * 256 + dim_1_1[0]) + 0);
                                _vec_load_41[0] = _v2_249.x;
                                _vec_load_41[0 + 1] = _v2_249.y;
                            }
                            p0_1[11] = _vec_load_41[0];
                            p1_1[11] = _vec_load_41[1];
                        }
                        lse_1[12] = -CUDART_INF_F;
                        if (base_1 + 12 < num_splits) {
                            long long part_row_28 = (long long)(base_1 + 12) * split_pitch + row_head_1[0];
                            lse_1[12] = partial_lse[part_row_28];
                            float _vec_load_42[2];
                            {
                                float2 _v2_250 = *reinterpret_cast<const float2*>(partial_out + (part_row_28 * 256 + dim_1_1[0]) + 0);
                                _vec_load_42[0] = _v2_250.x;
                                _vec_load_42[0 + 1] = _v2_250.y;
                            }
                            p0_1[12] = _vec_load_42[0];
                            p1_1[12] = _vec_load_42[1];
                        }
                        lse_1[13] = -CUDART_INF_F;
                        if (base_1 + 13 < num_splits) {
                            long long part_row_29 = (long long)(base_1 + 13) * split_pitch + row_head_1[0];
                            lse_1[13] = partial_lse[part_row_29];
                            float _vec_load_43[2];
                            {
                                float2 _v2_251 = *reinterpret_cast<const float2*>(partial_out + (part_row_29 * 256 + dim_1_1[0]) + 0);
                                _vec_load_43[0] = _v2_251.x;
                                _vec_load_43[0 + 1] = _v2_251.y;
                            }
                            p0_1[13] = _vec_load_43[0];
                            p1_1[13] = _vec_load_43[1];
                        }
                        lse_1[14] = -CUDART_INF_F;
                        if (base_1 + 14 < num_splits) {
                            long long part_row_30 = (long long)(base_1 + 14) * split_pitch + row_head_1[0];
                            lse_1[14] = partial_lse[part_row_30];
                            float _vec_load_44[2];
                            {
                                float2 _v2_252 = *reinterpret_cast<const float2*>(partial_out + (part_row_30 * 256 + dim_1_1[0]) + 0);
                                _vec_load_44[0] = _v2_252.x;
                                _vec_load_44[0 + 1] = _v2_252.y;
                            }
                            p0_1[14] = _vec_load_44[0];
                            p1_1[14] = _vec_load_44[1];
                        }
                        lse_1[15] = -CUDART_INF_F;
                        if (base_1 + 15 < num_splits) {
                            long long part_row_31 = (long long)(base_1 + 15) * split_pitch + row_head_1[0];
                            lse_1[15] = partial_lse[part_row_31];
                            float _vec_load_45[2];
                            {
                                float2 _v2_253 = *reinterpret_cast<const float2*>(partial_out + (part_row_31 * 256 + dim_1_1[0]) + 0);
                                _vec_load_45[0] = _v2_253.x;
                                _vec_load_45[0 + 1] = _v2_253.y;
                            }
                            p0_1[15] = _vec_load_45[0];
                            p1_1[15] = _vec_load_45[1];
                        }
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[0]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[1]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[2]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[3]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[4]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[5]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[6]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[7]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[8]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[9]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[10]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[11]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[12]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[13]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[14]);
                        lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[15]);
                        bool live_15 = base_1 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_15) {
                            float weight_16 = approx_exp2(lse_1[0] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_16;
                            acc0_1[0] = __fmaf_rn(weight_16, p0_1[0], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_16, p1_1[0], acc1_1[0]);
                        }
                        bool live_0_1 = base_1 + 1 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_0_1) {
                            float weight_17 = approx_exp2(lse_1[1] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_17;
                            acc0_1[0] = __fmaf_rn(weight_17, p0_1[1], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_17, p1_1[1], acc1_1[0]);
                        }
                        bool live_1_1 = base_1 + 2 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_1_1) {
                            float weight_18 = approx_exp2(lse_1[2] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_18;
                            acc0_1[0] = __fmaf_rn(weight_18, p0_1[2], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_18, p1_1[2], acc1_1[0]);
                        }
                        bool live_2_1 = base_1 + 3 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_2_1) {
                            float weight_19 = approx_exp2(lse_1[3] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_19;
                            acc0_1[0] = __fmaf_rn(weight_19, p0_1[3], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_19, p1_1[3], acc1_1[0]);
                        }
                        bool live_3_1 = base_1 + 4 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_3_1) {
                            float weight_20 = approx_exp2(lse_1[4] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_20;
                            acc0_1[0] = __fmaf_rn(weight_20, p0_1[4], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_20, p1_1[4], acc1_1[0]);
                        }
                        bool live_4_1 = base_1 + 5 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_4_1) {
                            float weight_21 = approx_exp2(lse_1[5] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_21;
                            acc0_1[0] = __fmaf_rn(weight_21, p0_1[5], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_21, p1_1[5], acc1_1[0]);
                        }
                        bool live_5_1 = base_1 + 6 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_5_1) {
                            float weight_22 = approx_exp2(lse_1[6] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_22;
                            acc0_1[0] = __fmaf_rn(weight_22, p0_1[6], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_22, p1_1[6], acc1_1[0]);
                        }
                        bool live_6_1 = base_1 + 7 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_6_1) {
                            float weight_23 = approx_exp2(lse_1[7] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_23;
                            acc0_1[0] = __fmaf_rn(weight_23, p0_1[7], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_23, p1_1[7], acc1_1[0]);
                        }
                        bool live_7_1 = base_1 + 8 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_7_1) {
                            float weight_24 = approx_exp2(lse_1[8] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_24;
                            acc0_1[0] = __fmaf_rn(weight_24, p0_1[8], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_24, p1_1[8], acc1_1[0]);
                        }
                        bool live_8_1 = base_1 + 9 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_8_1) {
                            float weight_25 = approx_exp2(lse_1[9] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_25;
                            acc0_1[0] = __fmaf_rn(weight_25, p0_1[9], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_25, p1_1[9], acc1_1[0]);
                        }
                        bool live_9_1 = base_1 + 10 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_9_1) {
                            float weight_26 = approx_exp2(lse_1[10] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_26;
                            acc0_1[0] = __fmaf_rn(weight_26, p0_1[10], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_26, p1_1[10], acc1_1[0]);
                        }
                        bool live_10_1 = base_1 + 11 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_10_1) {
                            float weight_27 = approx_exp2(lse_1[11] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_27;
                            acc0_1[0] = __fmaf_rn(weight_27, p0_1[11], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_27, p1_1[11], acc1_1[0]);
                        }
                        bool live_11_1 = base_1 + 12 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_11_1) {
                            float weight_28 = approx_exp2(lse_1[12] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_28;
                            acc0_1[0] = __fmaf_rn(weight_28, p0_1[12], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_28, p1_1[12], acc1_1[0]);
                        }
                        bool live_12_1 = base_1 + 13 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_12_1) {
                            float weight_29 = approx_exp2(lse_1[13] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_29;
                            acc0_1[0] = __fmaf_rn(weight_29, p0_1[13], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_29, p1_1[13], acc1_1[0]);
                        }
                        bool live_13_1 = base_1 + 14 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_13_1) {
                            float weight_30 = approx_exp2(lse_1[14] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_30;
                            acc0_1[0] = __fmaf_rn(weight_30, p0_1[14], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_30, p1_1[14], acc1_1[0]);
                        }
                        bool live_14_1 = base_1 + 15 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                        if (live_14_1) {
                            float weight_31 = approx_exp2(lse_1[15] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_31;
                            acc0_1[0] = __fmaf_rn(weight_31, p0_1[15], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_31, p1_1[15], acc1_1[0]);
                        }
                    }
                    if (valid_0_5[0] != 0) {
                        float inv_3 = 0.0f;
                        if (den_1[0] > 0.0f) {
                            float _rcp_254 = approx_rcp(den_1[0]);
                            inv_3 = _rcp_254;
                        }
                        float attn0_1 = (float)(__nv_bfloat16)(acc0_1[0] * inv_3);
                        float attn1_1 = (float)(__nv_bfloat16)(acc1_1[0] * inv_3);
                        float _exp2_256 = approx_exp2((-gate0_1[0]) * 1.442695f);
                        float _rcp_255 = approx_rcp(1.0f + _exp2_256);
                        float result0_1 = attn0_1 * _rcp_255;
                        float _exp2_258 = approx_exp2((-gate1_1[0]) * 1.442695f);
                        float _rcp_257 = approx_rcp(1.0f + _exp2_258);
                        float result1_1 = attn1_1 * _rcp_257;
                        __nv_bfloat162 _bit_cast_src_259 = __float22bfloat162_rn(make_float2(result0_1, result1_1));
                        word_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_259)[0];
                        long long out_offset_1 = row_i64 * out_stride_row + qo_head_3[0] * out_stride_head + dim_1_1[0];
                        *(reinterpret_cast<unsigned int*>(out + out_offset_1) + (0)) = word_1[0];
                    }
                }
            } else {
                if (num_splits > 4) {
                    float lse_2[16];
                    float p0_2[16];
                    float p1_2[16];
                    float acc0_2[2];
                    float acc1_2[2];
                    float den_2[2];
                    float lse_max_2[2];
                    int valid_0_6[2];
                    int tab_2[2];
                    long long dim_1_2[2];
                    long long qo_head_4[2];
                    long long row_head_2[2];
                    float gate0_2[2];
                    float gate1_2[2];
                    unsigned int word_2[1];
                    long long nqo_2 = (long long)num_qo_heads;
                    #pragma unroll 1
                    for (int p_2 = p_first; p_2 < end; p_2 += 256) {
                        int pk_2 = ((p_2) < (end - 1) ? (p_2) : (end - 1));
                        valid_0_6[0] = ((end > p_2) ? 1 : 0);
                        int head_0_2 = pk_2 >> 7;
                        dim_1_2[0] = (long long)((pk_2 & 127) * 2);
                        qo_head_4[0] = (long long)(first_head + head_0_2);
                        row_head_2[0] = row_i64 * nqo_2 + qo_head_4[0];
                        acc0_2[0] = 0.0f;
                        acc1_2[0] = 0.0f;
                        den_2[0] = 0.0f;
                        lse_max_2[0] = -CUDART_INF_F;
                        float _vec_load_46[2];
                        {
                            uint32_t _bf16x2_bits_260;
                            _bf16x2_bits_260 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2[0] * 256 + dim_1_2[0]) + 0);
                            (&_vec_load_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_260) << 16);
                            (&_vec_load_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_260) & 0xffff0000u);
                        }
                        gate0_2[0] = _vec_load_46[0];
                        gate1_2[0] = _vec_load_46[1];
                        int pk_1_1 = ((p_2 + 128) < (end - 1) ? (p_2 + 128) : (end - 1));
                        valid_0_6[1] = ((end > p_2 + 128) ? 1 : 0);
                        int head_2 = pk_1_1 >> 7;
                        dim_1_2[1] = (long long)((pk_1_1 & 127) * 2);
                        qo_head_4[1] = (long long)(first_head + head_2);
                        row_head_2[1] = row_i64 * nqo_2 + qo_head_4[1];
                        acc0_2[1] = 0.0f;
                        acc1_2[1] = 0.0f;
                        den_2[1] = 0.0f;
                        lse_max_2[1] = -CUDART_INF_F;
                        float _vec_load_47[2];
                        {
                            uint32_t _bf16x2_bits_261;
                            _bf16x2_bits_261 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2[1] * 256 + dim_1_2[1]) + 0);
                            (&_vec_load_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_261) << 16);
                            (&_vec_load_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_261) & 0xffff0000u);
                        }
                        gate0_2[1] = _vec_load_47[0];
                        gate1_2[1] = _vec_load_47[1];
                        #pragma unroll 1
                        for (int base_2 = 0; base_2 < num_splits; base_2 += 8) {
                            lse_2[0] = -CUDART_INF_F;
                            if (base_2 < num_splits) {
                                long long part_row_32 = (long long)base_2 * split_pitch + row_head_2[0];
                                lse_2[0] = partial_lse[part_row_32];
                                float _vec_load_48[2];
                                {
                                    float2 _v2_262 = *reinterpret_cast<const float2*>(partial_out + (part_row_32 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_48[0] = _v2_262.x;
                                    _vec_load_48[0 + 1] = _v2_262.y;
                                }
                                p0_2[0] = _vec_load_48[0];
                                p1_2[0] = _vec_load_48[1];
                            }
                            lse_2[1] = -CUDART_INF_F;
                            if (base_2 + 1 < num_splits) {
                                long long part_row_33 = (long long)(base_2 + 1) * split_pitch + row_head_2[0];
                                lse_2[1] = partial_lse[part_row_33];
                                float _vec_load_49[2];
                                {
                                    float2 _v2_263 = *reinterpret_cast<const float2*>(partial_out + (part_row_33 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_49[0] = _v2_263.x;
                                    _vec_load_49[0 + 1] = _v2_263.y;
                                }
                                p0_2[1] = _vec_load_49[0];
                                p1_2[1] = _vec_load_49[1];
                            }
                            lse_2[2] = -CUDART_INF_F;
                            if (base_2 + 2 < num_splits) {
                                long long part_row_34 = (long long)(base_2 + 2) * split_pitch + row_head_2[0];
                                lse_2[2] = partial_lse[part_row_34];
                                float _vec_load_50[2];
                                {
                                    float2 _v2_264 = *reinterpret_cast<const float2*>(partial_out + (part_row_34 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_50[0] = _v2_264.x;
                                    _vec_load_50[0 + 1] = _v2_264.y;
                                }
                                p0_2[2] = _vec_load_50[0];
                                p1_2[2] = _vec_load_50[1];
                            }
                            lse_2[3] = -CUDART_INF_F;
                            if (base_2 + 3 < num_splits) {
                                long long part_row_35 = (long long)(base_2 + 3) * split_pitch + row_head_2[0];
                                lse_2[3] = partial_lse[part_row_35];
                                float _vec_load_51[2];
                                {
                                    float2 _v2_265 = *reinterpret_cast<const float2*>(partial_out + (part_row_35 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_51[0] = _v2_265.x;
                                    _vec_load_51[0 + 1] = _v2_265.y;
                                }
                                p0_2[3] = _vec_load_51[0];
                                p1_2[3] = _vec_load_51[1];
                            }
                            lse_2[4] = -CUDART_INF_F;
                            if (base_2 + 4 < num_splits) {
                                long long part_row_36 = (long long)(base_2 + 4) * split_pitch + row_head_2[0];
                                lse_2[4] = partial_lse[part_row_36];
                                float _vec_load_52[2];
                                {
                                    float2 _v2_266 = *reinterpret_cast<const float2*>(partial_out + (part_row_36 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_52[0] = _v2_266.x;
                                    _vec_load_52[0 + 1] = _v2_266.y;
                                }
                                p0_2[4] = _vec_load_52[0];
                                p1_2[4] = _vec_load_52[1];
                            }
                            lse_2[5] = -CUDART_INF_F;
                            if (base_2 + 5 < num_splits) {
                                long long part_row_37 = (long long)(base_2 + 5) * split_pitch + row_head_2[0];
                                lse_2[5] = partial_lse[part_row_37];
                                float _vec_load_53[2];
                                {
                                    float2 _v2_267 = *reinterpret_cast<const float2*>(partial_out + (part_row_37 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_53[0] = _v2_267.x;
                                    _vec_load_53[0 + 1] = _v2_267.y;
                                }
                                p0_2[5] = _vec_load_53[0];
                                p1_2[5] = _vec_load_53[1];
                            }
                            lse_2[6] = -CUDART_INF_F;
                            if (base_2 + 6 < num_splits) {
                                long long part_row_38 = (long long)(base_2 + 6) * split_pitch + row_head_2[0];
                                lse_2[6] = partial_lse[part_row_38];
                                float _vec_load_54[2];
                                {
                                    float2 _v2_268 = *reinterpret_cast<const float2*>(partial_out + (part_row_38 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_54[0] = _v2_268.x;
                                    _vec_load_54[0 + 1] = _v2_268.y;
                                }
                                p0_2[6] = _vec_load_54[0];
                                p1_2[6] = _vec_load_54[1];
                            }
                            lse_2[7] = -CUDART_INF_F;
                            if (base_2 + 7 < num_splits) {
                                long long part_row_39 = (long long)(base_2 + 7) * split_pitch + row_head_2[0];
                                lse_2[7] = partial_lse[part_row_39];
                                float _vec_load_55[2];
                                {
                                    float2 _v2_269 = *reinterpret_cast<const float2*>(partial_out + (part_row_39 * 256 + dim_1_2[0]) + 0);
                                    _vec_load_55[0] = _v2_269.x;
                                    _vec_load_55[0 + 1] = _v2_269.y;
                                }
                                p0_2[7] = _vec_load_55[0];
                                p1_2[7] = _vec_load_55[1];
                            }
                            lse_2[8] = -CUDART_INF_F;
                            if (base_2 < num_splits) {
                                long long part_row_40 = (long long)base_2 * split_pitch + row_head_2[1];
                                lse_2[8] = partial_lse[part_row_40];
                                float _vec_load_56[2];
                                {
                                    float2 _v2_270 = *reinterpret_cast<const float2*>(partial_out + (part_row_40 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_56[0] = _v2_270.x;
                                    _vec_load_56[0 + 1] = _v2_270.y;
                                }
                                p0_2[8] = _vec_load_56[0];
                                p1_2[8] = _vec_load_56[1];
                            }
                            lse_2[9] = -CUDART_INF_F;
                            if (base_2 + 1 < num_splits) {
                                long long part_row_41 = (long long)(base_2 + 1) * split_pitch + row_head_2[1];
                                lse_2[9] = partial_lse[part_row_41];
                                float _vec_load_57[2];
                                {
                                    float2 _v2_271 = *reinterpret_cast<const float2*>(partial_out + (part_row_41 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_57[0] = _v2_271.x;
                                    _vec_load_57[0 + 1] = _v2_271.y;
                                }
                                p0_2[9] = _vec_load_57[0];
                                p1_2[9] = _vec_load_57[1];
                            }
                            lse_2[10] = -CUDART_INF_F;
                            if (base_2 + 2 < num_splits) {
                                long long part_row_42 = (long long)(base_2 + 2) * split_pitch + row_head_2[1];
                                lse_2[10] = partial_lse[part_row_42];
                                float _vec_load_58[2];
                                {
                                    float2 _v2_272 = *reinterpret_cast<const float2*>(partial_out + (part_row_42 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_58[0] = _v2_272.x;
                                    _vec_load_58[0 + 1] = _v2_272.y;
                                }
                                p0_2[10] = _vec_load_58[0];
                                p1_2[10] = _vec_load_58[1];
                            }
                            lse_2[11] = -CUDART_INF_F;
                            if (base_2 + 3 < num_splits) {
                                long long part_row_43 = (long long)(base_2 + 3) * split_pitch + row_head_2[1];
                                lse_2[11] = partial_lse[part_row_43];
                                float _vec_load_59[2];
                                {
                                    float2 _v2_273 = *reinterpret_cast<const float2*>(partial_out + (part_row_43 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_59[0] = _v2_273.x;
                                    _vec_load_59[0 + 1] = _v2_273.y;
                                }
                                p0_2[11] = _vec_load_59[0];
                                p1_2[11] = _vec_load_59[1];
                            }
                            lse_2[12] = -CUDART_INF_F;
                            if (base_2 + 4 < num_splits) {
                                long long part_row_44 = (long long)(base_2 + 4) * split_pitch + row_head_2[1];
                                lse_2[12] = partial_lse[part_row_44];
                                float _vec_load_60[2];
                                {
                                    float2 _v2_274 = *reinterpret_cast<const float2*>(partial_out + (part_row_44 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_60[0] = _v2_274.x;
                                    _vec_load_60[0 + 1] = _v2_274.y;
                                }
                                p0_2[12] = _vec_load_60[0];
                                p1_2[12] = _vec_load_60[1];
                            }
                            lse_2[13] = -CUDART_INF_F;
                            if (base_2 + 5 < num_splits) {
                                long long part_row_45 = (long long)(base_2 + 5) * split_pitch + row_head_2[1];
                                lse_2[13] = partial_lse[part_row_45];
                                float _vec_load_61[2];
                                {
                                    float2 _v2_275 = *reinterpret_cast<const float2*>(partial_out + (part_row_45 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_61[0] = _v2_275.x;
                                    _vec_load_61[0 + 1] = _v2_275.y;
                                }
                                p0_2[13] = _vec_load_61[0];
                                p1_2[13] = _vec_load_61[1];
                            }
                            lse_2[14] = -CUDART_INF_F;
                            if (base_2 + 6 < num_splits) {
                                long long part_row_46 = (long long)(base_2 + 6) * split_pitch + row_head_2[1];
                                lse_2[14] = partial_lse[part_row_46];
                                float _vec_load_62[2];
                                {
                                    float2 _v2_276 = *reinterpret_cast<const float2*>(partial_out + (part_row_46 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_62[0] = _v2_276.x;
                                    _vec_load_62[0 + 1] = _v2_276.y;
                                }
                                p0_2[14] = _vec_load_62[0];
                                p1_2[14] = _vec_load_62[1];
                            }
                            lse_2[15] = -CUDART_INF_F;
                            if (base_2 + 7 < num_splits) {
                                long long part_row_47 = (long long)(base_2 + 7) * split_pitch + row_head_2[1];
                                lse_2[15] = partial_lse[part_row_47];
                                float _vec_load_63[2];
                                {
                                    float2 _v2_277 = *reinterpret_cast<const float2*>(partial_out + (part_row_47 * 256 + dim_1_2[1]) + 0);
                                    _vec_load_63[0] = _v2_277.x;
                                    _vec_load_63[0 + 1] = _v2_277.y;
                                }
                                p0_2[15] = _vec_load_63[0];
                                p1_2[15] = _vec_load_63[1];
                            }
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[0]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[1]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[2]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[3]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[4]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[5]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[6]);
                            lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[7]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[8]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[9]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[10]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[11]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[12]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[13]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[14]);
                            lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[15]);
                            bool live_16 = base_2 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_16) {
                                float weight_32 = approx_exp2(lse_2[0] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_32;
                                acc0_2[0] = __fmaf_rn(weight_32, p0_2[0], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_32, p1_2[0], acc1_2[0]);
                            }
                            bool live_0_2 = base_2 + 1 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_0_2) {
                                float weight_33 = approx_exp2(lse_2[1] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_33;
                                acc0_2[0] = __fmaf_rn(weight_33, p0_2[1], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_33, p1_2[1], acc1_2[0]);
                            }
                            bool live_1_2 = base_2 + 2 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_1_2) {
                                float weight_34 = approx_exp2(lse_2[2] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_34;
                                acc0_2[0] = __fmaf_rn(weight_34, p0_2[2], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_34, p1_2[2], acc1_2[0]);
                            }
                            bool live_2_2 = base_2 + 3 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_2_2) {
                                float weight_35 = approx_exp2(lse_2[3] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_35;
                                acc0_2[0] = __fmaf_rn(weight_35, p0_2[3], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_35, p1_2[3], acc1_2[0]);
                            }
                            bool live_3_2 = base_2 + 4 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_3_2) {
                                float weight_36 = approx_exp2(lse_2[4] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_36;
                                acc0_2[0] = __fmaf_rn(weight_36, p0_2[4], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_36, p1_2[4], acc1_2[0]);
                            }
                            bool live_4_2 = base_2 + 5 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_4_2) {
                                float weight_37 = approx_exp2(lse_2[5] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_37;
                                acc0_2[0] = __fmaf_rn(weight_37, p0_2[5], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_37, p1_2[5], acc1_2[0]);
                            }
                            bool live_5_2 = base_2 + 6 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_5_2) {
                                float weight_38 = approx_exp2(lse_2[6] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_38;
                                acc0_2[0] = __fmaf_rn(weight_38, p0_2[6], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_38, p1_2[6], acc1_2[0]);
                            }
                            bool live_6_2 = base_2 + 7 < num_splits && lse_max_2[0] > -CUDART_INF_F;
                            if (live_6_2) {
                                float weight_39 = approx_exp2(lse_2[7] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_39;
                                acc0_2[0] = __fmaf_rn(weight_39, p0_2[7], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_39, p1_2[7], acc1_2[0]);
                            }
                            bool live_7_2 = base_2 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_7_2) {
                                float weight_40 = approx_exp2(lse_2[8] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_40;
                                acc0_2[1] = __fmaf_rn(weight_40, p0_2[8], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_40, p1_2[8], acc1_2[1]);
                            }
                            bool live_8_2 = base_2 + 1 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_8_2) {
                                float weight_41 = approx_exp2(lse_2[9] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_41;
                                acc0_2[1] = __fmaf_rn(weight_41, p0_2[9], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_41, p1_2[9], acc1_2[1]);
                            }
                            bool live_9_2 = base_2 + 2 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_9_2) {
                                float weight_42 = approx_exp2(lse_2[10] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_42;
                                acc0_2[1] = __fmaf_rn(weight_42, p0_2[10], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_42, p1_2[10], acc1_2[1]);
                            }
                            bool live_10_2 = base_2 + 3 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_10_2) {
                                float weight_43 = approx_exp2(lse_2[11] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_43;
                                acc0_2[1] = __fmaf_rn(weight_43, p0_2[11], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_43, p1_2[11], acc1_2[1]);
                            }
                            bool live_11_2 = base_2 + 4 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_11_2) {
                                float weight_44 = approx_exp2(lse_2[12] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_44;
                                acc0_2[1] = __fmaf_rn(weight_44, p0_2[12], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_44, p1_2[12], acc1_2[1]);
                            }
                            bool live_12_2 = base_2 + 5 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_12_2) {
                                float weight_45 = approx_exp2(lse_2[13] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_45;
                                acc0_2[1] = __fmaf_rn(weight_45, p0_2[13], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_45, p1_2[13], acc1_2[1]);
                            }
                            bool live_13_2 = base_2 + 6 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_13_2) {
                                float weight_46 = approx_exp2(lse_2[14] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_46;
                                acc0_2[1] = __fmaf_rn(weight_46, p0_2[14], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_46, p1_2[14], acc1_2[1]);
                            }
                            bool live_14_2 = base_2 + 7 < num_splits && lse_max_2[1] > -CUDART_INF_F;
                            if (live_14_2) {
                                float weight_47 = approx_exp2(lse_2[15] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_47;
                                acc0_2[1] = __fmaf_rn(weight_47, p0_2[15], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_47, p1_2[15], acc1_2[1]);
                            }
                        }
                        if (valid_0_6[0] != 0) {
                            float inv_4 = 0.0f;
                            if (den_2[0] > 0.0f) {
                                float _rcp_278 = approx_rcp(den_2[0]);
                                inv_4 = _rcp_278;
                            }
                            float attn0_2 = (float)(__nv_bfloat16)(acc0_2[0] * inv_4);
                            float attn1_2 = (float)(__nv_bfloat16)(acc1_2[0] * inv_4);
                            float _exp2_280 = approx_exp2((-gate0_2[0]) * 1.442695f);
                            float _rcp_279 = approx_rcp(1.0f + _exp2_280);
                            float result0_2 = attn0_2 * _rcp_279;
                            float _exp2_282 = approx_exp2((-gate1_2[0]) * 1.442695f);
                            float _rcp_281 = approx_rcp(1.0f + _exp2_282);
                            float result1_2 = attn1_2 * _rcp_281;
                            __nv_bfloat162 _bit_cast_src_283 = __float22bfloat162_rn(make_float2(result0_2, result1_2));
                            word_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_283)[0];
                            long long out_offset_2 = row_i64 * out_stride_row + qo_head_4[0] * out_stride_head + dim_1_2[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_2) + (0)) = word_2[0];
                        }
                        if (valid_0_6[1] != 0) {
                            float inv_5 = 0.0f;
                            if (den_2[1] > 0.0f) {
                                float _rcp_284 = approx_rcp(den_2[1]);
                                inv_5 = _rcp_284;
                            }
                            float attn0_3 = (float)(__nv_bfloat16)(acc0_2[1] * inv_5);
                            float attn1_3 = (float)(__nv_bfloat16)(acc1_2[1] * inv_5);
                            float _exp2_286 = approx_exp2((-gate0_2[1]) * 1.442695f);
                            float _rcp_285 = approx_rcp(1.0f + _exp2_286);
                            float result0_3 = attn0_3 * _rcp_285;
                            float _exp2_288 = approx_exp2((-gate1_2[1]) * 1.442695f);
                            float _rcp_287 = approx_rcp(1.0f + _exp2_288);
                            float result1_3 = attn1_3 * _rcp_287;
                            __nv_bfloat162 _bit_cast_src_289 = __float22bfloat162_rn(make_float2(result0_3, result1_3));
                            word_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_289)[0];
                            long long out_offset_3 = row_i64 * out_stride_row + qo_head_4[1] * out_stride_head + dim_1_2[1];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_3) + (0)) = word_2[0];
                        }
                    }
                } else {
                    float lse_3[16];
                    float p0_3[16];
                    float p1_3[16];
                    float acc0_3[4];
                    float acc1_3[4];
                    float den_3[4];
                    float lse_max_3[4];
                    int valid_0_7[4];
                    int tab_3[4];
                    long long dim_1_3[4];
                    long long qo_head_5[4];
                    long long row_head_3[4];
                    float gate0_3[4];
                    float gate1_3[4];
                    unsigned int word_3[1];
                    long long nqo_3 = (long long)num_qo_heads;
                    #pragma unroll 1
                    for (int p_3 = p_first; p_3 < end; p_3 += 512) {
                        int pk_3 = ((p_3) < (end - 1) ? (p_3) : (end - 1));
                        valid_0_7[0] = ((end > p_3) ? 1 : 0);
                        int head_0_3 = pk_3 >> 7;
                        dim_1_3[0] = (long long)((pk_3 & 127) * 2);
                        qo_head_5[0] = (long long)(first_head + head_0_3);
                        row_head_3[0] = row_i64 * nqo_3 + qo_head_5[0];
                        acc0_3[0] = 0.0f;
                        acc1_3[0] = 0.0f;
                        den_3[0] = 0.0f;
                        lse_max_3[0] = -CUDART_INF_F;
                        float _vec_load_64[2];
                        {
                            uint32_t _bf16x2_bits_290;
                            _bf16x2_bits_290 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[0] * 256 + dim_1_3[0]) + 0);
                            (&_vec_load_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_290) << 16);
                            (&_vec_load_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_290) & 0xffff0000u);
                        }
                        gate0_3[0] = _vec_load_64[0];
                        gate1_3[0] = _vec_load_64[1];
                        int pk_1_2 = ((p_3 + 128) < (end - 1) ? (p_3 + 128) : (end - 1));
                        valid_0_7[1] = ((end > p_3 + 128) ? 1 : 0);
                        int head_2_1 = pk_1_2 >> 7;
                        dim_1_3[1] = (long long)((pk_1_2 & 127) * 2);
                        qo_head_5[1] = (long long)(first_head + head_2_1);
                        row_head_3[1] = row_i64 * nqo_3 + qo_head_5[1];
                        acc0_3[1] = 0.0f;
                        acc1_3[1] = 0.0f;
                        den_3[1] = 0.0f;
                        lse_max_3[1] = -CUDART_INF_F;
                        float _vec_load_65[2];
                        {
                            uint32_t _bf16x2_bits_291;
                            _bf16x2_bits_291 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[1] * 256 + dim_1_3[1]) + 0);
                            (&_vec_load_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_291) << 16);
                            (&_vec_load_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_291) & 0xffff0000u);
                        }
                        gate0_3[1] = _vec_load_65[0];
                        gate1_3[1] = _vec_load_65[1];
                        int pk_3_1 = ((p_3 + 256) < (end - 1) ? (p_3 + 256) : (end - 1));
                        valid_0_7[2] = ((end > p_3 + 256) ? 1 : 0);
                        int head_4 = pk_3_1 >> 7;
                        dim_1_3[2] = (long long)((pk_3_1 & 127) * 2);
                        qo_head_5[2] = (long long)(first_head + head_4);
                        row_head_3[2] = row_i64 * nqo_3 + qo_head_5[2];
                        acc0_3[2] = 0.0f;
                        acc1_3[2] = 0.0f;
                        den_3[2] = 0.0f;
                        lse_max_3[2] = -CUDART_INF_F;
                        float _vec_load_66[2];
                        {
                            uint32_t _bf16x2_bits_292;
                            _bf16x2_bits_292 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[2] * 256 + dim_1_3[2]) + 0);
                            (&_vec_load_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_292) << 16);
                            (&_vec_load_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_292) & 0xffff0000u);
                        }
                        gate0_3[2] = _vec_load_66[0];
                        gate1_3[2] = _vec_load_66[1];
                        int pk_5 = ((p_3 + 384) < (end - 1) ? (p_3 + 384) : (end - 1));
                        valid_0_7[3] = ((end > p_3 + 384) ? 1 : 0);
                        int head_6 = pk_5 >> 7;
                        dim_1_3[3] = (long long)((pk_5 & 127) * 2);
                        qo_head_5[3] = (long long)(first_head + head_6);
                        row_head_3[3] = row_i64 * nqo_3 + qo_head_5[3];
                        acc0_3[3] = 0.0f;
                        acc1_3[3] = 0.0f;
                        den_3[3] = 0.0f;
                        lse_max_3[3] = -CUDART_INF_F;
                        float _vec_load_67[2];
                        {
                            uint32_t _bf16x2_bits_293;
                            _bf16x2_bits_293 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[3] * 256 + dim_1_3[3]) + 0);
                            (&_vec_load_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_293) << 16);
                            (&_vec_load_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_293) & 0xffff0000u);
                        }
                        gate0_3[3] = _vec_load_67[0];
                        gate1_3[3] = _vec_load_67[1];
                        #pragma unroll 1
                        for (int base_3 = 0; base_3 < num_splits; base_3 += 4) {
                            lse_3[0] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
                                long long part_row_48 = (long long)base_3 * split_pitch + row_head_3[0];
                                lse_3[0] = partial_lse[part_row_48];
                                float _vec_load_68[2];
                                {
                                    float2 _v2_294 = *reinterpret_cast<const float2*>(partial_out + (part_row_48 * 256 + dim_1_3[0]) + 0);
                                    _vec_load_68[0] = _v2_294.x;
                                    _vec_load_68[0 + 1] = _v2_294.y;
                                }
                                p0_3[0] = _vec_load_68[0];
                                p1_3[0] = _vec_load_68[1];
                            }
                            lse_3[1] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
                                long long part_row_49 = (long long)(base_3 + 1) * split_pitch + row_head_3[0];
                                lse_3[1] = partial_lse[part_row_49];
                                float _vec_load_69[2];
                                {
                                    float2 _v2_295 = *reinterpret_cast<const float2*>(partial_out + (part_row_49 * 256 + dim_1_3[0]) + 0);
                                    _vec_load_69[0] = _v2_295.x;
                                    _vec_load_69[0 + 1] = _v2_295.y;
                                }
                                p0_3[1] = _vec_load_69[0];
                                p1_3[1] = _vec_load_69[1];
                            }
                            lse_3[2] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
                                long long part_row_50 = (long long)(base_3 + 2) * split_pitch + row_head_3[0];
                                lse_3[2] = partial_lse[part_row_50];
                                float _vec_load_70[2];
                                {
                                    float2 _v2_296 = *reinterpret_cast<const float2*>(partial_out + (part_row_50 * 256 + dim_1_3[0]) + 0);
                                    _vec_load_70[0] = _v2_296.x;
                                    _vec_load_70[0 + 1] = _v2_296.y;
                                }
                                p0_3[2] = _vec_load_70[0];
                                p1_3[2] = _vec_load_70[1];
                            }
                            lse_3[3] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
                                long long part_row_51 = (long long)(base_3 + 3) * split_pitch + row_head_3[0];
                                lse_3[3] = partial_lse[part_row_51];
                                float _vec_load_71[2];
                                {
                                    float2 _v2_297 = *reinterpret_cast<const float2*>(partial_out + (part_row_51 * 256 + dim_1_3[0]) + 0);
                                    _vec_load_71[0] = _v2_297.x;
                                    _vec_load_71[0 + 1] = _v2_297.y;
                                }
                                p0_3[3] = _vec_load_71[0];
                                p1_3[3] = _vec_load_71[1];
                            }
                            lse_3[4] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
                                long long part_row_52 = (long long)base_3 * split_pitch + row_head_3[1];
                                lse_3[4] = partial_lse[part_row_52];
                                float _vec_load_72[2];
                                {
                                    float2 _v2_298 = *reinterpret_cast<const float2*>(partial_out + (part_row_52 * 256 + dim_1_3[1]) + 0);
                                    _vec_load_72[0] = _v2_298.x;
                                    _vec_load_72[0 + 1] = _v2_298.y;
                                }
                                p0_3[4] = _vec_load_72[0];
                                p1_3[4] = _vec_load_72[1];
                            }
                            lse_3[5] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
                                long long part_row_53 = (long long)(base_3 + 1) * split_pitch + row_head_3[1];
                                lse_3[5] = partial_lse[part_row_53];
                                float _vec_load_73[2];
                                {
                                    float2 _v2_299 = *reinterpret_cast<const float2*>(partial_out + (part_row_53 * 256 + dim_1_3[1]) + 0);
                                    _vec_load_73[0] = _v2_299.x;
                                    _vec_load_73[0 + 1] = _v2_299.y;
                                }
                                p0_3[5] = _vec_load_73[0];
                                p1_3[5] = _vec_load_73[1];
                            }
                            lse_3[6] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
                                long long part_row_54 = (long long)(base_3 + 2) * split_pitch + row_head_3[1];
                                lse_3[6] = partial_lse[part_row_54];
                                float _vec_load_74[2];
                                {
                                    float2 _v2_300 = *reinterpret_cast<const float2*>(partial_out + (part_row_54 * 256 + dim_1_3[1]) + 0);
                                    _vec_load_74[0] = _v2_300.x;
                                    _vec_load_74[0 + 1] = _v2_300.y;
                                }
                                p0_3[6] = _vec_load_74[0];
                                p1_3[6] = _vec_load_74[1];
                            }
                            lse_3[7] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
                                long long part_row_55 = (long long)(base_3 + 3) * split_pitch + row_head_3[1];
                                lse_3[7] = partial_lse[part_row_55];
                                float _vec_load_75[2];
                                {
                                    float2 _v2_301 = *reinterpret_cast<const float2*>(partial_out + (part_row_55 * 256 + dim_1_3[1]) + 0);
                                    _vec_load_75[0] = _v2_301.x;
                                    _vec_load_75[0 + 1] = _v2_301.y;
                                }
                                p0_3[7] = _vec_load_75[0];
                                p1_3[7] = _vec_load_75[1];
                            }
                            lse_3[8] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
                                long long part_row_56 = (long long)base_3 * split_pitch + row_head_3[2];
                                lse_3[8] = partial_lse[part_row_56];
                                float _vec_load_76[2];
                                {
                                    float2 _v2_302 = *reinterpret_cast<const float2*>(partial_out + (part_row_56 * 256 + dim_1_3[2]) + 0);
                                    _vec_load_76[0] = _v2_302.x;
                                    _vec_load_76[0 + 1] = _v2_302.y;
                                }
                                p0_3[8] = _vec_load_76[0];
                                p1_3[8] = _vec_load_76[1];
                            }
                            lse_3[9] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
                                long long part_row_57 = (long long)(base_3 + 1) * split_pitch + row_head_3[2];
                                lse_3[9] = partial_lse[part_row_57];
                                float _vec_load_77[2];
                                {
                                    float2 _v2_303 = *reinterpret_cast<const float2*>(partial_out + (part_row_57 * 256 + dim_1_3[2]) + 0);
                                    _vec_load_77[0] = _v2_303.x;
                                    _vec_load_77[0 + 1] = _v2_303.y;
                                }
                                p0_3[9] = _vec_load_77[0];
                                p1_3[9] = _vec_load_77[1];
                            }
                            lse_3[10] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
                                long long part_row_58 = (long long)(base_3 + 2) * split_pitch + row_head_3[2];
                                lse_3[10] = partial_lse[part_row_58];
                                float _vec_load_78[2];
                                {
                                    float2 _v2_304 = *reinterpret_cast<const float2*>(partial_out + (part_row_58 * 256 + dim_1_3[2]) + 0);
                                    _vec_load_78[0] = _v2_304.x;
                                    _vec_load_78[0 + 1] = _v2_304.y;
                                }
                                p0_3[10] = _vec_load_78[0];
                                p1_3[10] = _vec_load_78[1];
                            }
                            lse_3[11] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
                                long long part_row_59 = (long long)(base_3 + 3) * split_pitch + row_head_3[2];
                                lse_3[11] = partial_lse[part_row_59];
                                float _vec_load_79[2];
                                {
                                    float2 _v2_305 = *reinterpret_cast<const float2*>(partial_out + (part_row_59 * 256 + dim_1_3[2]) + 0);
                                    _vec_load_79[0] = _v2_305.x;
                                    _vec_load_79[0 + 1] = _v2_305.y;
                                }
                                p0_3[11] = _vec_load_79[0];
                                p1_3[11] = _vec_load_79[1];
                            }
                            lse_3[12] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
                                long long part_row_60 = (long long)base_3 * split_pitch + row_head_3[3];
                                lse_3[12] = partial_lse[part_row_60];
                                float _vec_load_80[2];
                                {
                                    float2 _v2_306 = *reinterpret_cast<const float2*>(partial_out + (part_row_60 * 256 + dim_1_3[3]) + 0);
                                    _vec_load_80[0] = _v2_306.x;
                                    _vec_load_80[0 + 1] = _v2_306.y;
                                }
                                p0_3[12] = _vec_load_80[0];
                                p1_3[12] = _vec_load_80[1];
                            }
                            lse_3[13] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
                                long long part_row_61 = (long long)(base_3 + 1) * split_pitch + row_head_3[3];
                                lse_3[13] = partial_lse[part_row_61];
                                float _vec_load_81[2];
                                {
                                    float2 _v2_307 = *reinterpret_cast<const float2*>(partial_out + (part_row_61 * 256 + dim_1_3[3]) + 0);
                                    _vec_load_81[0] = _v2_307.x;
                                    _vec_load_81[0 + 1] = _v2_307.y;
                                }
                                p0_3[13] = _vec_load_81[0];
                                p1_3[13] = _vec_load_81[1];
                            }
                            lse_3[14] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
                                long long part_row_62 = (long long)(base_3 + 2) * split_pitch + row_head_3[3];
                                lse_3[14] = partial_lse[part_row_62];
                                float _vec_load_82[2];
                                {
                                    float2 _v2_308 = *reinterpret_cast<const float2*>(partial_out + (part_row_62 * 256 + dim_1_3[3]) + 0);
                                    _vec_load_82[0] = _v2_308.x;
                                    _vec_load_82[0 + 1] = _v2_308.y;
                                }
                                p0_3[14] = _vec_load_82[0];
                                p1_3[14] = _vec_load_82[1];
                            }
                            lse_3[15] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
                                long long part_row_63 = (long long)(base_3 + 3) * split_pitch + row_head_3[3];
                                lse_3[15] = partial_lse[part_row_63];
                                float _vec_load_83[2];
                                {
                                    float2 _v2_309 = *reinterpret_cast<const float2*>(partial_out + (part_row_63 * 256 + dim_1_3[3]) + 0);
                                    _vec_load_83[0] = _v2_309.x;
                                    _vec_load_83[0 + 1] = _v2_309.y;
                                }
                                p0_3[15] = _vec_load_83[0];
                                p1_3[15] = _vec_load_83[1];
                            }
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[0]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[1]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[2]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[3]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[4]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[5]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[6]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[7]);
                            lse_max_3[2] = fmaxf(lse_max_3[2], lse_3[8]);
                            lse_max_3[2] = fmaxf(lse_max_3[2], lse_3[9]);
                            lse_max_3[2] = fmaxf(lse_max_3[2], lse_3[10]);
                            lse_max_3[2] = fmaxf(lse_max_3[2], lse_3[11]);
                            lse_max_3[3] = fmaxf(lse_max_3[3], lse_3[12]);
                            lse_max_3[3] = fmaxf(lse_max_3[3], lse_3[13]);
                            lse_max_3[3] = fmaxf(lse_max_3[3], lse_3[14]);
                            lse_max_3[3] = fmaxf(lse_max_3[3], lse_3[15]);
                            bool live_17 = base_3 < num_splits && lse_max_3[0] > -CUDART_INF_F;
                            if (live_17) {
                                float weight_48 = approx_exp2(lse_3[0] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_48;
                                acc0_3[0] = __fmaf_rn(weight_48, p0_3[0], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_48, p1_3[0], acc1_3[0]);
                            }
                            bool live_0_3 = base_3 + 1 < num_splits && lse_max_3[0] > -CUDART_INF_F;
                            if (live_0_3) {
                                float weight_49 = approx_exp2(lse_3[1] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_49;
                                acc0_3[0] = __fmaf_rn(weight_49, p0_3[1], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_49, p1_3[1], acc1_3[0]);
                            }
                            bool live_1_3 = base_3 + 2 < num_splits && lse_max_3[0] > -CUDART_INF_F;
                            if (live_1_3) {
                                float weight_50 = approx_exp2(lse_3[2] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_50;
                                acc0_3[0] = __fmaf_rn(weight_50, p0_3[2], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_50, p1_3[2], acc1_3[0]);
                            }
                            bool live_2_3 = base_3 + 3 < num_splits && lse_max_3[0] > -CUDART_INF_F;
                            if (live_2_3) {
                                float weight_51 = approx_exp2(lse_3[3] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_51;
                                acc0_3[0] = __fmaf_rn(weight_51, p0_3[3], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_51, p1_3[3], acc1_3[0]);
                            }
                            bool live_3_3 = base_3 < num_splits && lse_max_3[1] > -CUDART_INF_F;
                            if (live_3_3) {
                                float weight_52 = approx_exp2(lse_3[4] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_52;
                                acc0_3[1] = __fmaf_rn(weight_52, p0_3[4], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_52, p1_3[4], acc1_3[1]);
                            }
                            bool live_4_3 = base_3 + 1 < num_splits && lse_max_3[1] > -CUDART_INF_F;
                            if (live_4_3) {
                                float weight_53 = approx_exp2(lse_3[5] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_53;
                                acc0_3[1] = __fmaf_rn(weight_53, p0_3[5], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_53, p1_3[5], acc1_3[1]);
                            }
                            bool live_5_3 = base_3 + 2 < num_splits && lse_max_3[1] > -CUDART_INF_F;
                            if (live_5_3) {
                                float weight_54 = approx_exp2(lse_3[6] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_54;
                                acc0_3[1] = __fmaf_rn(weight_54, p0_3[6], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_54, p1_3[6], acc1_3[1]);
                            }
                            bool live_6_3 = base_3 + 3 < num_splits && lse_max_3[1] > -CUDART_INF_F;
                            if (live_6_3) {
                                float weight_55 = approx_exp2(lse_3[7] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_55;
                                acc0_3[1] = __fmaf_rn(weight_55, p0_3[7], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_55, p1_3[7], acc1_3[1]);
                            }
                            bool live_7_3 = base_3 < num_splits && lse_max_3[2] > -CUDART_INF_F;
                            if (live_7_3) {
                                float weight_56 = approx_exp2(lse_3[8] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_56;
                                acc0_3[2] = __fmaf_rn(weight_56, p0_3[8], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_56, p1_3[8], acc1_3[2]);
                            }
                            bool live_8_3 = base_3 + 1 < num_splits && lse_max_3[2] > -CUDART_INF_F;
                            if (live_8_3) {
                                float weight_57 = approx_exp2(lse_3[9] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_57;
                                acc0_3[2] = __fmaf_rn(weight_57, p0_3[9], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_57, p1_3[9], acc1_3[2]);
                            }
                            bool live_9_3 = base_3 + 2 < num_splits && lse_max_3[2] > -CUDART_INF_F;
                            if (live_9_3) {
                                float weight_58 = approx_exp2(lse_3[10] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_58;
                                acc0_3[2] = __fmaf_rn(weight_58, p0_3[10], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_58, p1_3[10], acc1_3[2]);
                            }
                            bool live_10_3 = base_3 + 3 < num_splits && lse_max_3[2] > -CUDART_INF_F;
                            if (live_10_3) {
                                float weight_59 = approx_exp2(lse_3[11] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_59;
                                acc0_3[2] = __fmaf_rn(weight_59, p0_3[11], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_59, p1_3[11], acc1_3[2]);
                            }
                            bool live_11_3 = base_3 < num_splits && lse_max_3[3] > -CUDART_INF_F;
                            if (live_11_3) {
                                float weight_60 = approx_exp2(lse_3[12] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_60;
                                acc0_3[3] = __fmaf_rn(weight_60, p0_3[12], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_60, p1_3[12], acc1_3[3]);
                            }
                            bool live_12_3 = base_3 + 1 < num_splits && lse_max_3[3] > -CUDART_INF_F;
                            if (live_12_3) {
                                float weight_61 = approx_exp2(lse_3[13] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_61;
                                acc0_3[3] = __fmaf_rn(weight_61, p0_3[13], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_61, p1_3[13], acc1_3[3]);
                            }
                            bool live_13_3 = base_3 + 2 < num_splits && lse_max_3[3] > -CUDART_INF_F;
                            if (live_13_3) {
                                float weight_62 = approx_exp2(lse_3[14] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_62;
                                acc0_3[3] = __fmaf_rn(weight_62, p0_3[14], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_62, p1_3[14], acc1_3[3]);
                            }
                            bool live_14_3 = base_3 + 3 < num_splits && lse_max_3[3] > -CUDART_INF_F;
                            if (live_14_3) {
                                float weight_63 = approx_exp2(lse_3[15] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_63;
                                acc0_3[3] = __fmaf_rn(weight_63, p0_3[15], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_63, p1_3[15], acc1_3[3]);
                            }
                        }
                        if (valid_0_7[0] != 0) {
                            float inv_6 = 0.0f;
                            if (den_3[0] > 0.0f) {
                                float _rcp_310 = approx_rcp(den_3[0]);
                                inv_6 = _rcp_310;
                            }
                            float attn0_4 = (float)(__nv_bfloat16)(acc0_3[0] * inv_6);
                            float attn1_4 = (float)(__nv_bfloat16)(acc1_3[0] * inv_6);
                            float _exp2_312 = approx_exp2((-gate0_3[0]) * 1.442695f);
                            float _rcp_311 = approx_rcp(1.0f + _exp2_312);
                            float result0_4 = attn0_4 * _rcp_311;
                            float _exp2_314 = approx_exp2((-gate1_3[0]) * 1.442695f);
                            float _rcp_313 = approx_rcp(1.0f + _exp2_314);
                            float result1_4 = attn1_4 * _rcp_313;
                            __nv_bfloat162 _bit_cast_src_315 = __float22bfloat162_rn(make_float2(result0_4, result1_4));
                            word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_315)[0];
                            long long out_offset_4 = row_i64 * out_stride_row + qo_head_5[0] * out_stride_head + dim_1_3[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_4) + (0)) = word_3[0];
                        }
                        if (valid_0_7[1] != 0) {
                            float inv_7 = 0.0f;
                            if (den_3[1] > 0.0f) {
                                float _rcp_316 = approx_rcp(den_3[1]);
                                inv_7 = _rcp_316;
                            }
                            float attn0_5 = (float)(__nv_bfloat16)(acc0_3[1] * inv_7);
                            float attn1_5 = (float)(__nv_bfloat16)(acc1_3[1] * inv_7);
                            float _exp2_318 = approx_exp2((-gate0_3[1]) * 1.442695f);
                            float _rcp_317 = approx_rcp(1.0f + _exp2_318);
                            float result0_5 = attn0_5 * _rcp_317;
                            float _exp2_320 = approx_exp2((-gate1_3[1]) * 1.442695f);
                            float _rcp_319 = approx_rcp(1.0f + _exp2_320);
                            float result1_5 = attn1_5 * _rcp_319;
                            __nv_bfloat162 _bit_cast_src_321 = __float22bfloat162_rn(make_float2(result0_5, result1_5));
                            word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_321)[0];
                            long long out_offset_5 = row_i64 * out_stride_row + qo_head_5[1] * out_stride_head + dim_1_3[1];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_5) + (0)) = word_3[0];
                        }
                        if (valid_0_7[2] != 0) {
                            float inv_8 = 0.0f;
                            if (den_3[2] > 0.0f) {
                                float _rcp_322 = approx_rcp(den_3[2]);
                                inv_8 = _rcp_322;
                            }
                            float attn0_6 = (float)(__nv_bfloat16)(acc0_3[2] * inv_8);
                            float attn1_6 = (float)(__nv_bfloat16)(acc1_3[2] * inv_8);
                            float _exp2_324 = approx_exp2((-gate0_3[2]) * 1.442695f);
                            float _rcp_323 = approx_rcp(1.0f + _exp2_324);
                            float result0_6 = attn0_6 * _rcp_323;
                            float _exp2_326 = approx_exp2((-gate1_3[2]) * 1.442695f);
                            float _rcp_325 = approx_rcp(1.0f + _exp2_326);
                            float result1_6 = attn1_6 * _rcp_325;
                            __nv_bfloat162 _bit_cast_src_327 = __float22bfloat162_rn(make_float2(result0_6, result1_6));
                            word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_327)[0];
                            long long out_offset_6 = row_i64 * out_stride_row + qo_head_5[2] * out_stride_head + dim_1_3[2];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_6) + (0)) = word_3[0];
                        }
                        if (valid_0_7[3] != 0) {
                            float inv_9 = 0.0f;
                            if (den_3[3] > 0.0f) {
                                float _rcp_328 = approx_rcp(den_3[3]);
                                inv_9 = _rcp_328;
                            }
                            float attn0_7 = (float)(__nv_bfloat16)(acc0_3[3] * inv_9);
                            float attn1_7 = (float)(__nv_bfloat16)(acc1_3[3] * inv_9);
                            float _exp2_330 = approx_exp2((-gate0_3[3]) * 1.442695f);
                            float _rcp_329 = approx_rcp(1.0f + _exp2_330);
                            float result0_7 = attn0_7 * _rcp_329;
                            float _exp2_332 = approx_exp2((-gate1_3[3]) * 1.442695f);
                            float _rcp_331 = approx_rcp(1.0f + _exp2_332);
                            float result1_7 = attn1_7 * _rcp_331;
                            __nv_bfloat162 _bit_cast_src_333 = __float22bfloat162_rn(make_float2(result0_7, result1_7));
                            word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_333)[0];
                            long long out_offset_7 = row_i64 * out_stride_row + qo_head_5[3] * out_stride_head + dim_1_3[3];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_7) + (0)) = word_3[0];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
