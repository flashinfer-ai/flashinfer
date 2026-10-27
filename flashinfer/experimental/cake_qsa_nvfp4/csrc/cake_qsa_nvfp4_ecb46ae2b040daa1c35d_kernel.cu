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
kernel_cake_qsa_nvfp4_ecb46ae2b040daa1c35d(__nv_bfloat16* __restrict__ q, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int* __restrict__ indices, int* __restrict__ block_table, int* __restrict__ token_to_req, __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, float* partial_out, float* partial_lse, unsigned int* __restrict__ merge_counters, long long q_stride_row, long long q_stride_head, long long data_stride_block, long long data_stride_token, long long data_stride_head, long long sf_stride_block, long long sf_stride_token, long long sf_stride_head, long long indices_stride_row, long long table_stride_req, long long out_stride_row, long long out_stride_head, float score_scale, float output_scale, int num_rows, int num_requests, int num_blocks, CakeFastDivmod page_div, int table_width, int topk, int group_size, int num_qo_heads, int num_tiles, int num_splits)
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
    if (tid_0 == 0) {
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(lut_s_addr), "r"(2071689984), "r"(3229614080u) : "memory");
    }
    unsigned int lut_regs[2];
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
            int side = warp_1 >> 1;
            int half = warp_1 & 1;
            unsigned int bits_1 = slot_valid[half];
            if (tid_0 < 2) {
                valid_s[tid_0] = slot_valid[tid_0];
            }
            unsigned long long data_base = ((side != 0) ? v_base : k_base);
            unsigned long long sf_base = ((side != 0) ? v_sf_base : k_sf_base);
            int col = lane_1 & 7;
            int sub = lane_1 >> 3;
            int token0 = 32 * half + sub;
            unsigned int entry_addr = addr_ring_addr + (unsigned int)(token0 * 16);
            unsigned int bits_sub = bits_1 >> (unsigned int)sub;
            unsigned long long col_off = (unsigned long long)(col * 16);
#if __CUDA_ARCH__ == 1200
            unsigned int dst0 = ((kv_addr + (unsigned int)((32 * warp_1 + sub) * 144 + col * 16)) < (kv_addr + 36848) ? (kv_addr + (unsigned int)((32 * warp_1 + sub) * 144 + col * 16)) : (kv_addr + 36848));
#else
            unsigned int dst0 = kv_addr + (unsigned int)((32 * warp_1 + sub) * 144 + col * 16);
#endif
            unsigned int entry_0[4];
            __int128_t _smem_raw_11;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(entry_addr));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
            unsigned long long data_off_1 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src = data_base + data_off_1 + col_off;
            unsigned int src_size = (bits_sub & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0), "l"(src), "r"(src_size), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0), "l"(src), "r"(src_size));
#endif
            __int128_t _smem_raw_12;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(entry_addr + 64));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
            unsigned long long data_off_1_1 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_2 = data_base + data_off_1_1 + col_off;
            unsigned int src_size_3 = (bits_sub >> 4 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 576), "l"(src_2), "r"(src_size_3), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 576), "l"(src_2), "r"(src_size_3));
#endif
            __int128_t _smem_raw_13;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(entry_addr + 128));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
            unsigned long long data_off_4 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_5 = data_base + data_off_4 + col_off;
            unsigned int src_size_6 = (bits_sub >> 8 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 1152), "l"(src_5), "r"(src_size_6), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 1152), "l"(src_5), "r"(src_size_6));
#endif
            __int128_t _smem_raw_14;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(entry_addr + 192));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
            unsigned long long data_off_7 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_8 = data_base + data_off_7 + col_off;
            unsigned int src_size_9 = (bits_sub >> 12 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 1728), "l"(src_8), "r"(src_size_9), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 1728), "l"(src_8), "r"(src_size_9));
#endif
            __int128_t _smem_raw_15;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_15) : "r"(entry_addr + 256));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[3];
            unsigned long long data_off_10 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_11 = data_base + data_off_10 + col_off;
            unsigned int src_size_12 = (bits_sub >> 16 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 2304), "l"(src_11), "r"(src_size_12), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 2304), "l"(src_11), "r"(src_size_12));
#endif
            __int128_t _smem_raw_16;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_16) : "r"(entry_addr + 320));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[3];
            unsigned long long data_off_13 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_14 = data_base + data_off_13 + col_off;
            unsigned int src_size_15 = (bits_sub >> 20 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 2880), "l"(src_14), "r"(src_size_15), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 2880), "l"(src_14), "r"(src_size_15));
#endif
            __int128_t _smem_raw_17;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(entry_addr + 384));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
            unsigned long long data_off_16 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_17 = data_base + data_off_16 + col_off;
            unsigned int src_size_18 = (bits_sub >> 24 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 3456), "l"(src_17), "r"(src_size_18), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 3456), "l"(src_17), "r"(src_size_18));
#endif
            __int128_t _smem_raw_18;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(entry_addr + 448));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
            unsigned long long data_off_19 = (unsigned long long)entry_0[0] | (unsigned long long)entry_0[1] << 32;
            unsigned long long src_20 = data_base + data_off_19 + col_off;
            unsigned int src_size_21 = (bits_sub >> 28 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0 + 4032), "l"(src_20), "r"(src_size_21), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0 + 4032), "l"(src_20), "r"(src_size_21));
#endif
            __int128_t _smem_raw_19;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(addr_ring_addr + (unsigned int)((32 * half + lane_1) * 16)));
            entry_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
            entry_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
            entry_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
            entry_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
            unsigned long long sf_off_1 = (unsigned long long)entry_0[2] | (unsigned long long)entry_0[3] << 32;
            unsigned long long sf_src = sf_base + sf_off_1;
            unsigned int sf_size = (bits_1 >> (unsigned int)lane_1 & 1) << 4;
            unsigned int sf_dst = kv_addr + (unsigned int)((32 * warp_1 + lane_1) * 144 + 128);
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(sf_dst), "l"(sf_src), "r"(sf_size), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(sf_dst), "l"(sf_src), "r"(sf_size));
#endif
        }
        asm volatile("cp.async.commit_group;");
        if (warp_1 < 2) {
            bool valid_0_1 = phys[1] >= 0 && phys[1] < num_blocks;
            int _max_20 = ((phys[1]) > (0) ? (phys[1]) : (0));
            unsigned long long block_1 = (unsigned long long)_max_20;
            unsigned long long local_u64_1 = (unsigned long long)local[1];
            unsigned long long data_off_2 = block_1 * data_stride_block_u64 + local_u64_1 * data_stride_token_u64;
            unsigned long long sf_off_2 = block_1 * sf_stride_block_u64 + local_u64_1 * sf_stride_token_u64;
            unsigned int w0_1_1 = (unsigned int)data_off_2;
            unsigned int w1_2_1 = (unsigned int)(data_off_2 >> 32);
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
            int _max_21 = ((phys[0]) > (0) ? (phys[0]) : (0));
            unsigned long long block_2 = (unsigned long long)_max_21;
            unsigned long long local_u64_2 = (unsigned long long)local[0];
            bool is_v = tid_0 >= 64;
            unsigned long long data_base_1 = ((is_v) ? v_base : k_base);
            unsigned long long sf_base_1 = ((is_v) ? v_sf_base : k_sf_base);
            unsigned long long data_src = data_base_1 + block_2 * data_stride_block_u64 + local_u64_2 * data_stride_token_u64;
            unsigned long long sf_src_1 = sf_base_1 + block_2 * sf_stride_block_u64 + local_u64_2 * sf_stride_token_u64;
            unsigned int dst = kv_addr + (unsigned int)(tid_0 * 144);
            int nbytes = ((valid_0_2) ? 16 : 0);
            unsigned int src_size_1 = (unsigned int)nbytes;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst), "l"(data_src), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 16), "l"(data_src + 16), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 32), "l"(data_src + 32), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 48), "l"(data_src + 48), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 64), "l"(data_src + 64), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 80), "l"(data_src + 80), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 96), "l"(data_src + 96), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 112), "l"(data_src + 112), "r"(src_size_1), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst + 128), "l"(sf_src_1), "r"(src_size_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst), "l"(data_src), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 16), "l"(data_src + 16), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 32), "l"(data_src + 32), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 48), "l"(data_src + 48), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 64), "l"(data_src + 64), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 80), "l"(data_src + 80), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 96), "l"(data_src + 96), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 112), "l"(data_src + 112), "r"(src_size_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst + 128), "l"(sf_src_1), "r"(src_size_1));
#endif
            unsigned int bits_3 = __ballot_sync(0xFFFFFFFF, valid_0_2);
            if (warp_1 < 2 && lane_1 == 0) {
                valid_s[warp_1] = bits_3;
            }
        }
        asm volatile("cp.async.commit_group;");
    }
    int stage = 0;
    int parity = 0;
    #pragma unroll 1
    for (int tile = split; tile < tile_end; tile += num_splits) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lut_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_regs[(0) + 1]))
            : "r"(lut_s_addr));
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
        if (tile_end > tile + num_splits) {
            int side_1 = warp_1 >> 1;
            int half_1 = warp_1 & 1;
            unsigned int bits_4 = slot_valid[issue_slot * 2 + half_1];
            if (tid_0 < 2) {
                valid_s[issue_stage * 2 + tid_0] = slot_valid[issue_slot * 2 + tid_0];
            }
            unsigned long long data_base_2 = ((side_1 != 0) ? v_base : k_base);
            unsigned long long sf_base_2 = ((side_1 != 0) ? v_sf_base : k_sf_base);
            int col_1 = lane_1 & 7;
            int sub_1 = lane_1 >> 3;
            int token0_1 = 32 * half_1 + sub_1;
            unsigned int entry_addr_1 = addr_ring_addr + (unsigned int)(issue_slot * 1024 + token0_1 * 16);
            unsigned int bits_sub_1 = bits_4 >> (unsigned int)sub_1;
            unsigned long long col_off_1 = (unsigned long long)(col_1 * 16);
#if __CUDA_ARCH__ == 1200
            unsigned int dst0_1 = ((kv_addr + (unsigned int)(issue_stage * 18432 + (32 * warp_1 + sub_1) * 144 + col_1 * 16)) < (kv_addr + 36848) ? (kv_addr + (unsigned int)(issue_stage * 18432 + (32 * warp_1 + sub_1) * 144 + col_1 * 16)) : (kv_addr + 36848));
#else
            unsigned int dst0_1 = kv_addr + (unsigned int)(issue_stage * 18432 + (32 * warp_1 + sub_1) * 144 + col_1 * 16);
#endif
            unsigned int entry_0_1[4];
            __int128_t _smem_raw_22;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_22) : "r"(entry_addr_1));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[3];
            unsigned long long data_off_3 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_1 = data_base_2 + data_off_3 + col_off_1;
            unsigned int src_size_2 = (bits_sub_1 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1), "l"(src_1), "r"(src_size_2), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1), "l"(src_1), "r"(src_size_2));
#endif
            __int128_t _smem_raw_23;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_23) : "r"(entry_addr_1 + 64));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[3];
            unsigned long long data_off_1_2 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_2_1 = data_base_2 + data_off_1_2 + col_off_1;
            unsigned int src_size_3_1 = (bits_sub_1 >> 4 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 576), "l"(src_2_1), "r"(src_size_3_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 576), "l"(src_2_1), "r"(src_size_3_1));
#endif
            __int128_t _smem_raw_24;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_24) : "r"(entry_addr_1 + 128));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[3];
            unsigned long long data_off_4_1 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_5_1 = data_base_2 + data_off_4_1 + col_off_1;
            unsigned int src_size_6_1 = (bits_sub_1 >> 8 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 1152), "l"(src_5_1), "r"(src_size_6_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 1152), "l"(src_5_1), "r"(src_size_6_1));
#endif
            __int128_t _smem_raw_25;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_25) : "r"(entry_addr_1 + 192));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[3];
            unsigned long long data_off_7_1 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_8_1 = data_base_2 + data_off_7_1 + col_off_1;
            unsigned int src_size_9_1 = (bits_sub_1 >> 12 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 1728), "l"(src_8_1), "r"(src_size_9_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 1728), "l"(src_8_1), "r"(src_size_9_1));
#endif
            __int128_t _smem_raw_26;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_26) : "r"(entry_addr_1 + 256));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[3];
            unsigned long long data_off_10_1 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_11_1 = data_base_2 + data_off_10_1 + col_off_1;
            unsigned int src_size_12_1 = (bits_sub_1 >> 16 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 2304), "l"(src_11_1), "r"(src_size_12_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 2304), "l"(src_11_1), "r"(src_size_12_1));
#endif
            __int128_t _smem_raw_27;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_27) : "r"(entry_addr_1 + 320));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[3];
            unsigned long long data_off_13_1 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_14_1 = data_base_2 + data_off_13_1 + col_off_1;
            unsigned int src_size_15_1 = (bits_sub_1 >> 20 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 2880), "l"(src_14_1), "r"(src_size_15_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 2880), "l"(src_14_1), "r"(src_size_15_1));
#endif
            __int128_t _smem_raw_28;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_28) : "r"(entry_addr_1 + 384));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[3];
            unsigned long long data_off_16_1 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_17_1 = data_base_2 + data_off_16_1 + col_off_1;
            unsigned int src_size_18_1 = (bits_sub_1 >> 24 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 3456), "l"(src_17_1), "r"(src_size_18_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 3456), "l"(src_17_1), "r"(src_size_18_1));
#endif
            __int128_t _smem_raw_29;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_29) : "r"(entry_addr_1 + 448));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[3];
            unsigned long long data_off_19_1 = (unsigned long long)entry_0_1[0] | (unsigned long long)entry_0_1[1] << 32;
            unsigned long long src_20_1 = data_base_2 + data_off_19_1 + col_off_1;
            unsigned int src_size_21_1 = (bits_sub_1 >> 28 & 1) << 4;
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst0_1 + 4032), "l"(src_20_1), "r"(src_size_21_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_1 + 4032), "l"(src_20_1), "r"(src_size_21_1));
#endif
            __int128_t _smem_raw_30;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_30) : "r"(addr_ring_addr + (unsigned int)(issue_slot * 1024 + (32 * half_1 + lane_1) * 16)));
            entry_0_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[0];
            entry_0_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[1];
            entry_0_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[2];
            entry_0_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[3];
            unsigned long long sf_off_3 = (unsigned long long)entry_0_1[2] | (unsigned long long)entry_0_1[3] << 32;
            unsigned long long sf_src_2 = sf_base_2 + sf_off_3;
            unsigned int sf_size_1 = (bits_4 >> (unsigned int)lane_1 & 1) << 4;
            unsigned int sf_dst_1 = kv_addr + (unsigned int)(issue_stage * 18432 + (32 * warp_1 + lane_1) * 144 + 128);
#if __CUDA_ARCH__ == 1200
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(sf_dst_1), "l"(sf_src_2), "r"(sf_size_1), "l"(0x12F0000000000000ULL));
#else
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(sf_dst_1), "l"(sf_src_2), "r"(sf_size_1));
#endif
        }
        asm volatile("cp.async.commit_group;");
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
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_2) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[0] & 2004318071));
        uint32_t _prmt_3;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_3) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[0] & 2004318071) >> 16));
        uint32_t _prmt_4;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_4) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[0] & 2004318071));
        uint32_t _prmt_5;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_5) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[0] & 2004318071) >> 16));
        uint32_t _prmt_6;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_6) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_7;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_7) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_31) : "r"(_prmt_4), "r"(_prmt_2 | _prmt_6 & 2155905152u));
        d0[0] = _prmt_31;
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_32) : "r"(_prmt_4), "r"(_prmt_2 | _prmt_6 & 2155905152u));
        d0[1] = _prmt_32;
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_33) : "r"(_prmt_5), "r"(_prmt_3 | _prmt_7 & 2155905152u));
        d0[2] = _prmt_33;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_34) : "r"(_prmt_5), "r"(_prmt_3 | _prmt_7 & 2155905152u));
        d0[3] = _prmt_34;
        uint32_t _prmt_8;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_8) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[0] & 2004318071));
        uint32_t _prmt_9;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_9) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[0] & 2004318071) >> 16));
        uint32_t _prmt_10;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_10) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[0] & 2004318071));
        uint32_t _prmt_11;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_11) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[0] & 2004318071) >> 16));
        uint32_t _prmt_12;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_12) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_13;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_13) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_35) : "r"(_prmt_10), "r"(_prmt_8 | _prmt_12 & 2155905152u));
        d1[0] = _prmt_35;
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_36) : "r"(_prmt_10), "r"(_prmt_8 | _prmt_12 & 2155905152u));
        d1[1] = _prmt_36;
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_37) : "r"(_prmt_11), "r"(_prmt_9 | _prmt_13 & 2155905152u));
        d1[2] = _prmt_37;
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_38) : "r"(_prmt_11), "r"(_prmt_9 | _prmt_13 & 2155905152u));
        d1[3] = _prmt_38;
        uint32_t _prmt_14;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_14) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_15;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_15) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_0;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_0) : "r"(d0[0]), "r"(_prmt_14), "r"(2147516416u));
        d0[0] = _bf16x2_fma_0;
        uint32_t _bf16x2_fma_1;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_1) : "r"(d1[0]), "r"(_prmt_15), "r"(2147516416u));
        d1[0] = _bf16x2_fma_1;
        uint32_t _bf16x2_fma_2;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_2) : "r"(d0[1]), "r"(_prmt_14), "r"(2147516416u));
        d0[1] = _bf16x2_fma_2;
        uint32_t _bf16x2_fma_3;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_3) : "r"(d1[1]), "r"(_prmt_15), "r"(2147516416u));
        d1[1] = _bf16x2_fma_3;
        uint32_t _bf16x2_fma_4;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_4) : "r"(d0[2]), "r"(_prmt_14), "r"(2147516416u));
        d0[2] = _bf16x2_fma_4;
        uint32_t _bf16x2_fma_5;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_5) : "r"(d1[2]), "r"(_prmt_15), "r"(2147516416u));
        d1[2] = _bf16x2_fma_5;
        uint32_t _bf16x2_fma_6;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_6) : "r"(d0[3]), "r"(_prmt_14), "r"(2147516416u));
        d0[3] = _bf16x2_fma_6;
        uint32_t _bf16x2_fma_7;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_7) : "r"(d1[3]), "r"(_prmt_15), "r"(2147516416u));
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
        uint32_t _prmt_16;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_16) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[1] & 2004318071));
        uint32_t _prmt_17;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_17) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[1] & 2004318071) >> 16));
        uint32_t _prmt_18;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_18) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[1] & 2004318071));
        uint32_t _prmt_19;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_19) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[1] & 2004318071) >> 16));
        uint32_t _prmt_20;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_20) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_21;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_21) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_39) : "r"(_prmt_18), "r"(_prmt_16 | _prmt_20 & 2155905152u));
        d0[0] = _prmt_39;
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_40) : "r"(_prmt_18), "r"(_prmt_16 | _prmt_20 & 2155905152u));
        d0[1] = _prmt_40;
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_41) : "r"(_prmt_19), "r"(_prmt_17 | _prmt_21 & 2155905152u));
        d0[2] = _prmt_41;
        uint32_t _prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_42) : "r"(_prmt_19), "r"(_prmt_17 | _prmt_21 & 2155905152u));
        d0[3] = _prmt_42;
        uint32_t _prmt_22;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_22) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[1] & 2004318071));
        uint32_t _prmt_23;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_23) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[1] & 2004318071) >> 16));
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_24) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[1] & 2004318071));
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_25) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[1] & 2004318071) >> 16));
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_26) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_27) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_43;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_43) : "r"(_prmt_24), "r"(_prmt_22 | _prmt_26 & 2155905152u));
        d1[0] = _prmt_43;
        uint32_t _prmt_44;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_44) : "r"(_prmt_24), "r"(_prmt_22 | _prmt_26 & 2155905152u));
        d1[1] = _prmt_44;
        uint32_t _prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_45) : "r"(_prmt_25), "r"(_prmt_23 | _prmt_27 & 2155905152u));
        d1[2] = _prmt_45;
        uint32_t _prmt_46;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_46) : "r"(_prmt_25), "r"(_prmt_23 | _prmt_27 & 2155905152u));
        d1[3] = _prmt_46;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_28) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_29) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_8;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_8) : "r"(d0[0]), "r"(_prmt_28), "r"(2147516416u));
        d0[0] = _bf16x2_fma_8;
        uint32_t _bf16x2_fma_9;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_9) : "r"(d1[0]), "r"(_prmt_29), "r"(2147516416u));
        d1[0] = _bf16x2_fma_9;
        uint32_t _bf16x2_fma_10;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_10) : "r"(d0[1]), "r"(_prmt_28), "r"(2147516416u));
        d0[1] = _bf16x2_fma_10;
        uint32_t _bf16x2_fma_11;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_11) : "r"(d1[1]), "r"(_prmt_29), "r"(2147516416u));
        d1[1] = _bf16x2_fma_11;
        uint32_t _bf16x2_fma_12;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_12) : "r"(d0[2]), "r"(_prmt_28), "r"(2147516416u));
        d0[2] = _bf16x2_fma_12;
        uint32_t _bf16x2_fma_13;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_13) : "r"(d1[2]), "r"(_prmt_29), "r"(2147516416u));
        d1[2] = _bf16x2_fma_13;
        uint32_t _bf16x2_fma_14;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_14) : "r"(d0[3]), "r"(_prmt_28), "r"(2147516416u));
        d0[3] = _bf16x2_fma_14;
        uint32_t _bf16x2_fma_15;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_15) : "r"(d1[3]), "r"(_prmt_29), "r"(2147516416u));
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
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_30) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[2] & 2004318071));
        uint32_t _prmt_31_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_31_1) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[2] & 2004318071) >> 16));
        uint32_t _prmt_32_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_32_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[2] & 2004318071));
        uint32_t _prmt_33_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_33_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[2] & 2004318071) >> 16));
        uint32_t _prmt_34_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_34_1) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_35_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_35_1) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_47;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_47) : "r"(_prmt_32_1), "r"(_prmt_30 | _prmt_34_1 & 2155905152u));
        d0[0] = _prmt_47;
        uint32_t _prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_48) : "r"(_prmt_32_1), "r"(_prmt_30 | _prmt_34_1 & 2155905152u));
        d0[1] = _prmt_48;
        uint32_t _prmt_49;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_49) : "r"(_prmt_33_1), "r"(_prmt_31_1 | _prmt_35_1 & 2155905152u));
        d0[2] = _prmt_49;
        uint32_t _prmt_50;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_50) : "r"(_prmt_33_1), "r"(_prmt_31_1 | _prmt_35_1 & 2155905152u));
        d0[3] = _prmt_50;
        uint32_t _prmt_36_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_36_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[2] & 2004318071));
        uint32_t _prmt_37_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37_1) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[2] & 2004318071) >> 16));
        uint32_t _prmt_38_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[2] & 2004318071));
        uint32_t _prmt_39_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_39_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[2] & 2004318071) >> 16));
        uint32_t _prmt_40_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_40_1) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_41_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_41_1) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_51;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_51) : "r"(_prmt_38_1), "r"(_prmt_36_1 | _prmt_40_1 & 2155905152u));
        d1[0] = _prmt_51;
        uint32_t _prmt_52;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_52) : "r"(_prmt_38_1), "r"(_prmt_36_1 | _prmt_40_1 & 2155905152u));
        d1[1] = _prmt_52;
        uint32_t _prmt_53;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_53) : "r"(_prmt_39_1), "r"(_prmt_37_1 | _prmt_41_1 & 2155905152u));
        d1[2] = _prmt_53;
        uint32_t _prmt_54;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_54) : "r"(_prmt_39_1), "r"(_prmt_37_1 | _prmt_41_1 & 2155905152u));
        d1[3] = _prmt_54;
        uint32_t _prmt_42_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_42_1) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_43_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_43_1) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_16;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_16) : "r"(d0[0]), "r"(_prmt_42_1), "r"(2147516416u));
        d0[0] = _bf16x2_fma_16;
        uint32_t _bf16x2_fma_17;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_17) : "r"(d1[0]), "r"(_prmt_43_1), "r"(2147516416u));
        d1[0] = _bf16x2_fma_17;
        uint32_t _bf16x2_fma_18;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_18) : "r"(d0[1]), "r"(_prmt_42_1), "r"(2147516416u));
        d0[1] = _bf16x2_fma_18;
        uint32_t _bf16x2_fma_19;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_19) : "r"(d1[1]), "r"(_prmt_43_1), "r"(2147516416u));
        d1[1] = _bf16x2_fma_19;
        uint32_t _bf16x2_fma_20;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_20) : "r"(d0[2]), "r"(_prmt_42_1), "r"(2147516416u));
        d0[2] = _bf16x2_fma_20;
        uint32_t _bf16x2_fma_21;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_21) : "r"(d1[2]), "r"(_prmt_43_1), "r"(2147516416u));
        d1[2] = _bf16x2_fma_21;
        uint32_t _bf16x2_fma_22;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_22) : "r"(d0[3]), "r"(_prmt_42_1), "r"(2147516416u));
        d0[3] = _bf16x2_fma_22;
        uint32_t _bf16x2_fma_23;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_23) : "r"(d1[3]), "r"(_prmt_43_1), "r"(2147516416u));
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
        uint32_t _prmt_44_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_44_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[3] & 2004318071));
        uint32_t _prmt_45_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_45_1) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[3] & 2004318071) >> 16));
        uint32_t _prmt_46_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_46_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[3] & 2004318071));
        uint32_t _prmt_47_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_47_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[3] & 2004318071) >> 16));
        uint32_t _prmt_48_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_48_1) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_49_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_49_1) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_55;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_55) : "r"(_prmt_46_1), "r"(_prmt_44_1 | _prmt_48_1 & 2155905152u));
        d0[0] = _prmt_55;
        uint32_t _prmt_56;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_56) : "r"(_prmt_46_1), "r"(_prmt_44_1 | _prmt_48_1 & 2155905152u));
        d0[1] = _prmt_56;
        uint32_t _prmt_57;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_57) : "r"(_prmt_47_1), "r"(_prmt_45_1 | _prmt_49_1 & 2155905152u));
        d0[2] = _prmt_57;
        uint32_t _prmt_58;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_58) : "r"(_prmt_47_1), "r"(_prmt_45_1 | _prmt_49_1 & 2155905152u));
        d0[3] = _prmt_58;
        uint32_t _prmt_50_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_50_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[3] & 2004318071));
        uint32_t _prmt_51_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_51_1) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[3] & 2004318071) >> 16));
        uint32_t _prmt_52_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_52_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[3] & 2004318071));
        uint32_t _prmt_53_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_53_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[3] & 2004318071) >> 16));
        uint32_t _prmt_54_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_54_1) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_55_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_55_1) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_59;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_59) : "r"(_prmt_52_1), "r"(_prmt_50_1 | _prmt_54_1 & 2155905152u));
        d1[0] = _prmt_59;
        uint32_t _prmt_60;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_60) : "r"(_prmt_52_1), "r"(_prmt_50_1 | _prmt_54_1 & 2155905152u));
        d1[1] = _prmt_60;
        uint32_t _prmt_61;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_61) : "r"(_prmt_53_1), "r"(_prmt_51_1 | _prmt_55_1 & 2155905152u));
        d1[2] = _prmt_61;
        uint32_t _prmt_62;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_62) : "r"(_prmt_53_1), "r"(_prmt_51_1 | _prmt_55_1 & 2155905152u));
        d1[3] = _prmt_62;
        uint32_t _prmt_56_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_56_1) : "r"(sf0[0]), "r"(0));
        uint32_t _prmt_57_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_57_1) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_24;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_24) : "r"(d0[0]), "r"(_prmt_56_1), "r"(2147516416u));
        d0[0] = _bf16x2_fma_24;
        uint32_t _bf16x2_fma_25;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_25) : "r"(d1[0]), "r"(_prmt_57_1), "r"(2147516416u));
        d1[0] = _bf16x2_fma_25;
        uint32_t _bf16x2_fma_26;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_26) : "r"(d0[1]), "r"(_prmt_56_1), "r"(2147516416u));
        d0[1] = _bf16x2_fma_26;
        uint32_t _bf16x2_fma_27;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_27) : "r"(d1[1]), "r"(_prmt_57_1), "r"(2147516416u));
        d1[1] = _bf16x2_fma_27;
        uint32_t _bf16x2_fma_28;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_28) : "r"(d0[2]), "r"(_prmt_56_1), "r"(2147516416u));
        d0[2] = _bf16x2_fma_28;
        uint32_t _bf16x2_fma_29;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_29) : "r"(d1[2]), "r"(_prmt_57_1), "r"(2147516416u));
        d1[2] = _bf16x2_fma_29;
        uint32_t _bf16x2_fma_30;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_30) : "r"(d0[3]), "r"(_prmt_56_1), "r"(2147516416u));
        d0[3] = _bf16x2_fma_30;
        uint32_t _bf16x2_fma_31;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_31) : "r"(d1[3]), "r"(_prmt_57_1), "r"(2147516416u));
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
        uint32_t _prmt_58_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_58_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[0] & 2004318071));
        uint32_t _prmt_59_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_59_1) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[0] & 2004318071) >> 16));
        uint32_t _prmt_60_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_60_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[0] & 2004318071));
        uint32_t _prmt_61_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_61_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[0] & 2004318071) >> 16));
        uint32_t _prmt_62_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_62_1) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_63;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_63) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_63_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_63_1) : "r"(_prmt_60_1), "r"(_prmt_58_1 | _prmt_62_1 & 2155905152u));
        d0[0] = _prmt_63_1;
        uint32_t _prmt_64;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_64) : "r"(_prmt_60_1), "r"(_prmt_58_1 | _prmt_62_1 & 2155905152u));
        d0[1] = _prmt_64;
        uint32_t _prmt_65;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_65) : "r"(_prmt_61_1), "r"(_prmt_59_1 | _prmt_63 & 2155905152u));
        d0[2] = _prmt_65;
        uint32_t _prmt_66;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_66) : "r"(_prmt_61_1), "r"(_prmt_59_1 | _prmt_63 & 2155905152u));
        d0[3] = _prmt_66;
        uint32_t _prmt_64_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_64_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[0] & 2004318071));
        uint32_t _prmt_65_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_65_1) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[0] & 2004318071) >> 16));
        uint32_t _prmt_66_1;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_66_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[0] & 2004318071));
        uint32_t _prmt_67;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_67) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[0] & 2004318071) >> 16));
        uint32_t _prmt_68;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_68) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_69;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_69) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_67_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_67_1) : "r"(_prmt_66_1), "r"(_prmt_64_1 | _prmt_68 & 2155905152u));
        d1[0] = _prmt_67_1;
        uint32_t _prmt_68_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_68_1) : "r"(_prmt_66_1), "r"(_prmt_64_1 | _prmt_68 & 2155905152u));
        d1[1] = _prmt_68_1;
        uint32_t _prmt_69_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_69_1) : "r"(_prmt_67), "r"(_prmt_65_1 | _prmt_69 & 2155905152u));
        d1[2] = _prmt_69_1;
        uint32_t _prmt_70;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_70) : "r"(_prmt_67), "r"(_prmt_65_1 | _prmt_69 & 2155905152u));
        d1[3] = _prmt_70;
        uint32_t _prmt_70_1;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_70_1) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_71;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_71) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_32;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_32) : "r"(d0[0]), "r"(_prmt_70_1), "r"(2147516416u));
        d0[0] = _bf16x2_fma_32;
        uint32_t _bf16x2_fma_33;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_33) : "r"(d1[0]), "r"(_prmt_71), "r"(2147516416u));
        d1[0] = _bf16x2_fma_33;
        uint32_t _bf16x2_fma_34;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_34) : "r"(d0[1]), "r"(_prmt_70_1), "r"(2147516416u));
        d0[1] = _bf16x2_fma_34;
        uint32_t _bf16x2_fma_35;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_35) : "r"(d1[1]), "r"(_prmt_71), "r"(2147516416u));
        d1[1] = _bf16x2_fma_35;
        uint32_t _bf16x2_fma_36;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_36) : "r"(d0[2]), "r"(_prmt_70_1), "r"(2147516416u));
        d0[2] = _bf16x2_fma_36;
        uint32_t _bf16x2_fma_37;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_37) : "r"(d1[2]), "r"(_prmt_71), "r"(2147516416u));
        d1[2] = _bf16x2_fma_37;
        uint32_t _bf16x2_fma_38;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_38) : "r"(d0[3]), "r"(_prmt_70_1), "r"(2147516416u));
        d0[3] = _bf16x2_fma_38;
        uint32_t _bf16x2_fma_39;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_39) : "r"(d1[3]), "r"(_prmt_71), "r"(2147516416u));
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
        uint32_t _prmt_72;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_72) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[1] & 2004318071));
        uint32_t _prmt_73;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_73) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[1] & 2004318071) >> 16));
        uint32_t _prmt_74;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[1] & 2004318071));
        uint32_t _prmt_75;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[1] & 2004318071) >> 16));
        uint32_t _prmt_76;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_76) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_77;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_77) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_71_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_71_1) : "r"(_prmt_74), "r"(_prmt_72 | _prmt_76 & 2155905152u));
        d0[0] = _prmt_71_1;
        uint32_t _prmt_72_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_72_1) : "r"(_prmt_74), "r"(_prmt_72 | _prmt_76 & 2155905152u));
        d0[1] = _prmt_72_1;
        uint32_t _prmt_73_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_73_1) : "r"(_prmt_75), "r"(_prmt_73 | _prmt_77 & 2155905152u));
        d0[2] = _prmt_73_1;
        uint32_t _prmt_74_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_74_1) : "r"(_prmt_75), "r"(_prmt_73 | _prmt_77 & 2155905152u));
        d0[3] = _prmt_74_1;
        uint32_t _prmt_78;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_78) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[1] & 2004318071));
        uint32_t _prmt_79;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_79) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[1] & 2004318071) >> 16));
        uint32_t _prmt_80;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_80) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[1] & 2004318071));
        uint32_t _prmt_81;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_81) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[1] & 2004318071) >> 16));
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_82) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_83;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_83) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_75_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_75_1) : "r"(_prmt_80), "r"(_prmt_78 | _prmt_82 & 2155905152u));
        d1[0] = _prmt_75_1;
        uint32_t _prmt_76_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_76_1) : "r"(_prmt_80), "r"(_prmt_78 | _prmt_82 & 2155905152u));
        d1[1] = _prmt_76_1;
        uint32_t _prmt_77_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_77_1) : "r"(_prmt_81), "r"(_prmt_79 | _prmt_83 & 2155905152u));
        d1[2] = _prmt_77_1;
        uint32_t _prmt_78_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_78_1) : "r"(_prmt_81), "r"(_prmt_79 | _prmt_83 & 2155905152u));
        d1[3] = _prmt_78_1;
        uint32_t _prmt_84;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_84) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_85;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_85) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_40;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_40) : "r"(d0[0]), "r"(_prmt_84), "r"(2147516416u));
        d0[0] = _bf16x2_fma_40;
        uint32_t _bf16x2_fma_41;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_41) : "r"(d1[0]), "r"(_prmt_85), "r"(2147516416u));
        d1[0] = _bf16x2_fma_41;
        uint32_t _bf16x2_fma_42;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_42) : "r"(d0[1]), "r"(_prmt_84), "r"(2147516416u));
        d0[1] = _bf16x2_fma_42;
        uint32_t _bf16x2_fma_43;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_43) : "r"(d1[1]), "r"(_prmt_85), "r"(2147516416u));
        d1[1] = _bf16x2_fma_43;
        uint32_t _bf16x2_fma_44;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_44) : "r"(d0[2]), "r"(_prmt_84), "r"(2147516416u));
        d0[2] = _bf16x2_fma_44;
        uint32_t _bf16x2_fma_45;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_45) : "r"(d1[2]), "r"(_prmt_85), "r"(2147516416u));
        d1[2] = _bf16x2_fma_45;
        uint32_t _bf16x2_fma_46;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_46) : "r"(d0[3]), "r"(_prmt_84), "r"(2147516416u));
        d0[3] = _bf16x2_fma_46;
        uint32_t _bf16x2_fma_47;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_47) : "r"(d1[3]), "r"(_prmt_85), "r"(2147516416u));
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
        uint32_t _prmt_86;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_86) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[2] & 2004318071));
        uint32_t _prmt_87;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_87) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[2] & 2004318071) >> 16));
        uint32_t _prmt_88;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_88) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[2] & 2004318071));
        uint32_t _prmt_89;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_89) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[2] & 2004318071) >> 16));
        uint32_t _prmt_90;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_90) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_91;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_91) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_79_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_79_1) : "r"(_prmt_88), "r"(_prmt_86 | _prmt_90 & 2155905152u));
        d0[0] = _prmt_79_1;
        uint32_t _prmt_80_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_80_1) : "r"(_prmt_88), "r"(_prmt_86 | _prmt_90 & 2155905152u));
        d0[1] = _prmt_80_1;
        uint32_t _prmt_81_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_81_1) : "r"(_prmt_89), "r"(_prmt_87 | _prmt_91 & 2155905152u));
        d0[2] = _prmt_81_1;
        uint32_t _prmt_82_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_82_1) : "r"(_prmt_89), "r"(_prmt_87 | _prmt_91 & 2155905152u));
        d0[3] = _prmt_82_1;
        uint32_t _prmt_92;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_92) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[2] & 2004318071));
        uint32_t _prmt_93;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_93) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[2] & 2004318071) >> 16));
        uint32_t _prmt_94;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_94) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[2] & 2004318071));
        uint32_t _prmt_95;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_95) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[2] & 2004318071) >> 16));
        uint32_t _prmt_96;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_96) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_97;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_97) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_83_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_83_1) : "r"(_prmt_94), "r"(_prmt_92 | _prmt_96 & 2155905152u));
        d1[0] = _prmt_83_1;
        uint32_t _prmt_84_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_84_1) : "r"(_prmt_94), "r"(_prmt_92 | _prmt_96 & 2155905152u));
        d1[1] = _prmt_84_1;
        uint32_t _prmt_85_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_85_1) : "r"(_prmt_95), "r"(_prmt_93 | _prmt_97 & 2155905152u));
        d1[2] = _prmt_85_1;
        uint32_t _prmt_86_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_86_1) : "r"(_prmt_95), "r"(_prmt_93 | _prmt_97 & 2155905152u));
        d1[3] = _prmt_86_1;
        uint32_t _prmt_98;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_98) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_99;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_99) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_48;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_48) : "r"(d0[0]), "r"(_prmt_98), "r"(2147516416u));
        d0[0] = _bf16x2_fma_48;
        uint32_t _bf16x2_fma_49;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_49) : "r"(d1[0]), "r"(_prmt_99), "r"(2147516416u));
        d1[0] = _bf16x2_fma_49;
        uint32_t _bf16x2_fma_50;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_50) : "r"(d0[1]), "r"(_prmt_98), "r"(2147516416u));
        d0[1] = _bf16x2_fma_50;
        uint32_t _bf16x2_fma_51;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_51) : "r"(d1[1]), "r"(_prmt_99), "r"(2147516416u));
        d1[1] = _bf16x2_fma_51;
        uint32_t _bf16x2_fma_52;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_52) : "r"(d0[2]), "r"(_prmt_98), "r"(2147516416u));
        d0[2] = _bf16x2_fma_52;
        uint32_t _bf16x2_fma_53;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_53) : "r"(d1[2]), "r"(_prmt_99), "r"(2147516416u));
        d1[2] = _bf16x2_fma_53;
        uint32_t _bf16x2_fma_54;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_54) : "r"(d0[3]), "r"(_prmt_98), "r"(2147516416u));
        d0[3] = _bf16x2_fma_54;
        uint32_t _bf16x2_fma_55;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_55) : "r"(d1[3]), "r"(_prmt_99), "r"(2147516416u));
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
        uint32_t _prmt_100;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_100) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[3] & 2004318071));
        uint32_t _prmt_101;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_101) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw0[3] & 2004318071) >> 16));
        uint32_t _prmt_102;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_102) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[3] & 2004318071));
        uint32_t _prmt_103;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_103) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw0[3] & 2004318071) >> 16));
        uint32_t _prmt_104;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_104) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_105;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_105) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_87_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_87_1) : "r"(_prmt_102), "r"(_prmt_100 | _prmt_104 & 2155905152u));
        d0[0] = _prmt_87_1;
        uint32_t _prmt_88_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_88_1) : "r"(_prmt_102), "r"(_prmt_100 | _prmt_104 & 2155905152u));
        d0[1] = _prmt_88_1;
        uint32_t _prmt_89_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_89_1) : "r"(_prmt_103), "r"(_prmt_101 | _prmt_105 & 2155905152u));
        d0[2] = _prmt_89_1;
        uint32_t _prmt_90_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_90_1) : "r"(_prmt_103), "r"(_prmt_101 | _prmt_105 & 2155905152u));
        d0[3] = _prmt_90_1;
        uint32_t _prmt_106;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_106) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[3] & 2004318071));
        uint32_t _prmt_107;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_107) : "r"(lut_regs[0]), "r"(2088533116), "r"((raw1[3] & 2004318071) >> 16));
        uint32_t _prmt_108;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_108) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[3] & 2004318071));
        uint32_t _prmt_109;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_109) : "r"(lut_regs[1]), "r"(3229630464u), "r"((raw1[3] & 2004318071) >> 16));
        uint32_t _prmt_110;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_110) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_111;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_111) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_91_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_91_1) : "r"(_prmt_108), "r"(_prmt_106 | _prmt_110 & 2155905152u));
        d1[0] = _prmt_91_1;
        uint32_t _prmt_92_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_92_1) : "r"(_prmt_108), "r"(_prmt_106 | _prmt_110 & 2155905152u));
        d1[1] = _prmt_92_1;
        uint32_t _prmt_93_1;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_93_1) : "r"(_prmt_109), "r"(_prmt_107 | _prmt_111 & 2155905152u));
        d1[2] = _prmt_93_1;
        uint32_t _prmt_94_1;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_94_1) : "r"(_prmt_109), "r"(_prmt_107 | _prmt_111 & 2155905152u));
        d1[3] = _prmt_94_1;
        uint32_t _prmt_112;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_112) : "r"(sf0[1]), "r"(0));
        uint32_t _prmt_113;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_113) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_56;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_56) : "r"(d0[0]), "r"(_prmt_112), "r"(2147516416u));
        d0[0] = _bf16x2_fma_56;
        uint32_t _bf16x2_fma_57;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_57) : "r"(d1[0]), "r"(_prmt_113), "r"(2147516416u));
        d1[0] = _bf16x2_fma_57;
        uint32_t _bf16x2_fma_58;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_58) : "r"(d0[1]), "r"(_prmt_112), "r"(2147516416u));
        d0[1] = _bf16x2_fma_58;
        uint32_t _bf16x2_fma_59;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_59) : "r"(d1[1]), "r"(_prmt_113), "r"(2147516416u));
        d1[1] = _bf16x2_fma_59;
        uint32_t _bf16x2_fma_60;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_60) : "r"(d0[2]), "r"(_prmt_112), "r"(2147516416u));
        d0[2] = _bf16x2_fma_60;
        uint32_t _bf16x2_fma_61;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_61) : "r"(d1[2]), "r"(_prmt_113), "r"(2147516416u));
        d1[2] = _bf16x2_fma_61;
        uint32_t _bf16x2_fma_62;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_62) : "r"(d0[3]), "r"(_prmt_112), "r"(2147516416u));
        d0[3] = _bf16x2_fma_62;
        uint32_t _bf16x2_fma_63;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_63) : "r"(d1[3]), "r"(_prmt_113), "r"(2147516416u));
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
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, row_max[0], 1);
        row_max[0] = fmaxf(row_max[0], _shfl_xor_sync_95);
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, row_max[0], 2);
        row_max[0] = fmaxf(row_max[0], _shfl_xor_sync_96);
        if (t == 0) {
            row_max_s[warp_1 * 16 + g] = row_max[0];
        }
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, row_max[1], 1);
        row_max[1] = fmaxf(row_max[1], _shfl_xor_sync_97);
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, row_max[1], 2);
        row_max[1] = fmaxf(row_max[1], _shfl_xor_sync_98);
        if (t == 0) {
            row_max_s[warp_1 * 16 + g + 8] = row_max[1];
        }
        __syncthreads();
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
        float _exp2_99 = approx_exp2(s[0] - m_new[0]);
        s[0] = ((ok[0] != 0) ? _exp2_99 : 0.0f);
        row_sum[0] = row_sum[0] + s[0];
        float _exp2_100 = approx_exp2(s[1] - m_new[0]);
        s[1] = ((ok[1] != 0) ? _exp2_100 : 0.0f);
        row_sum[0] = row_sum[0] + s[1];
        float _exp2_101 = approx_exp2(s[2] - m_new[1]);
        s[2] = ((ok[2] != 0) ? _exp2_101 : 0.0f);
        row_sum[1] = row_sum[1] + s[2];
        float _exp2_102 = approx_exp2(s[3] - m_new[1]);
        s[3] = ((ok[3] != 0) ? _exp2_102 : 0.0f);
        row_sum[1] = row_sum[1] + s[3];
        float _exp2_103 = approx_exp2(s[4] - m_new[0]);
        s[4] = ((ok[4] != 0) ? _exp2_103 : 0.0f);
        row_sum[0] = row_sum[0] + s[4];
        float _exp2_104 = approx_exp2(s[5] - m_new[0]);
        s[5] = ((ok[5] != 0) ? _exp2_104 : 0.0f);
        row_sum[0] = row_sum[0] + s[5];
        float _exp2_105 = approx_exp2(s[6] - m_new[1]);
        s[6] = ((ok[6] != 0) ? _exp2_105 : 0.0f);
        row_sum[1] = row_sum[1] + s[6];
        float _exp2_106 = approx_exp2(s[7] - m_new[1]);
        s[7] = ((ok[7] != 0) ? _exp2_106 : 0.0f);
        row_sum[1] = row_sum[1] + s[7];
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 1);
        row_sum[0] = row_sum[0] + _shfl_xor_sync_107;
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 2);
        row_sum[0] = row_sum[0] + _shfl_xor_sync_108;
        if (t == 0) {
            row_sum_s[warp_1 * 16 + g] = row_sum[0];
        }
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 1);
        row_sum[1] = row_sum[1] + _shfl_xor_sync_109;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 2);
        row_sum[1] = row_sum[1] + _shfl_xor_sync_110;
        if (t == 0) {
            row_sum_s[warp_1 * 16 + g + 8] = row_sum[1];
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(p_frag_addr + (unsigned int)(16 * (32 * warp_1 + lane_1))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[0], s[1])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[2], s[3])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[4], s[5])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(s[6], s[7])))) : "memory");
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
        int v_row = v_stage + 2 * t * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row + v_col)));
        unsigned int scale_byte = (unsigned int)kv_u8[v_row + v_scale_col];
        int v_row_3 = v_stage + (2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_3 + v_col)));
        unsigned int scale_byte_4 = (unsigned int)kv_u8[v_row_3 + v_scale_col];
        int v_row_5 = v_stage + (2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_5 + v_col)));
        unsigned int scale_byte_6 = (unsigned int)kv_u8[v_row_5 + v_scale_col];
        int v_row_7 = v_stage + (2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_7 + v_col)));
        unsigned int scale_byte_8 = (unsigned int)kv_u8[v_row_7 + v_scale_col];
        uint32_t _prmt_114;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_114) : "r"(scale_byte), "r"(scale_byte_4));
        sfv[0] = _prmt_114 & 2147516416u | (_prmt_114 & 2130738944) >> 4;
        uint32_t _prmt_115;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_115) : "r"(scale_byte_6), "r"(scale_byte_8));
        sfv[1] = _prmt_115 & 2147516416u | (_prmt_115 & 2130738944) >> 4;
        uint32_t w_lo;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi) : "r"(vw[0]), "r"(vw[1]));
        uint32_t _prmt_116;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_116) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo & 2004318071));
        uint32_t _prmt_117;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_117) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo & 2004318071) >> 16));
        uint32_t _prmt_118;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_118) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo & 2004318071));
        uint32_t _prmt_119;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_119) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo & 2004318071) >> 16));
        uint32_t _prmt_120;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_120) : "r"(w_lo << 4), "r"(w_lo));
        uint32_t _prmt_121;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_121) : "r"(w_lo << 4), "r"(w_lo));
        uint32_t _prmt_111_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_111_1) : "r"(_prmt_118), "r"(_prmt_116 | _prmt_120 & 2155905152u));
        dp_lo[0] = _prmt_111_1;
        uint32_t _prmt_112_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_112_1) : "r"(_prmt_118), "r"(_prmt_116 | _prmt_120 & 2155905152u));
        dp_lo[1] = _prmt_112_1;
        uint32_t _prmt_113_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_113_1) : "r"(_prmt_119), "r"(_prmt_117 | _prmt_121 & 2155905152u));
        dp_lo[2] = _prmt_113_1;
        uint32_t _prmt_114_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_114_1) : "r"(_prmt_119), "r"(_prmt_117 | _prmt_121 & 2155905152u));
        dp_lo[3] = _prmt_114_1;
        uint32_t _prmt_122;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_122) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi & 2004318071));
        uint32_t _prmt_123;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_123) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi & 2004318071) >> 16));
        uint32_t _prmt_124;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_124) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi & 2004318071));
        uint32_t _prmt_125;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_125) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi & 2004318071) >> 16));
        uint32_t _prmt_126;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_126) : "r"(w_hi << 4), "r"(w_hi));
        uint32_t _prmt_127;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_127) : "r"(w_hi << 4), "r"(w_hi));
        uint32_t _prmt_115_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_115_1) : "r"(_prmt_124), "r"(_prmt_122 | _prmt_126 & 2155905152u));
        dp_hi[0] = _prmt_115_1;
        uint32_t _prmt_116_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_116_1) : "r"(_prmt_124), "r"(_prmt_122 | _prmt_126 & 2155905152u));
        dp_hi[1] = _prmt_116_1;
        uint32_t _prmt_117_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_117_1) : "r"(_prmt_125), "r"(_prmt_123 | _prmt_127 & 2155905152u));
        dp_hi[2] = _prmt_117_1;
        uint32_t _prmt_118_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_118_1) : "r"(_prmt_125), "r"(_prmt_123 | _prmt_127 & 2155905152u));
        dp_hi[3] = _prmt_118_1;
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
        uint32_t _prmt_128;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_128) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_9 & 2004318071));
        uint32_t _prmt_129;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_129) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_9 & 2004318071) >> 16));
        uint32_t _prmt_130;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_130) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_9 & 2004318071));
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_131) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_9 & 2004318071) >> 16));
        uint32_t _prmt_132;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_132) : "r"(w_lo_9 << 4), "r"(w_lo_9));
        uint32_t _prmt_133;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_133) : "r"(w_lo_9 << 4), "r"(w_lo_9));
        uint32_t _prmt_119_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_119_1) : "r"(_prmt_130), "r"(_prmt_128 | _prmt_132 & 2155905152u));
        dp_lo[4] = _prmt_119_1;
        uint32_t _prmt_120_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_120_1) : "r"(_prmt_130), "r"(_prmt_128 | _prmt_132 & 2155905152u));
        dp_lo[5] = _prmt_120_1;
        uint32_t _prmt_121_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_121_1) : "r"(_prmt_131), "r"(_prmt_129 | _prmt_133 & 2155905152u));
        dp_lo[6] = _prmt_121_1;
        uint32_t _prmt_122_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_122_1) : "r"(_prmt_131), "r"(_prmt_129 | _prmt_133 & 2155905152u));
        dp_lo[7] = _prmt_122_1;
        uint32_t _prmt_134;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_10 & 2004318071));
        uint32_t _prmt_135;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_135) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_10 & 2004318071) >> 16));
        uint32_t _prmt_136;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_136) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_10 & 2004318071));
        uint32_t _prmt_137;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_137) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_10 & 2004318071) >> 16));
        uint32_t _prmt_138;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_138) : "r"(w_hi_10 << 4), "r"(w_hi_10));
        uint32_t _prmt_139;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_139) : "r"(w_hi_10 << 4), "r"(w_hi_10));
        uint32_t _prmt_123_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_123_1) : "r"(_prmt_136), "r"(_prmt_134 | _prmt_138 & 2155905152u));
        dp_hi[4] = _prmt_123_1;
        uint32_t _prmt_124_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_124_1) : "r"(_prmt_136), "r"(_prmt_134 | _prmt_138 & 2155905152u));
        dp_hi[5] = _prmt_124_1;
        uint32_t _prmt_125_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_125_1) : "r"(_prmt_137), "r"(_prmt_135 | _prmt_139 & 2155905152u));
        dp_hi[6] = _prmt_125_1;
        uint32_t _prmt_126_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_126_1) : "r"(_prmt_137), "r"(_prmt_135 | _prmt_139 & 2155905152u));
        dp_hi[7] = _prmt_126_1;
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
        uint32_t _prmt_140;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_140) : "r"(scale_byte_12), "r"(scale_byte_14));
        sfv[0] = _prmt_140 & 2147516416u | (_prmt_140 & 2130738944) >> 4;
        uint32_t _prmt_141;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_141) : "r"(scale_byte_16), "r"(scale_byte_18));
        sfv[1] = _prmt_141 & 2147516416u | (_prmt_141 & 2130738944) >> 4;
        uint32_t w_lo_19;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_19) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_20;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_20) : "r"(vw[0]), "r"(vw[1]));
        uint32_t _prmt_142;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_142) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_19 & 2004318071));
        uint32_t _prmt_143;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_143) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_19 & 2004318071) >> 16));
        uint32_t _prmt_144;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_144) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_19 & 2004318071));
        uint32_t _prmt_145;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_145) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_19 & 2004318071) >> 16));
        uint32_t _prmt_146;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_146) : "r"(w_lo_19 << 4), "r"(w_lo_19));
        uint32_t _prmt_147;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_147) : "r"(w_lo_19 << 4), "r"(w_lo_19));
        uint32_t _prmt_127_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_127_1) : "r"(_prmt_144), "r"(_prmt_142 | _prmt_146 & 2155905152u));
        dp_lo[0] = _prmt_127_1;
        uint32_t _prmt_128_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_128_1) : "r"(_prmt_144), "r"(_prmt_142 | _prmt_146 & 2155905152u));
        dp_lo[1] = _prmt_128_1;
        uint32_t _prmt_129_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_129_1) : "r"(_prmt_145), "r"(_prmt_143 | _prmt_147 & 2155905152u));
        dp_lo[2] = _prmt_129_1;
        uint32_t _prmt_130_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_130_1) : "r"(_prmt_145), "r"(_prmt_143 | _prmt_147 & 2155905152u));
        dp_lo[3] = _prmt_130_1;
        uint32_t _prmt_148;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_148) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_20 & 2004318071));
        uint32_t _prmt_149;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_149) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_20 & 2004318071) >> 16));
        uint32_t _prmt_150;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_150) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_20 & 2004318071));
        uint32_t _prmt_151;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_151) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_20 & 2004318071) >> 16));
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_152) : "r"(w_hi_20 << 4), "r"(w_hi_20));
        uint32_t _prmt_153;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_153) : "r"(w_hi_20 << 4), "r"(w_hi_20));
        uint32_t _prmt_131_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_131_1) : "r"(_prmt_150), "r"(_prmt_148 | _prmt_152 & 2155905152u));
        dp_hi[0] = _prmt_131_1;
        uint32_t _prmt_132_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_132_1) : "r"(_prmt_150), "r"(_prmt_148 | _prmt_152 & 2155905152u));
        dp_hi[1] = _prmt_132_1;
        uint32_t _prmt_133_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_133_1) : "r"(_prmt_151), "r"(_prmt_149 | _prmt_153 & 2155905152u));
        dp_hi[2] = _prmt_133_1;
        uint32_t _prmt_134_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_134_1) : "r"(_prmt_151), "r"(_prmt_149 | _prmt_153 & 2155905152u));
        dp_hi[3] = _prmt_134_1;
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
        uint32_t w_lo_21;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_21) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_22;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_22) : "r"(vw[2]), "r"(vw[3]));
        uint32_t _prmt_154;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_154) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_21 & 2004318071));
        uint32_t _prmt_155;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_155) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_21 & 2004318071) >> 16));
        uint32_t _prmt_156;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_156) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_21 & 2004318071));
        uint32_t _prmt_157;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_157) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_21 & 2004318071) >> 16));
        uint32_t _prmt_158;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_158) : "r"(w_lo_21 << 4), "r"(w_lo_21));
        uint32_t _prmt_159;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_159) : "r"(w_lo_21 << 4), "r"(w_lo_21));
        uint32_t _prmt_135_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_135_1) : "r"(_prmt_156), "r"(_prmt_154 | _prmt_158 & 2155905152u));
        dp_lo[4] = _prmt_135_1;
        uint32_t _prmt_136_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_136_1) : "r"(_prmt_156), "r"(_prmt_154 | _prmt_158 & 2155905152u));
        dp_lo[5] = _prmt_136_1;
        uint32_t _prmt_137_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_137_1) : "r"(_prmt_157), "r"(_prmt_155 | _prmt_159 & 2155905152u));
        dp_lo[6] = _prmt_137_1;
        uint32_t _prmt_138_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_138_1) : "r"(_prmt_157), "r"(_prmt_155 | _prmt_159 & 2155905152u));
        dp_lo[7] = _prmt_138_1;
        uint32_t _prmt_160;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_160) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_22 & 2004318071));
        uint32_t _prmt_161;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_161) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_22 & 2004318071) >> 16));
        uint32_t _prmt_162;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_162) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_22 & 2004318071));
        uint32_t _prmt_163;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_163) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_22 & 2004318071) >> 16));
        uint32_t _prmt_164;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_164) : "r"(w_hi_22 << 4), "r"(w_hi_22));
        uint32_t _prmt_165;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_165) : "r"(w_hi_22 << 4), "r"(w_hi_22));
        uint32_t _prmt_139_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_139_1) : "r"(_prmt_162), "r"(_prmt_160 | _prmt_164 & 2155905152u));
        dp_hi[4] = _prmt_139_1;
        uint32_t _prmt_140_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_140_1) : "r"(_prmt_162), "r"(_prmt_160 | _prmt_164 & 2155905152u));
        dp_hi[5] = _prmt_140_1;
        uint32_t _prmt_141_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_141_1) : "r"(_prmt_163), "r"(_prmt_161 | _prmt_165 & 2155905152u));
        dp_hi[6] = _prmt_141_1;
        uint32_t _prmt_142_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_142_1) : "r"(_prmt_163), "r"(_prmt_161 | _prmt_165 & 2155905152u));
        dp_hi[7] = _prmt_142_1;
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
        int v_row_23 = v_stage + (32 + 2 * t) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])) : "r"(kv_addr + (unsigned int)(v_row_23 + v_col)));
        unsigned int scale_byte_24 = (unsigned int)kv_u8[v_row_23 + v_scale_col];
        int v_row_25 = v_stage + (32 + 2 * t + 1) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[1])) : "r"(kv_addr + (unsigned int)(v_row_25 + v_col)));
        unsigned int scale_byte_26 = (unsigned int)kv_u8[v_row_25 + v_scale_col];
        int v_row_27 = v_stage + (32 + 2 * t + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[2])) : "r"(kv_addr + (unsigned int)(v_row_27 + v_col)));
        unsigned int scale_byte_28 = (unsigned int)kv_u8[v_row_27 + v_scale_col];
        int v_row_29 = v_stage + (32 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_29 + v_col)));
        unsigned int scale_byte_30 = (unsigned int)kv_u8[v_row_29 + v_scale_col];
        uint32_t _prmt_166;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_166) : "r"(scale_byte_24), "r"(scale_byte_26));
        sfv[0] = _prmt_166 & 2147516416u | (_prmt_166 & 2130738944) >> 4;
        uint32_t _prmt_167;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_167) : "r"(scale_byte_28), "r"(scale_byte_30));
        sfv[1] = _prmt_167 & 2147516416u | (_prmt_167 & 2130738944) >> 4;
        uint32_t w_lo_31;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_31) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_32;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_32) : "r"(vw[0]), "r"(vw[1]));
        uint32_t _prmt_168;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_168) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_31 & 2004318071));
        uint32_t _prmt_169;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_169) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_31 & 2004318071) >> 16));
        uint32_t _prmt_170;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_170) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_31 & 2004318071));
        uint32_t _prmt_171;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_171) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_31 & 2004318071) >> 16));
        uint32_t _prmt_172;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_172) : "r"(w_lo_31 << 4), "r"(w_lo_31));
        uint32_t _prmt_173;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_173) : "r"(w_lo_31 << 4), "r"(w_lo_31));
        uint32_t _prmt_143_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_143_1) : "r"(_prmt_170), "r"(_prmt_168 | _prmt_172 & 2155905152u));
        dp_lo[0] = _prmt_143_1;
        uint32_t _prmt_144_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_144_1) : "r"(_prmt_170), "r"(_prmt_168 | _prmt_172 & 2155905152u));
        dp_lo[1] = _prmt_144_1;
        uint32_t _prmt_145_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_145_1) : "r"(_prmt_171), "r"(_prmt_169 | _prmt_173 & 2155905152u));
        dp_lo[2] = _prmt_145_1;
        uint32_t _prmt_146_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_146_1) : "r"(_prmt_171), "r"(_prmt_169 | _prmt_173 & 2155905152u));
        dp_lo[3] = _prmt_146_1;
        uint32_t _prmt_174;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_174) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_32 & 2004318071));
        uint32_t _prmt_175;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_175) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_32 & 2004318071) >> 16));
        uint32_t _prmt_176;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_176) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_32 & 2004318071));
        uint32_t _prmt_177;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_177) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_32 & 2004318071) >> 16));
        uint32_t _prmt_178;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_178) : "r"(w_hi_32 << 4), "r"(w_hi_32));
        uint32_t _prmt_179;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_179) : "r"(w_hi_32 << 4), "r"(w_hi_32));
        uint32_t _prmt_147_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_147_1) : "r"(_prmt_176), "r"(_prmt_174 | _prmt_178 & 2155905152u));
        dp_hi[0] = _prmt_147_1;
        uint32_t _prmt_148_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_148_1) : "r"(_prmt_176), "r"(_prmt_174 | _prmt_178 & 2155905152u));
        dp_hi[1] = _prmt_148_1;
        uint32_t _prmt_149_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_149_1) : "r"(_prmt_177), "r"(_prmt_175 | _prmt_179 & 2155905152u));
        dp_hi[2] = _prmt_149_1;
        uint32_t _prmt_150_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_150_1) : "r"(_prmt_177), "r"(_prmt_175 | _prmt_179 & 2155905152u));
        dp_hi[3] = _prmt_150_1;
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
        uint32_t w_lo_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_33) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_34) : "r"(vw[2]), "r"(vw[3]));
        uint32_t _prmt_180;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_180) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_33 & 2004318071));
        uint32_t _prmt_181;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_181) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_33 & 2004318071) >> 16));
        uint32_t _prmt_182;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_182) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_33 & 2004318071));
        uint32_t _prmt_183;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_183) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_33 & 2004318071) >> 16));
        uint32_t _prmt_184;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_184) : "r"(w_lo_33 << 4), "r"(w_lo_33));
        uint32_t _prmt_185;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_185) : "r"(w_lo_33 << 4), "r"(w_lo_33));
        uint32_t _prmt_151_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_151_1) : "r"(_prmt_182), "r"(_prmt_180 | _prmt_184 & 2155905152u));
        dp_lo[4] = _prmt_151_1;
        uint32_t _prmt_152_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_152_1) : "r"(_prmt_182), "r"(_prmt_180 | _prmt_184 & 2155905152u));
        dp_lo[5] = _prmt_152_1;
        uint32_t _prmt_153_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_153_1) : "r"(_prmt_183), "r"(_prmt_181 | _prmt_185 & 2155905152u));
        dp_lo[6] = _prmt_153_1;
        uint32_t _prmt_154_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_154_1) : "r"(_prmt_183), "r"(_prmt_181 | _prmt_185 & 2155905152u));
        dp_lo[7] = _prmt_154_1;
        uint32_t _prmt_186;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_186) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_34 & 2004318071));
        uint32_t _prmt_187;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_187) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_34 & 2004318071) >> 16));
        uint32_t _prmt_188;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_188) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_34 & 2004318071));
        uint32_t _prmt_189;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_189) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_34 & 2004318071) >> 16));
        uint32_t _prmt_190;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_190) : "r"(w_hi_34 << 4), "r"(w_hi_34));
        uint32_t _prmt_191;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_191) : "r"(w_hi_34 << 4), "r"(w_hi_34));
        uint32_t _prmt_155_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_155_1) : "r"(_prmt_188), "r"(_prmt_186 | _prmt_190 & 2155905152u));
        dp_hi[4] = _prmt_155_1;
        uint32_t _prmt_156_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_156_1) : "r"(_prmt_188), "r"(_prmt_186 | _prmt_190 & 2155905152u));
        dp_hi[5] = _prmt_156_1;
        uint32_t _prmt_157_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_157_1) : "r"(_prmt_189), "r"(_prmt_187 | _prmt_191 & 2155905152u));
        dp_hi[6] = _prmt_157_1;
        uint32_t _prmt_158_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_158_1) : "r"(_prmt_189), "r"(_prmt_187 | _prmt_191 & 2155905152u));
        dp_hi[7] = _prmt_158_1;
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
        uint32_t _prmt_192;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_192) : "r"(scale_byte_36), "r"(scale_byte_38));
        sfv[0] = _prmt_192 & 2147516416u | (_prmt_192 & 2130738944) >> 4;
        uint32_t _prmt_193;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_193) : "r"(scale_byte_40), "r"(scale_byte_42));
        sfv[1] = _prmt_193 & 2147516416u | (_prmt_193 & 2130738944) >> 4;
        uint32_t w_lo_43;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_43) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_44;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_44) : "r"(vw[0]), "r"(vw[1]));
        uint32_t _prmt_194;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_194) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_43 & 2004318071));
        uint32_t _prmt_195;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_195) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_43 & 2004318071) >> 16));
        uint32_t _prmt_196;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_196) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_43 & 2004318071));
        uint32_t _prmt_197;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_197) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_43 & 2004318071) >> 16));
        uint32_t _prmt_198;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_198) : "r"(w_lo_43 << 4), "r"(w_lo_43));
        uint32_t _prmt_199;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_199) : "r"(w_lo_43 << 4), "r"(w_lo_43));
        uint32_t _prmt_159_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_159_1) : "r"(_prmt_196), "r"(_prmt_194 | _prmt_198 & 2155905152u));
        dp_lo[0] = _prmt_159_1;
        uint32_t _prmt_160_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_160_1) : "r"(_prmt_196), "r"(_prmt_194 | _prmt_198 & 2155905152u));
        dp_lo[1] = _prmt_160_1;
        uint32_t _prmt_161_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_161_1) : "r"(_prmt_197), "r"(_prmt_195 | _prmt_199 & 2155905152u));
        dp_lo[2] = _prmt_161_1;
        uint32_t _prmt_162_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_162_1) : "r"(_prmt_197), "r"(_prmt_195 | _prmt_199 & 2155905152u));
        dp_lo[3] = _prmt_162_1;
        uint32_t _prmt_200;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_200) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_44 & 2004318071));
        uint32_t _prmt_201;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_201) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_44 & 2004318071) >> 16));
        uint32_t _prmt_202;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_202) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_44 & 2004318071));
        uint32_t _prmt_203;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_203) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_44 & 2004318071) >> 16));
        uint32_t _prmt_204;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_204) : "r"(w_hi_44 << 4), "r"(w_hi_44));
        uint32_t _prmt_205;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_205) : "r"(w_hi_44 << 4), "r"(w_hi_44));
        uint32_t _prmt_163_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_163_1) : "r"(_prmt_202), "r"(_prmt_200 | _prmt_204 & 2155905152u));
        dp_hi[0] = _prmt_163_1;
        uint32_t _prmt_164_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_164_1) : "r"(_prmt_202), "r"(_prmt_200 | _prmt_204 & 2155905152u));
        dp_hi[1] = _prmt_164_1;
        uint32_t _prmt_165_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_165_1) : "r"(_prmt_203), "r"(_prmt_201 | _prmt_205 & 2155905152u));
        dp_hi[2] = _prmt_165_1;
        uint32_t _prmt_166_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_166_1) : "r"(_prmt_203), "r"(_prmt_201 | _prmt_205 & 2155905152u));
        dp_hi[3] = _prmt_166_1;
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
        uint32_t w_lo_45;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_45) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_46;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_46) : "r"(vw[2]), "r"(vw[3]));
        uint32_t _prmt_206;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_206) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_45 & 2004318071));
        uint32_t _prmt_207;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_207) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_lo_45 & 2004318071) >> 16));
        uint32_t _prmt_208;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_208) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_45 & 2004318071));
        uint32_t _prmt_209;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_209) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_lo_45 & 2004318071) >> 16));
        uint32_t _prmt_210;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_210) : "r"(w_lo_45 << 4), "r"(w_lo_45));
        uint32_t _prmt_211;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_211) : "r"(w_lo_45 << 4), "r"(w_lo_45));
        uint32_t _prmt_167_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_167_1) : "r"(_prmt_208), "r"(_prmt_206 | _prmt_210 & 2155905152u));
        dp_lo[4] = _prmt_167_1;
        uint32_t _prmt_168_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_168_1) : "r"(_prmt_208), "r"(_prmt_206 | _prmt_210 & 2155905152u));
        dp_lo[5] = _prmt_168_1;
        uint32_t _prmt_169_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_169_1) : "r"(_prmt_209), "r"(_prmt_207 | _prmt_211 & 2155905152u));
        dp_lo[6] = _prmt_169_1;
        uint32_t _prmt_170_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_170_1) : "r"(_prmt_209), "r"(_prmt_207 | _prmt_211 & 2155905152u));
        dp_lo[7] = _prmt_170_1;
        uint32_t _prmt_212;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_212) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_46 & 2004318071));
        uint32_t _prmt_213;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_213) : "r"(lut_regs[0]), "r"(2088533116), "r"((w_hi_46 & 2004318071) >> 16));
        uint32_t _prmt_214;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_214) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_46 & 2004318071));
        uint32_t _prmt_215;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_215) : "r"(lut_regs[1]), "r"(3229630464u), "r"((w_hi_46 & 2004318071) >> 16));
        uint32_t _prmt_216;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(_prmt_216) : "r"(w_hi_46 << 4), "r"(w_hi_46));
        uint32_t _prmt_217;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(_prmt_217) : "r"(w_hi_46 << 4), "r"(w_hi_46));
        uint32_t _prmt_171_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_171_1) : "r"(_prmt_214), "r"(_prmt_212 | _prmt_216 & 2155905152u));
        dp_hi[4] = _prmt_171_1;
        uint32_t _prmt_172_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_172_1) : "r"(_prmt_214), "r"(_prmt_212 | _prmt_216 & 2155905152u));
        dp_hi[5] = _prmt_172_1;
        uint32_t _prmt_173_1;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_173_1) : "r"(_prmt_215), "r"(_prmt_213 | _prmt_217 & 2155905152u));
        dp_hi[6] = _prmt_173_1;
        uint32_t _prmt_174_1;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_174_1) : "r"(_prmt_215), "r"(_prmt_213 | _prmt_217 & 2155905152u));
        dp_hi[7] = _prmt_174_1;
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
            int _max_175 = ((phys[0]) > (0) ? (phys[0]) : (0));
            unsigned long long block_3 = (unsigned long long)_max_175;
            unsigned long long local_u64_3 = (unsigned long long)local[0];
            unsigned long long data_off_5 = block_3 * data_stride_block_u64 + local_u64_3 * data_stride_token_u64;
            unsigned long long sf_off_4 = block_3 * sf_stride_block_u64 + local_u64_3 * sf_stride_token_u64;
            unsigned int w0_1_2 = (unsigned int)data_off_5;
            unsigned int w1_2_2 = (unsigned int)(data_off_5 >> 32);
            unsigned int w2_3_2 = (unsigned int)sf_off_4;
            unsigned int w3_4_2 = (unsigned int)(sf_off_4 >> 32);
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
        float _rcp_176 = approx_rcp(fmaxf(l[0], 1e-20f));
        float inv = output_scale * _rcp_176;
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
                const uint4* _vptr_177 = reinterpret_cast<const uint4*>(gate + gate_off + 0);
                uint4 _vld_177[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_177[_blk] = _vptr_177[_blk];
                    uint32_t* _vpairs_177 = reinterpret_cast<uint32_t*>(&_vld_177[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_177[_pair]) << 16);
                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_177[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn = (float)(__nv_bfloat16)v[0];
            float _exp2_178 = approx_exp2((-_vec_load_8[0]) * 1.442695f);
            float sigmoid = approx_rcp(1.0f + _exp2_178);
            gated[0] = attn * sigmoid;
            float attn_0 = (float)(__nv_bfloat16)v[1];
            float _exp2_179 = approx_exp2((-_vec_load_8[1]) * 1.442695f);
            float sigmoid_1 = approx_rcp(1.0f + _exp2_179);
            gated[1] = attn_0 * sigmoid_1;
            float attn_2 = (float)(__nv_bfloat16)v[2];
            float _exp2_180 = approx_exp2((-_vec_load_8[2]) * 1.442695f);
            float sigmoid_3 = approx_rcp(1.0f + _exp2_180);
            gated[2] = attn_2 * sigmoid_3;
            float attn_4 = (float)(__nv_bfloat16)v[3];
            float _exp2_181 = approx_exp2((-_vec_load_8[3]) * 1.442695f);
            float sigmoid_5 = approx_rcp(1.0f + _exp2_181);
            gated[3] = attn_4 * sigmoid_5;
            float attn_6 = (float)(__nv_bfloat16)v[4];
            float _exp2_182 = approx_exp2((-_vec_load_8[4]) * 1.442695f);
            float sigmoid_7 = approx_rcp(1.0f + _exp2_182);
            gated[4] = attn_6 * sigmoid_7;
            float attn_8 = (float)(__nv_bfloat16)v[5];
            float _exp2_183 = approx_exp2((-_vec_load_8[5]) * 1.442695f);
            float sigmoid_9 = approx_rcp(1.0f + _exp2_183);
            gated[5] = attn_8 * sigmoid_9;
            float attn_10 = (float)(__nv_bfloat16)v[6];
            float _exp2_184 = approx_exp2((-_vec_load_8[6]) * 1.442695f);
            float sigmoid_11 = approx_rcp(1.0f + _exp2_184);
            gated[6] = attn_10 * sigmoid_11;
            float attn_12 = (float)(__nv_bfloat16)v[7];
            float _exp2_185 = approx_exp2((-_vec_load_8[7]) * 1.442695f);
            float sigmoid_13 = approx_rcp(1.0f + _exp2_185);
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
                const uint4* _vptr_186 = reinterpret_cast<const uint4*>(gate + (gate_off + 8) + 0);
                uint4 _vld_186[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_186[_blk] = _vptr_186[_blk];
                    uint32_t* _vpairs_186 = reinterpret_cast<uint32_t*>(&_vld_186[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_186[_pair]) << 16);
                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_186[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_14 = (float)(__nv_bfloat16)v[8];
            float _exp2_187 = approx_exp2((-_vec_load_9[0]) * 1.442695f);
            float sigmoid_15 = approx_rcp(1.0f + _exp2_187);
            gated[0] = attn_14 * sigmoid_15;
            float attn_16 = (float)(__nv_bfloat16)v[9];
            float _exp2_188 = approx_exp2((-_vec_load_9[1]) * 1.442695f);
            float sigmoid_17 = approx_rcp(1.0f + _exp2_188);
            gated[1] = attn_16 * sigmoid_17;
            float attn_18 = (float)(__nv_bfloat16)v[10];
            float _exp2_189 = approx_exp2((-_vec_load_9[2]) * 1.442695f);
            float sigmoid_19 = approx_rcp(1.0f + _exp2_189);
            gated[2] = attn_18 * sigmoid_19;
            float attn_20 = (float)(__nv_bfloat16)v[11];
            float _exp2_190 = approx_exp2((-_vec_load_9[3]) * 1.442695f);
            float sigmoid_21 = approx_rcp(1.0f + _exp2_190);
            gated[3] = attn_20 * sigmoid_21;
            float attn_22 = (float)(__nv_bfloat16)v[12];
            float _exp2_191 = approx_exp2((-_vec_load_9[4]) * 1.442695f);
            float sigmoid_23 = approx_rcp(1.0f + _exp2_191);
            gated[4] = attn_22 * sigmoid_23;
            float attn_24 = (float)(__nv_bfloat16)v[13];
            float _exp2_192 = approx_exp2((-_vec_load_9[5]) * 1.442695f);
            float sigmoid_25 = approx_rcp(1.0f + _exp2_192);
            gated[5] = attn_24 * sigmoid_25;
            float attn_26 = (float)(__nv_bfloat16)v[14];
            float _exp2_193 = approx_exp2((-_vec_load_9[6]) * 1.442695f);
            float sigmoid_27 = approx_rcp(1.0f + _exp2_193);
            gated[6] = attn_26 * sigmoid_27;
            float attn_28 = (float)(__nv_bfloat16)v[15];
            float _exp2_194 = approx_exp2((-_vec_load_9[7]) * 1.442695f);
            float sigmoid_29 = approx_rcp(1.0f + _exp2_194);
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
                float _log2_195;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_195) : "f"(l[0]));
                *(reinterpret_cast<float*>(partial_lse + partial_row) + (0)) = ((has_values) ? m[0] + _log2_195 : -CUDART_INF_F);
            }
        }
    }
    int head_61 = g + 8;
    if (head_61 < group_size) {
        bool has_values_1 = l[1] > 0.0f;
        float _rcp_196 = approx_rcp(fmaxf(l[1], 1e-20f));
        float inv_1 = output_scale * _rcp_196;
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
                const uint4* _vptr_197 = reinterpret_cast<const uint4*>(gate + gate_off_1 + 0);
                uint4 _vld_197[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_197[_blk] = _vptr_197[_blk];
                    uint32_t* _vpairs_197 = reinterpret_cast<uint32_t*>(&_vld_197[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_197[_pair]) << 16);
                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_197[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_1 = (float)(__nv_bfloat16)v[0];
            float _exp2_198 = approx_exp2((-_vec_load_10[0]) * 1.442695f);
            float sigmoid_2 = approx_rcp(1.0f + _exp2_198);
            gated[0] = attn_1 * sigmoid_2;
            float attn_0_1 = (float)(__nv_bfloat16)v[1];
            float _exp2_199 = approx_exp2((-_vec_load_10[1]) * 1.442695f);
            float sigmoid_1_1 = approx_rcp(1.0f + _exp2_199);
            gated[1] = attn_0_1 * sigmoid_1_1;
            float attn_2_1 = (float)(__nv_bfloat16)v[2];
            float _exp2_200 = approx_exp2((-_vec_load_10[2]) * 1.442695f);
            float sigmoid_3_1 = approx_rcp(1.0f + _exp2_200);
            gated[2] = attn_2_1 * sigmoid_3_1;
            float attn_4_1 = (float)(__nv_bfloat16)v[3];
            float _exp2_201 = approx_exp2((-_vec_load_10[3]) * 1.442695f);
            float sigmoid_5_1 = approx_rcp(1.0f + _exp2_201);
            gated[3] = attn_4_1 * sigmoid_5_1;
            float attn_6_1 = (float)(__nv_bfloat16)v[4];
            float _exp2_202 = approx_exp2((-_vec_load_10[4]) * 1.442695f);
            float sigmoid_7_1 = approx_rcp(1.0f + _exp2_202);
            gated[4] = attn_6_1 * sigmoid_7_1;
            float attn_8_1 = (float)(__nv_bfloat16)v[5];
            float _exp2_203 = approx_exp2((-_vec_load_10[5]) * 1.442695f);
            float sigmoid_9_1 = approx_rcp(1.0f + _exp2_203);
            gated[5] = attn_8_1 * sigmoid_9_1;
            float attn_10_1 = (float)(__nv_bfloat16)v[6];
            float _exp2_204 = approx_exp2((-_vec_load_10[6]) * 1.442695f);
            float sigmoid_11_1 = approx_rcp(1.0f + _exp2_204);
            gated[6] = attn_10_1 * sigmoid_11_1;
            float attn_12_1 = (float)(__nv_bfloat16)v[7];
            float _exp2_205 = approx_exp2((-_vec_load_10[7]) * 1.442695f);
            float sigmoid_13_1 = approx_rcp(1.0f + _exp2_205);
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
                const uint4* _vptr_206 = reinterpret_cast<const uint4*>(gate + (gate_off_1 + 8) + 0);
                uint4 _vld_206[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_206[_blk] = _vptr_206[_blk];
                    uint32_t* _vpairs_206 = reinterpret_cast<uint32_t*>(&_vld_206[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_206[_pair]) << 16);
                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_206[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_14_1 = (float)(__nv_bfloat16)v[8];
            float _exp2_207 = approx_exp2((-_vec_load_11[0]) * 1.442695f);
            float sigmoid_15_1 = approx_rcp(1.0f + _exp2_207);
            gated[0] = attn_14_1 * sigmoid_15_1;
            float attn_16_1 = (float)(__nv_bfloat16)v[9];
            float _exp2_208 = approx_exp2((-_vec_load_11[1]) * 1.442695f);
            float sigmoid_17_1 = approx_rcp(1.0f + _exp2_208);
            gated[1] = attn_16_1 * sigmoid_17_1;
            float attn_18_1 = (float)(__nv_bfloat16)v[10];
            float _exp2_209 = approx_exp2((-_vec_load_11[2]) * 1.442695f);
            float sigmoid_19_1 = approx_rcp(1.0f + _exp2_209);
            gated[2] = attn_18_1 * sigmoid_19_1;
            float attn_20_1 = (float)(__nv_bfloat16)v[11];
            float _exp2_210 = approx_exp2((-_vec_load_11[3]) * 1.442695f);
            float sigmoid_21_1 = approx_rcp(1.0f + _exp2_210);
            gated[3] = attn_20_1 * sigmoid_21_1;
            float attn_22_1 = (float)(__nv_bfloat16)v[12];
            float _exp2_211 = approx_exp2((-_vec_load_11[4]) * 1.442695f);
            float sigmoid_23_1 = approx_rcp(1.0f + _exp2_211);
            gated[4] = attn_22_1 * sigmoid_23_1;
            float attn_24_1 = (float)(__nv_bfloat16)v[13];
            float _exp2_212 = approx_exp2((-_vec_load_11[5]) * 1.442695f);
            float sigmoid_25_1 = approx_rcp(1.0f + _exp2_212);
            gated[5] = attn_24_1 * sigmoid_25_1;
            float attn_26_1 = (float)(__nv_bfloat16)v[14];
            float _exp2_213 = approx_exp2((-_vec_load_11[6]) * 1.442695f);
            float sigmoid_27_1 = approx_rcp(1.0f + _exp2_213);
            gated[6] = attn_26_1 * sigmoid_27_1;
            float attn_28_1 = (float)(__nv_bfloat16)v[15];
            float _exp2_214 = approx_exp2((-_vec_load_11[7]) * 1.442695f);
            float sigmoid_29_1 = approx_rcp(1.0f + _exp2_214);
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
                float _log2_215;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_215) : "f"(l[1]));
                *(reinterpret_cast<float*>(partial_lse + partial_row_1) + (0)) = ((has_values_1) ? m[1] + _log2_215 : -CUDART_INF_F);
            }
        }
    }
}

} // extern "C"
