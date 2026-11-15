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
#if __CUDA_ARCH__ == 800
#else
#define SMEM_LSE_TAB2_OFF 35840
#define SMEM_LSE_TAB2_STAGE_BYTES 960
#define SMEM_LSE_TAB2_STRIDE 960
#endif
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
#if __CUDA_ARCH__ == 800
#else
    float* lse_tab2 = reinterpret_cast<float*>(smem_raw + 35840);
    const int lse_tab2_addr = smem + 35840;
#endif

    // === Task calls (declaration order) ===
    int row = blockIdx.x;
    int kv_head = blockIdx.y;
#if __CUDA_ARCH__ == 800
    int linear = blockIdx.x + gridDim.x * blockIdx.y;
    row = linear / gridDim.y;
    kv_head = linear - row * gridDim.y;
#endif
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
    int column_0 = (split + num_splits) * 64 + token;
    int safe_column_0 = ((column_0) < (topk - 1) ? (column_0) : (topk - 1));
    int raw_0 = indices[indices_row + (long long)safe_column_0];
    idx[1] = raw_0;
    int column_1 = (split + 2 * num_splits) * 64 + token;
    int safe_column_1 = ((column_1) < (topk - 1) ? (column_1) : (topk - 1));
    int raw_2 = indices[indices_row + (long long)safe_column_1];
    idx[2] = raw_2;
    long long q_row = row_i64 * q_stride_row;
    unsigned int zero_word = 0;
    int e_1_0 = tid_0;
    int e_step_1_0 = e_1_0 >> 5;
    int e_lane_1_0 = e_1_0 & 31;
    int e_g_1_0 = e_lane_1_0 >> 2;
    int e_t_1_0 = e_lane_1_0 & 3;
    bool lo_ok_1_0 = e_g_1_0 < group_size;
    bool hi_ok_1_0 = e_g_1_0 + 8 < group_size;
    long long dim_1_0 = q_row + (long long)(64 * e_t_1_0 + 4 * e_step_1_0);
    long long lo_src_1_0 = dim_1_0 + (long long)(first_head + e_g_1_0) * q_stride_head;
    long long hi_src_1_0 = lo_src_1_0 + 8 * q_stride_head;
    unsigned int w0_1_0 = zero_word;
    unsigned int w1_1_0 = zero_word;
    unsigned int w2_1_0 = zero_word;
    unsigned int w3_1_0 = zero_word;
    if (lo_ok_1_0) {
        unsigned int _vec_load_0[2];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(q + lo_src_1_0 + 0);
            uint2* _vdst_2 = reinterpret_cast<uint2*>(&_vec_load_0[0]);
            *_vdst_2 = _vld_2;
        }
        w0_1_0 = _vec_load_0[0];
        w2_1_0 = _vec_load_0[1];
    }
    if (hi_ok_1_0) {
        unsigned int _vec_load_12_0[2];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(q + hi_src_1_0 + 0);
            uint2* _vdst_3 = reinterpret_cast<uint2*>(&_vec_load_12_0[0]);
            *_vdst_3 = _vld_3;
        }
        w1_1_0 = _vec_load_12_0[0];
        w3_1_0 = _vec_load_12_0[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_1_0 + e_lane_1_0))), "r"(w0_1_0), "r"(w1_1_0), "r"(w2_1_0), "r"(w3_1_0) : "memory");
    int e_1_1 = tid_0 + 128;
    int e_step_1_1 = e_1_1 >> 5;
    int e_lane_1_1 = e_1_1 & 31;
    int e_g_1_1 = e_lane_1_1 >> 2;
    int e_t_1_1 = e_lane_1_1 & 3;
    bool lo_ok_1_1 = e_g_1_1 < group_size;
    bool hi_ok_1_1 = e_g_1_1 + 8 < group_size;
    long long dim_1_1 = q_row + (long long)(64 * e_t_1_1 + 4 * e_step_1_1);
    long long lo_src_1_1 = dim_1_1 + (long long)(first_head + e_g_1_1) * q_stride_head;
    long long hi_src_1_1 = lo_src_1_1 + 8 * q_stride_head;
    unsigned int w0_1_1 = zero_word;
    unsigned int w1_1_1 = zero_word;
    unsigned int w2_1_1 = zero_word;
    unsigned int w3_1_1 = zero_word;
    if (lo_ok_1_1) {
        unsigned int _vec_load_1[2];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(q + lo_src_1_1 + 0);
            uint2* _vdst_4 = reinterpret_cast<uint2*>(&_vec_load_1[0]);
            *_vdst_4 = _vld_4;
        }
        w0_1_1 = _vec_load_1[0];
        w2_1_1 = _vec_load_1[1];
    }
    if (hi_ok_1_1) {
        unsigned int _vec_load_12_1[2];
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(q + hi_src_1_1 + 0);
            uint2* _vdst_5 = reinterpret_cast<uint2*>(&_vec_load_12_1[0]);
            *_vdst_5 = _vld_5;
        }
        w1_1_1 = _vec_load_12_1[0];
        w3_1_1 = _vec_load_12_1[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_1_1 + e_lane_1_1))), "r"(w0_1_1), "r"(w1_1_1), "r"(w2_1_1), "r"(w3_1_1) : "memory");
    int e_1_2 = tid_0 + 256;
    int e_step_1_2 = e_1_2 >> 5;
    int e_lane_1_2 = e_1_2 & 31;
    int e_g_1_2 = e_lane_1_2 >> 2;
    int e_t_1_2 = e_lane_1_2 & 3;
    bool lo_ok_1_2 = e_g_1_2 < group_size;
    bool hi_ok_1_2 = e_g_1_2 + 8 < group_size;
    long long dim_1_2 = q_row + (long long)(64 * e_t_1_2 + 4 * e_step_1_2);
    long long lo_src_1_2 = dim_1_2 + (long long)(first_head + e_g_1_2) * q_stride_head;
    long long hi_src_1_2 = lo_src_1_2 + 8 * q_stride_head;
    unsigned int w0_1_2 = zero_word;
    unsigned int w1_1_2 = zero_word;
    unsigned int w2_1_2 = zero_word;
    unsigned int w3_1_2 = zero_word;
    if (lo_ok_1_2) {
        unsigned int _vec_load_2[2];
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(q + lo_src_1_2 + 0);
            uint2* _vdst_6 = reinterpret_cast<uint2*>(&_vec_load_2[0]);
            *_vdst_6 = _vld_6;
        }
        w0_1_2 = _vec_load_2[0];
        w2_1_2 = _vec_load_2[1];
    }
    if (hi_ok_1_2) {
        unsigned int _vec_load_12_2[2];
        {
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(q + hi_src_1_2 + 0);
            uint2* _vdst_7 = reinterpret_cast<uint2*>(&_vec_load_12_2[0]);
            *_vdst_7 = _vld_7;
        }
        w1_1_2 = _vec_load_12_2[0];
        w3_1_2 = _vec_load_12_2[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_1_2 + e_lane_1_2))), "r"(w0_1_2), "r"(w1_1_2), "r"(w2_1_2), "r"(w3_1_2) : "memory");
    int e_1_3 = tid_0 + 384;
    int e_step_1_3 = e_1_3 >> 5;
    int e_lane_1_3 = e_1_3 & 31;
    int e_g_1_3 = e_lane_1_3 >> 2;
    int e_t_1_3 = e_lane_1_3 & 3;
    bool lo_ok_1_3 = e_g_1_3 < group_size;
    bool hi_ok_1_3 = e_g_1_3 + 8 < group_size;
    long long dim_1_3 = q_row + (long long)(64 * e_t_1_3 + 4 * e_step_1_3);
    long long lo_src_1_3 = dim_1_3 + (long long)(first_head + e_g_1_3) * q_stride_head;
    long long hi_src_1_3 = lo_src_1_3 + 8 * q_stride_head;
    unsigned int w0_1_3 = zero_word;
    unsigned int w1_1_3 = zero_word;
    unsigned int w2_1_3 = zero_word;
    unsigned int w3_1_3 = zero_word;
    if (lo_ok_1_3) {
        unsigned int _vec_load_3[2];
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(q + lo_src_1_3 + 0);
            uint2* _vdst_8 = reinterpret_cast<uint2*>(&_vec_load_3[0]);
            *_vdst_8 = _vld_8;
        }
        w0_1_3 = _vec_load_3[0];
        w2_1_3 = _vec_load_3[1];
    }
    if (hi_ok_1_3) {
        unsigned int _vec_load_12_3[2];
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(q + hi_src_1_3 + 0);
            uint2* _vdst_9 = reinterpret_cast<uint2*>(&_vec_load_12_3[0]);
            *_vdst_9 = _vld_9;
        }
        w1_1_3 = _vec_load_12_3[0];
        w3_1_3 = _vec_load_12_3[1];
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_frag_addr + (unsigned int)(16 * (32 * e_step_1_3 + e_lane_1_3))), "r"(w0_1_3), "r"(w1_1_3), "r"(w2_1_3), "r"(w3_1_3) : "memory");
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
            unsigned int src_size_0 = (v_bits[0] & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0), "l"(v_src[0]), "r"(src_size_0), "l"(0x12F0000000000000ULL));
            unsigned int src_size_2 = (v_bits[0] >> 4 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0 + 576), "l"(v_src[1]), "r"(src_size_2), "l"(0x12F0000000000000ULL));
            unsigned int src_size_4 = (v_bits[0] >> 8 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0 + 1152), "l"(v_src[2]), "r"(src_size_4), "l"(0x12F0000000000000ULL));
            unsigned int src_size_6 = (v_bits[0] >> 12 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0 + 1728), "l"(v_src[3]), "r"(src_size_6), "l"(0x12F0000000000000ULL));
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
            unsigned int w0_1_4 = (unsigned int)data_off_3;
            unsigned int w1_2_1 = (unsigned int)(data_off_3 >> 32);
            unsigned int w2_3_1 = (unsigned int)sf_off_2;
            unsigned int w3_4_1 = (unsigned int)(sf_off_2 >> 32);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(1024 + token * 16)), "r"(w0_1_4), "r"(w1_2_1), "r"(w2_3_1), "r"(w3_4_1) : "memory");
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
            unsigned int src_size_7 = (unsigned int)nbytes;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k), "l"(k_src), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 16), "l"(k_src + 16), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 32), "l"(k_src + 32), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 48), "l"(k_src + 48), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(sf_dst_1), "l"(sf_base_1 + sf_off_3), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216), "l"(v_src_2), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216 + 16), "l"(v_src_2 + 16), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216 + 32), "l"(v_src_2 + 32), "r"(src_size_7), "l"(0x12F0000000000000ULL));
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k + 9216 + 48), "l"(v_src_2 + 48), "r"(src_size_7), "l"(0x12F0000000000000ULL));
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
#endif
        if (warp_1 < 2 && multi) {
            int column_0_0 = (tile + 3 * num_splits) * 64 + token;
            bool in_range_1 = column_0_0 < topk && tile_end > tile + 3 * num_splits;
            int safe_column_3 = ((column_0_0) < (topk - 1) ? (column_0_0) : (topk - 1));
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
            unsigned int src_size_8 = (bits_sub_1 & 1) << 4;
            v_src[0] = v_src_base_1 + data_off_6;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_k0_1), "l"(k_src_base_1 + data_off_6), "r"(src_size_8), "l"(0x12F0000000000000ULL));
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
        uint32_t prmt;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt) : "r"(sf_words[0]), "r"(sf_words[0]));
        sf0[0] = prmt & 2147516416u | (prmt & 2130738944) >> 4;
        sf0[1] = prmt << 8 & 2147516416u | (prmt << 8 & 2130738944) >> 4;
        uint32_t prmt_0;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_0) : "r"(sf_words[1]), "r"(sf_words[1]));
        sf1[0] = prmt_0 & 2147516416u | (prmt_0 & 2130738944) >> 4;
        sf1[1] = prmt_0 << 8 & 2147516416u | (prmt_0 << 8 & 2130738944) >> 4;
        s[0] = 0.0f;
        s[1] = 0.0f;
        s[2] = 0.0f;
        s[3] = 0.0f;
        s[4] = 0.0f;
        s[5] = 0.0f;
        s[6] = 0.0f;
        s[7] = 0.0f;
        unsigned int data_col_1_0 = (unsigned int)(32 * t);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 3]))
            : "r"(k0_addr + data_col_1_0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 3]))
            : "r"(k1_addr + data_col_1_0));
        uint32_t prmt_1;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_1) : "r"(raw0[0] & 2004318071), "r"(raw0[0] & 2004318071));
        uint32_t prmt_1_0;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_0) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_0) : "r"(2071689984), "r"(2088533116), "r"(raw0[0] & 2004318071));
#endif
        uint32_t prmt_2;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2) : "r"(2071689984), "r"(2088533116), "r"(prmt_1));
#endif
        uint32_t prmt_3;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#endif
        uint32_t prmt_4;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_1));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_1));
#endif
        uint32_t prmt_5;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t prmt_6;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_23;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_23) : "r"(prmt_3), "r"(prmt_1_0 | prmt_5 & 2155905152u));
        d0[0] = _prmt_23;
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_24) : "r"(prmt_3), "r"(prmt_1_0 | prmt_5 & 2155905152u));
        d0[1] = _prmt_24;
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_25) : "r"(prmt_4), "r"(prmt_2 | prmt_6 & 2155905152u));
        d0[2] = _prmt_25;
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_26) : "r"(prmt_4), "r"(prmt_2 | prmt_6 & 2155905152u));
        d0[3] = _prmt_26;
        uint32_t prmt_7;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_7) : "r"(raw1[0] & 2004318071), "r"(raw1[0] & 2004318071));
        uint32_t prmt_1_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_1) : "r"(2071689984), "r"(2088533116), "r"(raw1[0] & 2004318071));
#endif
        uint32_t prmt_2_0;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_0) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_7));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_0) : "r"(2071689984), "r"(2088533116), "r"(prmt_7));
#endif
        uint32_t prmt_3_0;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_0) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_0) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#endif
        uint32_t prmt_4_0;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_0) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_7));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_0) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_7));
#endif
        uint32_t prmt_5_0;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_0) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t prmt_6_0;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_0) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_27) : "r"(prmt_3_0), "r"(prmt_1_1 | prmt_5_0 & 2155905152u));
        d1[0] = _prmt_27;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_28) : "r"(prmt_3_0), "r"(prmt_1_1 | prmt_5_0 & 2155905152u));
        d1[1] = _prmt_28;
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_29) : "r"(prmt_4_0), "r"(prmt_2_0 | prmt_6_0 & 2155905152u));
        d1[2] = _prmt_29;
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_30) : "r"(prmt_4_0), "r"(prmt_2_0 | prmt_6_0 & 2155905152u));
        d1[3] = _prmt_30;
        uint32_t prmt_8;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_8) : "r"(sf0[0]), "r"(0));
        uint32_t prmt_9;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_9) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_0;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_0) : "r"(d0[0]), "r"(prmt_8), "r"(2147516416u));
        d0[0] = _bf16x2_fma_0;
        uint32_t _bf16x2_fma_128_0;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_0) : "r"(d1[0]), "r"(prmt_9), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_0;
        uint32_t _bf16x2_fma_1;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_1) : "r"(d0[1]), "r"(prmt_8), "r"(2147516416u));
        d0[1] = _bf16x2_fma_1;
        uint32_t _bf16x2_fma_128_1;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_1) : "r"(d1[1]), "r"(prmt_9), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_1;
        uint32_t _bf16x2_fma_2;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_2) : "r"(d0[2]), "r"(prmt_8), "r"(2147516416u));
        d0[2] = _bf16x2_fma_2;
        uint32_t _bf16x2_fma_128_2;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_2) : "r"(d1[2]), "r"(prmt_9), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_2;
        uint32_t _bf16x2_fma_3;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_3) : "r"(d0[3]), "r"(prmt_8), "r"(2147516416u));
        d0[3] = _bf16x2_fma_3;
        uint32_t _bf16x2_fma_128_3;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_3) : "r"(d1[3]), "r"(prmt_9), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_3;
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
        uint32_t prmt_10;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_10) : "r"(raw0[1] & 2004318071), "r"(raw0[1] & 2004318071));
        uint32_t prmt_1_2;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_2) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_2) : "r"(2071689984), "r"(2088533116), "r"(raw0[1] & 2004318071));
#endif
        uint32_t prmt_2_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_1) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_10));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_1) : "r"(2071689984), "r"(2088533116), "r"(prmt_10));
#endif
        uint32_t prmt_3_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_1) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#endif
        uint32_t prmt_4_1;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_1) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_10));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_1) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_10));
#endif
        uint32_t prmt_5_1;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_1) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t prmt_6_1;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_1) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_31) : "r"(prmt_3_1), "r"(prmt_1_2 | prmt_5_1 & 2155905152u));
        d0[0] = _prmt_31;
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_32) : "r"(prmt_3_1), "r"(prmt_1_2 | prmt_5_1 & 2155905152u));
        d0[1] = _prmt_32;
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_33) : "r"(prmt_4_1), "r"(prmt_2_1 | prmt_6_1 & 2155905152u));
        d0[2] = _prmt_33;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_34) : "r"(prmt_4_1), "r"(prmt_2_1 | prmt_6_1 & 2155905152u));
        d0[3] = _prmt_34;
        uint32_t prmt_11;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_11) : "r"(raw1[1] & 2004318071), "r"(raw1[1] & 2004318071));
        uint32_t prmt_1_3;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_3) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_3) : "r"(2071689984), "r"(2088533116), "r"(raw1[1] & 2004318071));
#endif
        uint32_t prmt_2_2;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_2) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_11));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_2) : "r"(2071689984), "r"(2088533116), "r"(prmt_11));
#endif
        uint32_t prmt_3_2;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_2) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_2) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#endif
        uint32_t prmt_4_2;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_2) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_11));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_2) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_11));
#endif
        uint32_t prmt_5_2;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_2) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t prmt_6_2;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_2) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_35) : "r"(prmt_3_2), "r"(prmt_1_3 | prmt_5_2 & 2155905152u));
        d1[0] = _prmt_35;
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_36) : "r"(prmt_3_2), "r"(prmt_1_3 | prmt_5_2 & 2155905152u));
        d1[1] = _prmt_36;
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_37) : "r"(prmt_4_2), "r"(prmt_2_2 | prmt_6_2 & 2155905152u));
        d1[2] = _prmt_37;
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_38) : "r"(prmt_4_2), "r"(prmt_2_2 | prmt_6_2 & 2155905152u));
        d1[3] = _prmt_38;
        uint32_t prmt_12;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_12) : "r"(sf0[0]), "r"(0));
        uint32_t prmt_13;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_13) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_4;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_4) : "r"(d0[0]), "r"(prmt_12), "r"(2147516416u));
        d0[0] = _bf16x2_fma_4;
        uint32_t _bf16x2_fma_128_4;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_4) : "r"(d1[0]), "r"(prmt_13), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_4;
        uint32_t _bf16x2_fma_5;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_5) : "r"(d0[1]), "r"(prmt_12), "r"(2147516416u));
        d0[1] = _bf16x2_fma_5;
        uint32_t _bf16x2_fma_128_5;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_5) : "r"(d1[1]), "r"(prmt_13), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_5;
        uint32_t _bf16x2_fma_6;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_6) : "r"(d0[2]), "r"(prmt_12), "r"(2147516416u));
        d0[2] = _bf16x2_fma_6;
        uint32_t _bf16x2_fma_128_6;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_6) : "r"(d1[2]), "r"(prmt_13), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_6;
        uint32_t _bf16x2_fma_7;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_7) : "r"(d0[3]), "r"(prmt_12), "r"(2147516416u));
        d0[3] = _bf16x2_fma_7;
        uint32_t _bf16x2_fma_128_7;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_7) : "r"(d1[3]), "r"(prmt_13), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_7;
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
        uint32_t prmt_14;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_14) : "r"(raw0[2] & 2004318071), "r"(raw0[2] & 2004318071));
        uint32_t prmt_1_4;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_4) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_4) : "r"(2071689984), "r"(2088533116), "r"(raw0[2] & 2004318071));
#endif
        uint32_t prmt_2_3;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_3) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_14));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_3) : "r"(2071689984), "r"(2088533116), "r"(prmt_14));
#endif
        uint32_t prmt_3_3;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_3) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_3) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#endif
        uint32_t prmt_4_3;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_3) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_14));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_3) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_14));
#endif
        uint32_t prmt_5_3;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_3) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t prmt_6_3;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_3) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_39) : "r"(prmt_3_3), "r"(prmt_1_4 | prmt_5_3 & 2155905152u));
        d0[0] = _prmt_39;
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_40) : "r"(prmt_3_3), "r"(prmt_1_4 | prmt_5_3 & 2155905152u));
        d0[1] = _prmt_40;
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_41) : "r"(prmt_4_3), "r"(prmt_2_3 | prmt_6_3 & 2155905152u));
        d0[2] = _prmt_41;
        uint32_t _prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_42) : "r"(prmt_4_3), "r"(prmt_2_3 | prmt_6_3 & 2155905152u));
        d0[3] = _prmt_42;
        uint32_t prmt_15;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_15) : "r"(raw1[2] & 2004318071), "r"(raw1[2] & 2004318071));
        uint32_t prmt_1_5;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_5) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_5) : "r"(2071689984), "r"(2088533116), "r"(raw1[2] & 2004318071));
#endif
        uint32_t prmt_2_4;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_4) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_15));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_4) : "r"(2071689984), "r"(2088533116), "r"(prmt_15));
#endif
        uint32_t prmt_3_4;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_4) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_4) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#endif
        uint32_t prmt_4_4;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_4) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_15));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_4) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_15));
#endif
        uint32_t prmt_5_4;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_4) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t prmt_6_4;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_4) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_43;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_43) : "r"(prmt_3_4), "r"(prmt_1_5 | prmt_5_4 & 2155905152u));
        d1[0] = _prmt_43;
        uint32_t _prmt_44;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_44) : "r"(prmt_3_4), "r"(prmt_1_5 | prmt_5_4 & 2155905152u));
        d1[1] = _prmt_44;
        uint32_t _prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_45) : "r"(prmt_4_4), "r"(prmt_2_4 | prmt_6_4 & 2155905152u));
        d1[2] = _prmt_45;
        uint32_t _prmt_46;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_46) : "r"(prmt_4_4), "r"(prmt_2_4 | prmt_6_4 & 2155905152u));
        d1[3] = _prmt_46;
        uint32_t prmt_16;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_16) : "r"(sf0[0]), "r"(0));
        uint32_t prmt_17;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_17) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_8;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_8) : "r"(d0[0]), "r"(prmt_16), "r"(2147516416u));
        d0[0] = _bf16x2_fma_8;
        uint32_t _bf16x2_fma_128_8;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_8) : "r"(d1[0]), "r"(prmt_17), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_8;
        uint32_t _bf16x2_fma_9;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_9) : "r"(d0[1]), "r"(prmt_16), "r"(2147516416u));
        d0[1] = _bf16x2_fma_9;
        uint32_t _bf16x2_fma_128_9;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_9) : "r"(d1[1]), "r"(prmt_17), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_9;
        uint32_t _bf16x2_fma_10;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_10) : "r"(d0[2]), "r"(prmt_16), "r"(2147516416u));
        d0[2] = _bf16x2_fma_10;
        uint32_t _bf16x2_fma_128_10;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_10) : "r"(d1[2]), "r"(prmt_17), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_10;
        uint32_t _bf16x2_fma_11;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_11) : "r"(d0[3]), "r"(prmt_16), "r"(2147516416u));
        d0[3] = _bf16x2_fma_11;
        uint32_t _bf16x2_fma_128_11;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_11) : "r"(d1[3]), "r"(prmt_17), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_11;
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
        uint32_t prmt_18;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_18) : "r"(raw0[3] & 2004318071), "r"(raw0[3] & 2004318071));
        uint32_t prmt_1_6;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_6) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_6) : "r"(2071689984), "r"(2088533116), "r"(raw0[3] & 2004318071));
#endif
        uint32_t prmt_2_5;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_5) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_18));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_5) : "r"(2071689984), "r"(2088533116), "r"(prmt_18));
#endif
        uint32_t prmt_3_5;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_5) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_5) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#endif
        uint32_t prmt_4_5;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_5) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_18));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_5) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_18));
#endif
        uint32_t prmt_5_5;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_5) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t prmt_6_5;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_5) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_47;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_47) : "r"(prmt_3_5), "r"(prmt_1_6 | prmt_5_5 & 2155905152u));
        d0[0] = _prmt_47;
        uint32_t _prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_48) : "r"(prmt_3_5), "r"(prmt_1_6 | prmt_5_5 & 2155905152u));
        d0[1] = _prmt_48;
        uint32_t _prmt_49;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_49) : "r"(prmt_4_5), "r"(prmt_2_5 | prmt_6_5 & 2155905152u));
        d0[2] = _prmt_49;
        uint32_t _prmt_50;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_50) : "r"(prmt_4_5), "r"(prmt_2_5 | prmt_6_5 & 2155905152u));
        d0[3] = _prmt_50;
        uint32_t prmt_19;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_19) : "r"(raw1[3] & 2004318071), "r"(raw1[3] & 2004318071));
        uint32_t prmt_1_7;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_7) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_7) : "r"(2071689984), "r"(2088533116), "r"(raw1[3] & 2004318071));
#endif
        uint32_t prmt_2_6;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_6) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_19));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_6) : "r"(2071689984), "r"(2088533116), "r"(prmt_19));
#endif
        uint32_t prmt_3_6;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_6) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_6) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#endif
        uint32_t prmt_4_6;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_6) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_19));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_6) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_19));
#endif
        uint32_t prmt_5_6;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_6) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t prmt_6_6;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_6) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_51;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_51) : "r"(prmt_3_6), "r"(prmt_1_7 | prmt_5_6 & 2155905152u));
        d1[0] = _prmt_51;
        uint32_t _prmt_52;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_52) : "r"(prmt_3_6), "r"(prmt_1_7 | prmt_5_6 & 2155905152u));
        d1[1] = _prmt_52;
        uint32_t _prmt_53;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_53) : "r"(prmt_4_6), "r"(prmt_2_6 | prmt_6_6 & 2155905152u));
        d1[2] = _prmt_53;
        uint32_t _prmt_54;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_54) : "r"(prmt_4_6), "r"(prmt_2_6 | prmt_6_6 & 2155905152u));
        d1[3] = _prmt_54;
        uint32_t prmt_20;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_20) : "r"(sf0[0]), "r"(0));
        uint32_t prmt_21;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_21) : "r"(sf1[0]), "r"(0));
        uint32_t _bf16x2_fma_12;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_12) : "r"(d0[0]), "r"(prmt_20), "r"(2147516416u));
        d0[0] = _bf16x2_fma_12;
        uint32_t _bf16x2_fma_128_12;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_12) : "r"(d1[0]), "r"(prmt_21), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_12;
        uint32_t _bf16x2_fma_13;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_13) : "r"(d0[1]), "r"(prmt_20), "r"(2147516416u));
        d0[1] = _bf16x2_fma_13;
        uint32_t _bf16x2_fma_128_13;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_13) : "r"(d1[1]), "r"(prmt_21), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_13;
        uint32_t _bf16x2_fma_14;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_14) : "r"(d0[2]), "r"(prmt_20), "r"(2147516416u));
        d0[2] = _bf16x2_fma_14;
        uint32_t _bf16x2_fma_128_14;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_14) : "r"(d1[2]), "r"(prmt_21), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_14;
        uint32_t _bf16x2_fma_15;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_15) : "r"(d0[3]), "r"(prmt_20), "r"(2147516416u));
        d0[3] = _bf16x2_fma_15;
        uint32_t _bf16x2_fma_128_15;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_15) : "r"(d1[3]), "r"(prmt_21), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_15;
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
        unsigned int data_col_1_1 = (unsigned int)(32 * t + 16);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw0[(0) + 3]))
            : "r"(k0_addr + data_col_1_1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw1[(0) + 3]))
            : "r"(k1_addr + data_col_1_1));
        uint32_t prmt_22;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_22) : "r"(raw0[0] & 2004318071), "r"(raw0[0] & 2004318071));
        uint32_t prmt_1_8;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_8) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_8) : "r"(2071689984), "r"(2088533116), "r"(raw0[0] & 2004318071));
#endif
        uint32_t prmt_2_7;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_7) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_22));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_7) : "r"(2071689984), "r"(2088533116), "r"(prmt_22));
#endif
        uint32_t prmt_3_7;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_7) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_7) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[0] & 2004318071));
#endif
        uint32_t prmt_4_7;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_7) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_22));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_7) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_22));
#endif
        uint32_t prmt_5_7;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_7) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t prmt_6_7;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_7) : "r"(raw0[0] << 4), "r"(raw0[0]));
        uint32_t _prmt_55;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_55) : "r"(prmt_3_7), "r"(prmt_1_8 | prmt_5_7 & 2155905152u));
        d0[0] = _prmt_55;
        uint32_t _prmt_56;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_56) : "r"(prmt_3_7), "r"(prmt_1_8 | prmt_5_7 & 2155905152u));
        d0[1] = _prmt_56;
        uint32_t _prmt_57;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_57) : "r"(prmt_4_7), "r"(prmt_2_7 | prmt_6_7 & 2155905152u));
        d0[2] = _prmt_57;
        uint32_t _prmt_58;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_58) : "r"(prmt_4_7), "r"(prmt_2_7 | prmt_6_7 & 2155905152u));
        d0[3] = _prmt_58;
        uint32_t prmt_23;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_23) : "r"(raw1[0] & 2004318071), "r"(raw1[0] & 2004318071));
        uint32_t prmt_1_9;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_9) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_9) : "r"(2071689984), "r"(2088533116), "r"(raw1[0] & 2004318071));
#endif
        uint32_t prmt_2_8;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_8) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_23));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_8) : "r"(2071689984), "r"(2088533116), "r"(prmt_23));
#endif
        uint32_t prmt_3_8;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_8) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_8) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[0] & 2004318071));
#endif
        uint32_t prmt_4_8;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_8) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_23));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_8) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_23));
#endif
        uint32_t prmt_5_8;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_8) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t prmt_6_8;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_8) : "r"(raw1[0] << 4), "r"(raw1[0]));
        uint32_t _prmt_59;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_59) : "r"(prmt_3_8), "r"(prmt_1_9 | prmt_5_8 & 2155905152u));
        d1[0] = _prmt_59;
        uint32_t _prmt_60;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_60) : "r"(prmt_3_8), "r"(prmt_1_9 | prmt_5_8 & 2155905152u));
        d1[1] = _prmt_60;
        uint32_t _prmt_61;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_61) : "r"(prmt_4_8), "r"(prmt_2_8 | prmt_6_8 & 2155905152u));
        d1[2] = _prmt_61;
        uint32_t _prmt_62;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_62) : "r"(prmt_4_8), "r"(prmt_2_8 | prmt_6_8 & 2155905152u));
        d1[3] = _prmt_62;
        uint32_t prmt_24;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_24) : "r"(sf0[1]), "r"(0));
        uint32_t prmt_25;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_25) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_16;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_16) : "r"(d0[0]), "r"(prmt_24), "r"(2147516416u));
        d0[0] = _bf16x2_fma_16;
        uint32_t _bf16x2_fma_128_16;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_16) : "r"(d1[0]), "r"(prmt_25), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_16;
        uint32_t _bf16x2_fma_17;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_17) : "r"(d0[1]), "r"(prmt_24), "r"(2147516416u));
        d0[1] = _bf16x2_fma_17;
        uint32_t _bf16x2_fma_128_17;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_17) : "r"(d1[1]), "r"(prmt_25), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_17;
        uint32_t _bf16x2_fma_18;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_18) : "r"(d0[2]), "r"(prmt_24), "r"(2147516416u));
        d0[2] = _bf16x2_fma_18;
        uint32_t _bf16x2_fma_128_18;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_18) : "r"(d1[2]), "r"(prmt_25), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_18;
        uint32_t _bf16x2_fma_19;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_19) : "r"(d0[3]), "r"(prmt_24), "r"(2147516416u));
        d0[3] = _bf16x2_fma_19;
        uint32_t _bf16x2_fma_128_19;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_19) : "r"(d1[3]), "r"(prmt_25), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_19;
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
        uint32_t prmt_26;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_26) : "r"(raw0[1] & 2004318071), "r"(raw0[1] & 2004318071));
        uint32_t prmt_1_10;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_10) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_10) : "r"(2071689984), "r"(2088533116), "r"(raw0[1] & 2004318071));
#endif
        uint32_t prmt_2_9;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_9) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_26));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_9) : "r"(2071689984), "r"(2088533116), "r"(prmt_26));
#endif
        uint32_t prmt_3_9;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_9) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_9) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[1] & 2004318071));
#endif
        uint32_t prmt_4_9;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_9) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_26));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_9) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_26));
#endif
        uint32_t prmt_5_9;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_9) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t prmt_6_9;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_9) : "r"(raw0[1] << 4), "r"(raw0[1]));
        uint32_t _prmt_63;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_63) : "r"(prmt_3_9), "r"(prmt_1_10 | prmt_5_9 & 2155905152u));
        d0[0] = _prmt_63;
        uint32_t _prmt_64;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_64) : "r"(prmt_3_9), "r"(prmt_1_10 | prmt_5_9 & 2155905152u));
        d0[1] = _prmt_64;
        uint32_t _prmt_65;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_65) : "r"(prmt_4_9), "r"(prmt_2_9 | prmt_6_9 & 2155905152u));
        d0[2] = _prmt_65;
        uint32_t _prmt_66;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_66) : "r"(prmt_4_9), "r"(prmt_2_9 | prmt_6_9 & 2155905152u));
        d0[3] = _prmt_66;
        uint32_t prmt_27;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_27) : "r"(raw1[1] & 2004318071), "r"(raw1[1] & 2004318071));
        uint32_t prmt_1_11;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_11) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_11) : "r"(2071689984), "r"(2088533116), "r"(raw1[1] & 2004318071));
#endif
        uint32_t prmt_2_10;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_10) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_27));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_10) : "r"(2071689984), "r"(2088533116), "r"(prmt_27));
#endif
        uint32_t prmt_3_10;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_10) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_10) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[1] & 2004318071));
#endif
        uint32_t prmt_4_10;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_10) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_27));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_10) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_27));
#endif
        uint32_t prmt_5_10;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_10) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t prmt_6_10;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_10) : "r"(raw1[1] << 4), "r"(raw1[1]));
        uint32_t _prmt_67;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_67) : "r"(prmt_3_10), "r"(prmt_1_11 | prmt_5_10 & 2155905152u));
        d1[0] = _prmt_67;
        uint32_t _prmt_68;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_68) : "r"(prmt_3_10), "r"(prmt_1_11 | prmt_5_10 & 2155905152u));
        d1[1] = _prmt_68;
        uint32_t _prmt_69;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_69) : "r"(prmt_4_10), "r"(prmt_2_10 | prmt_6_10 & 2155905152u));
        d1[2] = _prmt_69;
        uint32_t _prmt_70;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_70) : "r"(prmt_4_10), "r"(prmt_2_10 | prmt_6_10 & 2155905152u));
        d1[3] = _prmt_70;
        uint32_t prmt_28;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_28) : "r"(sf0[1]), "r"(0));
        uint32_t prmt_29;
        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(prmt_29) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_20;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_20) : "r"(d0[0]), "r"(prmt_28), "r"(2147516416u));
        d0[0] = _bf16x2_fma_20;
        uint32_t _bf16x2_fma_128_20;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_20) : "r"(d1[0]), "r"(prmt_29), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_20;
        uint32_t _bf16x2_fma_21;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_21) : "r"(d0[1]), "r"(prmt_28), "r"(2147516416u));
        d0[1] = _bf16x2_fma_21;
        uint32_t _bf16x2_fma_128_21;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_21) : "r"(d1[1]), "r"(prmt_29), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_21;
        uint32_t _bf16x2_fma_22;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_22) : "r"(d0[2]), "r"(prmt_28), "r"(2147516416u));
        d0[2] = _bf16x2_fma_22;
        uint32_t _bf16x2_fma_128_22;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_22) : "r"(d1[2]), "r"(prmt_29), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_22;
        uint32_t _bf16x2_fma_23;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_23) : "r"(d0[3]), "r"(prmt_28), "r"(2147516416u));
        d0[3] = _bf16x2_fma_23;
        uint32_t _bf16x2_fma_128_23;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_23) : "r"(d1[3]), "r"(prmt_29), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_23;
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
        uint32_t prmt_30;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_30) : "r"(raw0[2] & 2004318071), "r"(raw0[2] & 2004318071));
        uint32_t prmt_1_12;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_12) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_12) : "r"(2071689984), "r"(2088533116), "r"(raw0[2] & 2004318071));
#endif
        uint32_t prmt_2_11;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_11) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_30));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_11) : "r"(2071689984), "r"(2088533116), "r"(prmt_30));
#endif
        uint32_t prmt_3_11;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_11) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_11) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[2] & 2004318071));
#endif
        uint32_t prmt_4_11;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_11) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_30));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_11) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_30));
#endif
        uint32_t prmt_5_11;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_11) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t prmt_6_11;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_11) : "r"(raw0[2] << 4), "r"(raw0[2]));
        uint32_t _prmt_71;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_71) : "r"(prmt_3_11), "r"(prmt_1_12 | prmt_5_11 & 2155905152u));
        d0[0] = _prmt_71;
        uint32_t _prmt_72;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_72) : "r"(prmt_3_11), "r"(prmt_1_12 | prmt_5_11 & 2155905152u));
        d0[1] = _prmt_72;
        uint32_t _prmt_73;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_73) : "r"(prmt_4_11), "r"(prmt_2_11 | prmt_6_11 & 2155905152u));
        d0[2] = _prmt_73;
        uint32_t _prmt_74;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_74) : "r"(prmt_4_11), "r"(prmt_2_11 | prmt_6_11 & 2155905152u));
        d0[3] = _prmt_74;
        uint32_t prmt_31;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_31) : "r"(raw1[2] & 2004318071), "r"(raw1[2] & 2004318071));
        uint32_t prmt_1_13;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_13) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_13) : "r"(2071689984), "r"(2088533116), "r"(raw1[2] & 2004318071));
#endif
        uint32_t prmt_2_12;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_12) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_31));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_12) : "r"(2071689984), "r"(2088533116), "r"(prmt_31));
#endif
        uint32_t prmt_3_12;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_12) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_12) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[2] & 2004318071));
#endif
        uint32_t prmt_4_12;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_12) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_31));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_12) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_31));
#endif
        uint32_t prmt_5_12;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_12) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t prmt_6_12;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_12) : "r"(raw1[2] << 4), "r"(raw1[2]));
        uint32_t _prmt_75;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_75) : "r"(prmt_3_12), "r"(prmt_1_13 | prmt_5_12 & 2155905152u));
        d1[0] = _prmt_75;
        uint32_t _prmt_76;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_76) : "r"(prmt_3_12), "r"(prmt_1_13 | prmt_5_12 & 2155905152u));
        d1[1] = _prmt_76;
        uint32_t _prmt_77;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_77) : "r"(prmt_4_12), "r"(prmt_2_12 | prmt_6_12 & 2155905152u));
        d1[2] = _prmt_77;
        uint32_t _prmt_78;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_78) : "r"(prmt_4_12), "r"(prmt_2_12 | prmt_6_12 & 2155905152u));
        d1[3] = _prmt_78;
        uint32_t prmt_32;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_32) : "r"(sf0[1]), "r"(0));
        uint32_t prmt_33;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_33) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_24;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_24) : "r"(d0[0]), "r"(prmt_32), "r"(2147516416u));
        d0[0] = _bf16x2_fma_24;
        uint32_t _bf16x2_fma_128_24;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_24) : "r"(d1[0]), "r"(prmt_33), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_24;
        uint32_t _bf16x2_fma_25;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_25) : "r"(d0[1]), "r"(prmt_32), "r"(2147516416u));
        d0[1] = _bf16x2_fma_25;
        uint32_t _bf16x2_fma_128_25;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_25) : "r"(d1[1]), "r"(prmt_33), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_25;
        uint32_t _bf16x2_fma_26;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_26) : "r"(d0[2]), "r"(prmt_32), "r"(2147516416u));
        d0[2] = _bf16x2_fma_26;
        uint32_t _bf16x2_fma_128_26;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_26) : "r"(d1[2]), "r"(prmt_33), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_26;
        uint32_t _bf16x2_fma_27;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_27) : "r"(d0[3]), "r"(prmt_32), "r"(2147516416u));
        d0[3] = _bf16x2_fma_27;
        uint32_t _bf16x2_fma_128_27;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_27) : "r"(d1[3]), "r"(prmt_33), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_27;
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
        uint32_t prmt_34;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_34) : "r"(raw0[3] & 2004318071), "r"(raw0[3] & 2004318071));
        uint32_t prmt_1_14;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_14) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_14) : "r"(2071689984), "r"(2088533116), "r"(raw0[3] & 2004318071));
#endif
        uint32_t prmt_2_13;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_13) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_34));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_13) : "r"(2071689984), "r"(2088533116), "r"(prmt_34));
#endif
        uint32_t prmt_3_13;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_13) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_13) : "r"(3229614080u), "r"(3229630464u), "r"(raw0[3] & 2004318071));
#endif
        uint32_t prmt_4_13;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_13) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_34));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_13) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_34));
#endif
        uint32_t prmt_5_13;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_13) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t prmt_6_13;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_13) : "r"(raw0[3] << 4), "r"(raw0[3]));
        uint32_t _prmt_79;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_79) : "r"(prmt_3_13), "r"(prmt_1_14 | prmt_5_13 & 2155905152u));
        d0[0] = _prmt_79;
        uint32_t _prmt_80;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_80) : "r"(prmt_3_13), "r"(prmt_1_14 | prmt_5_13 & 2155905152u));
        d0[1] = _prmt_80;
        uint32_t _prmt_81;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_81) : "r"(prmt_4_13), "r"(prmt_2_13 | prmt_6_13 & 2155905152u));
        d0[2] = _prmt_81;
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_82) : "r"(prmt_4_13), "r"(prmt_2_13 | prmt_6_13 & 2155905152u));
        d0[3] = _prmt_82;
        uint32_t prmt_35;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_35) : "r"(raw1[3] & 2004318071), "r"(raw1[3] & 2004318071));
        uint32_t prmt_1_15;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_15) : "r"(lut_regs[0]), "r"(2088533116), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_15) : "r"(2071689984), "r"(2088533116), "r"(raw1[3] & 2004318071));
#endif
        uint32_t prmt_2_14;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_14) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_35));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_14) : "r"(2071689984), "r"(2088533116), "r"(prmt_35));
#endif
        uint32_t prmt_3_14;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_14) : "r"(lut_regs[1]), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_14) : "r"(3229614080u), "r"(3229630464u), "r"(raw1[3] & 2004318071));
#endif
        uint32_t prmt_4_14;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_14) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_35));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_14) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_35));
#endif
        uint32_t prmt_5_14;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_14) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t prmt_6_14;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_14) : "r"(raw1[3] << 4), "r"(raw1[3]));
        uint32_t _prmt_83;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_83) : "r"(prmt_3_14), "r"(prmt_1_15 | prmt_5_14 & 2155905152u));
        d1[0] = _prmt_83;
        uint32_t _prmt_84;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_84) : "r"(prmt_3_14), "r"(prmt_1_15 | prmt_5_14 & 2155905152u));
        d1[1] = _prmt_84;
        uint32_t _prmt_85;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(_prmt_85) : "r"(prmt_4_14), "r"(prmt_2_14 | prmt_6_14 & 2155905152u));
        d1[2] = _prmt_85;
        uint32_t _prmt_86;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(_prmt_86) : "r"(prmt_4_14), "r"(prmt_2_14 | prmt_6_14 & 2155905152u));
        d1[3] = _prmt_86;
        uint32_t prmt_36;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_36) : "r"(sf0[1]), "r"(0));
        uint32_t prmt_37;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_37) : "r"(sf1[1]), "r"(0));
        uint32_t _bf16x2_fma_28;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_28) : "r"(d0[0]), "r"(prmt_36), "r"(2147516416u));
        d0[0] = _bf16x2_fma_28;
        uint32_t _bf16x2_fma_128_28;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_28) : "r"(d1[0]), "r"(prmt_37), "r"(2147516416u));
        d1[0] = _bf16x2_fma_128_28;
        uint32_t _bf16x2_fma_29;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_29) : "r"(d0[1]), "r"(prmt_36), "r"(2147516416u));
        d0[1] = _bf16x2_fma_29;
        uint32_t _bf16x2_fma_128_29;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_29) : "r"(d1[1]), "r"(prmt_37), "r"(2147516416u));
        d1[1] = _bf16x2_fma_128_29;
        uint32_t _bf16x2_fma_30;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_30) : "r"(d0[2]), "r"(prmt_36), "r"(2147516416u));
        d0[2] = _bf16x2_fma_30;
        uint32_t _bf16x2_fma_128_30;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_30) : "r"(d1[2]), "r"(prmt_37), "r"(2147516416u));
        d1[2] = _bf16x2_fma_128_30;
        uint32_t _bf16x2_fma_31;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_31) : "r"(d0[3]), "r"(prmt_36), "r"(2147516416u));
        d0[3] = _bf16x2_fma_31;
        uint32_t _bf16x2_fma_128_31;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_31) : "r"(d1[3]), "r"(prmt_37), "r"(2147516416u));
        d1[3] = _bf16x2_fma_128_31;
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
            unsigned int src_size_3_0 = (v_bits[0] & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1), "l"(v_src[0]), "r"(src_size_3_0), "l"(0x12F0000000000000ULL));
            unsigned int src_size_3_1_1 = (v_bits[0] >> 4 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1 + 576), "l"(v_src[1]), "r"(src_size_3_1_1), "l"(0x12F0000000000000ULL));
            unsigned int src_size_3_2 = (v_bits[0] >> 8 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1 + 1152), "l"(v_src[2]), "r"(src_size_3_2), "l"(0x12F0000000000000ULL));
            unsigned int src_size_3_3 = (v_bits[0] >> 12 & 1) << 4;
            asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                :: "r"(dst_v0_1 + 1728), "l"(v_src[3]), "r"(src_size_3_3), "l"(0x12F0000000000000ULL));
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
        float alpha_2_0 = approx_exp2(m[0] - m_new[0]);
        float tile_sum_1_0 = 0.0f;
        tile_sum_1_0 = tile_sum_1_0 + row_sum_s[g];
        tile_sum_1_0 = tile_sum_1_0 + row_sum_s[16 + g];
        tile_sum_1_0 = tile_sum_1_0 + row_sum_s[32 + g];
        tile_sum_1_0 = tile_sum_1_0 + row_sum_s[48 + g];
        l[0] = __fmaf_rn(l[0], alpha_2_0, tile_sum_1_0);
        m[0] = m_new[0];
        o[0] = o[0] * alpha_2_0;
        o[1] = o[1] * alpha_2_0;
        o[4] = o[4] * alpha_2_0;
        o[5] = o[5] * alpha_2_0;
        o[8] = o[8] * alpha_2_0;
        o[9] = o[9] * alpha_2_0;
        o[12] = o[12] * alpha_2_0;
        o[13] = o[13] * alpha_2_0;
        o[16] = o[16] * alpha_2_0;
        o[17] = o[17] * alpha_2_0;
        o[20] = o[20] * alpha_2_0;
        o[21] = o[21] * alpha_2_0;
        o[24] = o[24] * alpha_2_0;
        o[25] = o[25] * alpha_2_0;
        o[28] = o[28] * alpha_2_0;
        o[29] = o[29] * alpha_2_0;
        float alpha_2_1 = approx_exp2(m[1] - m_new[1]);
        float tile_sum_1_1 = 0.0f;
        tile_sum_1_1 = tile_sum_1_1 + row_sum_s[g + 8];
        tile_sum_1_1 = tile_sum_1_1 + row_sum_s[16 + g + 8];
        tile_sum_1_1 = tile_sum_1_1 + row_sum_s[32 + g + 8];
        tile_sum_1_1 = tile_sum_1_1 + row_sum_s[48 + g + 8];
        l[1] = __fmaf_rn(l[1], alpha_2_1, tile_sum_1_1);
        m[1] = m_new[1];
        o[2] = o[2] * alpha_2_1;
        o[3] = o[3] * alpha_2_1;
        o[6] = o[6] * alpha_2_1;
        o[7] = o[7] * alpha_2_1;
        o[10] = o[10] * alpha_2_1;
        o[11] = o[11] * alpha_2_1;
        o[14] = o[14] * alpha_2_1;
        o[15] = o[15] * alpha_2_1;
        o[18] = o[18] * alpha_2_1;
        o[19] = o[19] * alpha_2_1;
        o[22] = o[22] * alpha_2_1;
        o[23] = o[23] * alpha_2_1;
        o[26] = o[26] * alpha_2_1;
        o[27] = o[27] * alpha_2_1;
        o[30] = o[30] * alpha_2_1;
        o[31] = o[31] * alpha_2_1;
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
        uint32_t _prmt_0;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_0) : "r"(scale_byte), "r"(scale_byte_4));
        sfv[0] = _prmt_0 & 2147516416u | (_prmt_0 & 2130738944) >> 4;
        uint32_t _prmt_2;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_2) : "r"(scale_byte_6), "r"(scale_byte_8));
        sfv[1] = _prmt_2 & 2147516416u | (_prmt_2 & 2130738944) >> 4;
#else
        sf_packed = sf_packed | scale_byte_8 << 24;
        uint32_t prmt_38;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_38) : "r"(sf_packed), "r"(sf_packed));
        sfv[0] = prmt_38 & 2147516416u | (prmt_38 & 2130738944) >> 4;
        sfv[1] = prmt_38 << 8 & 2147516416u | (prmt_38 << 8 & 2130738944) >> 4;
#endif
        uint32_t w_lo;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi) : "r"(vw[0]), "r"(vw[1]));
#if __CUDA_ARCH__ == 900
        uint32_t prmt_38;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_38) : "r"(w_lo & 2004318071), "r"(w_lo & 2004318071));
#else
        uint32_t prmt_39;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_39) : "r"(w_lo & 2004318071), "r"(w_lo & 2004318071));
#endif
        uint32_t prmt_1_16;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_16) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_16) : "r"(2071689984), "r"(2088533116), "r"(w_lo & 2004318071));
#endif
        uint32_t prmt_2_15;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_15) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_38));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_15) : "r"(2071689984), "r"(2088533116), "r"(prmt_39));
#endif
        uint32_t prmt_3_15;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_15) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_15) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo & 2004318071));
#endif
        uint32_t prmt_4_15;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_15) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_38));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_15) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_39));
#endif
        uint32_t prmt_5_15;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_15) : "r"(w_lo << 4), "r"(w_lo));
        uint32_t prmt_6_15;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_15) : "r"(w_lo << 4), "r"(w_lo));
        uint32_t _prmt_103;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_103) : "r"(prmt_3_15), "r"(prmt_1_16 | prmt_5_15 & 2155905152u));
        dp_lo[0] = _prmt_103;
        uint32_t _prmt_104;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_104) : "r"(prmt_3_15), "r"(prmt_1_16 | prmt_5_15 & 2155905152u));
        dp_lo[1] = _prmt_104;
        uint32_t _prmt_105;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_105) : "r"(prmt_4_15), "r"(prmt_2_15 | prmt_6_15 & 2155905152u));
        dp_lo[2] = _prmt_105;
        uint32_t _prmt_106;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_106) : "r"(prmt_4_15), "r"(prmt_2_15 | prmt_6_15 & 2155905152u));
        dp_lo[3] = _prmt_106;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_39;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_39) : "r"(w_hi & 2004318071), "r"(w_hi & 2004318071));
#else
        uint32_t prmt_40;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_40) : "r"(w_hi & 2004318071), "r"(w_hi & 2004318071));
#endif
        uint32_t prmt_1_17;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_17) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_17) : "r"(2071689984), "r"(2088533116), "r"(w_hi & 2004318071));
#endif
        uint32_t prmt_2_16;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_16) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_39));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_16) : "r"(2071689984), "r"(2088533116), "r"(prmt_40));
#endif
        uint32_t prmt_3_16;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_16) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_16) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi & 2004318071));
#endif
        uint32_t prmt_4_16;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_16) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_39));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_16) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_40));
#endif
        uint32_t prmt_5_16;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_16) : "r"(w_hi << 4), "r"(w_hi));
        uint32_t prmt_6_16;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_16) : "r"(w_hi << 4), "r"(w_hi));
        uint32_t _prmt_107;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_107) : "r"(prmt_3_16), "r"(prmt_1_17 | prmt_5_16 & 2155905152u));
        dp_hi[0] = _prmt_107;
        uint32_t _prmt_108;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_108) : "r"(prmt_3_16), "r"(prmt_1_17 | prmt_5_16 & 2155905152u));
        dp_hi[1] = _prmt_108;
        uint32_t _prmt_109;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_109) : "r"(prmt_4_16), "r"(prmt_2_16 | prmt_6_16 & 2155905152u));
        dp_hi[2] = _prmt_109;
        uint32_t _prmt_110;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_110) : "r"(prmt_4_16), "r"(prmt_2_16 | prmt_6_16 & 2155905152u));
        dp_hi[3] = _prmt_110;
        uint32_t _bf16x2_fma_32;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_32) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_32;
        uint32_t _bf16x2_fma_128_32;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_32) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_128_32;
        uint32_t _bf16x2_fma_33;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_33) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_33;
        uint32_t _bf16x2_fma_128_33;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_33) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_128_33;
        uint32_t _bf16x2_fma_34;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_34) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_34;
        uint32_t _bf16x2_fma_128_34;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_34) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_128_34;
        uint32_t _bf16x2_fma_35;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_35) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_35;
        uint32_t _bf16x2_fma_128_35;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_35) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_128_35;
        uint32_t w_lo_9;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_9) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_10;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_10) : "r"(vw[2]), "r"(vw[3]));
#if __CUDA_ARCH__ == 900
        uint32_t prmt_40;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_40) : "r"(w_lo_9 & 2004318071), "r"(w_lo_9 & 2004318071));
#else
        uint32_t prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_41) : "r"(w_lo_9 & 2004318071), "r"(w_lo_9 & 2004318071));
#endif
        uint32_t prmt_1_18;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_18) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_9 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_18) : "r"(2071689984), "r"(2088533116), "r"(w_lo_9 & 2004318071));
#endif
        uint32_t prmt_2_17;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_17) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_40));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_17) : "r"(2071689984), "r"(2088533116), "r"(prmt_41));
#endif
        uint32_t prmt_3_17;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_17) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_9 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_17) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_9 & 2004318071));
#endif
        uint32_t prmt_4_17;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_17) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_40));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_17) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_41));
#endif
        uint32_t prmt_5_17;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_17) : "r"(w_lo_9 << 4), "r"(w_lo_9));
        uint32_t prmt_6_17;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_17) : "r"(w_lo_9 << 4), "r"(w_lo_9));
        uint32_t _prmt_111;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_111) : "r"(prmt_3_17), "r"(prmt_1_18 | prmt_5_17 & 2155905152u));
        dp_lo[4] = _prmt_111;
        uint32_t _prmt_112;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_112) : "r"(prmt_3_17), "r"(prmt_1_18 | prmt_5_17 & 2155905152u));
        dp_lo[5] = _prmt_112;
        uint32_t _prmt_113;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_113) : "r"(prmt_4_17), "r"(prmt_2_17 | prmt_6_17 & 2155905152u));
        dp_lo[6] = _prmt_113;
        uint32_t _prmt_114;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_114) : "r"(prmt_4_17), "r"(prmt_2_17 | prmt_6_17 & 2155905152u));
        dp_lo[7] = _prmt_114;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_41) : "r"(w_hi_10 & 2004318071), "r"(w_hi_10 & 2004318071));
#else
        uint32_t prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_42) : "r"(w_hi_10 & 2004318071), "r"(w_hi_10 & 2004318071));
#endif
        uint32_t prmt_1_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_19) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_10 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_19) : "r"(2071689984), "r"(2088533116), "r"(w_hi_10 & 2004318071));
#endif
        uint32_t prmt_2_18;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_18) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_41));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_18) : "r"(2071689984), "r"(2088533116), "r"(prmt_42));
#endif
        uint32_t prmt_3_18;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_18) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_10 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_18) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_10 & 2004318071));
#endif
        uint32_t prmt_4_18;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_18) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_41));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_18) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_42));
#endif
        uint32_t prmt_5_18;
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_18) : "r"(w_hi_10 << 4), "r"(w_hi_10));
        uint32_t prmt_6_18;
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_18) : "r"(w_hi_10 << 4), "r"(w_hi_10));
        uint32_t _prmt_115;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_115) : "r"(prmt_3_18), "r"(prmt_1_19 | prmt_5_18 & 2155905152u));
        dp_hi[4] = _prmt_115;
        uint32_t _prmt_116;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_116) : "r"(prmt_3_18), "r"(prmt_1_19 | prmt_5_18 & 2155905152u));
        dp_hi[5] = _prmt_116;
        uint32_t _prmt_117;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_117) : "r"(prmt_4_18), "r"(prmt_2_18 | prmt_6_18 & 2155905152u));
        dp_hi[6] = _prmt_117;
        uint32_t _prmt_118;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_118) : "r"(prmt_4_18), "r"(prmt_2_18 | prmt_6_18 & 2155905152u));
        dp_hi[7] = _prmt_118;
        uint32_t _bf16x2_fma_36;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_36) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_36;
        uint32_t _bf16x2_fma_128_36;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_36) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_128_36;
        uint32_t _bf16x2_fma_37;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_37) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_37;
        uint32_t _bf16x2_fma_128_37;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_37) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_128_37;
        uint32_t _bf16x2_fma_38;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_38) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_38;
        uint32_t _bf16x2_fma_128_38;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_38) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_128_38;
        uint32_t _bf16x2_fma_39;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_39) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_39;
        uint32_t _bf16x2_fma_128_39;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_39) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_128_39;
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
        uint32_t _prmt_3;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_3) : "r"(scale_byte_12), "r"(scale_byte_14));
        sfv[0] = _prmt_3 & 2147516416u | (_prmt_3 & 2130738944) >> 4;
        uint32_t _prmt_4;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_4) : "r"(scale_byte_16), "r"(scale_byte_18));
        sfv[1] = _prmt_4 & 2147516416u | (_prmt_4 & 2130738944) >> 4;
        uint32_t w_lo_19;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_19) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_20;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_20) : "r"(vw[0]), "r"(vw[1]));
        uint32_t prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_42) : "r"(w_lo_19 & 2004318071), "r"(w_lo_19 & 2004318071));
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
        uint32_t prmt_43;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_43) : "r"(sf_packed_11), "r"(sf_packed_11));
        sfv[0] = prmt_43 & 2147516416u | (prmt_43 & 2130738944) >> 4;
        sfv[1] = prmt_43 << 8 & 2147516416u | (prmt_43 << 8 & 2130738944) >> 4;
        uint32_t w_lo_20;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_20) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_21;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_21) : "r"(vw[0]), "r"(vw[1]));
        uint32_t prmt_44;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_44) : "r"(w_lo_20 & 2004318071), "r"(w_lo_20 & 2004318071));
#endif
        uint32_t prmt_1_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_20) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_19 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_20) : "r"(2071689984), "r"(2088533116), "r"(w_lo_20 & 2004318071));
#endif
        uint32_t prmt_2_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_19) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_42));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_19) : "r"(2071689984), "r"(2088533116), "r"(prmt_44));
#endif
        uint32_t prmt_3_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_19) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_19 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_19) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_20 & 2004318071));
#endif
        uint32_t prmt_4_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_19) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_42));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_19) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_44));
#endif
        uint32_t prmt_5_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_19) : "r"(w_lo_19 << 4), "r"(w_lo_19));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_19) : "r"(w_lo_20 << 4), "r"(w_lo_20));
#endif
        uint32_t prmt_6_19;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_19) : "r"(w_lo_19 << 4), "r"(w_lo_19));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_19) : "r"(w_lo_20 << 4), "r"(w_lo_20));
#endif
        uint32_t _prmt_119;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_119) : "r"(prmt_3_19), "r"(prmt_1_20 | prmt_5_19 & 2155905152u));
        dp_lo[0] = _prmt_119;
        uint32_t _prmt_120;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_120) : "r"(prmt_3_19), "r"(prmt_1_20 | prmt_5_19 & 2155905152u));
        dp_lo[1] = _prmt_120;
        uint32_t _prmt_121;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_121) : "r"(prmt_4_19), "r"(prmt_2_19 | prmt_6_19 & 2155905152u));
        dp_lo[2] = _prmt_121;
        uint32_t _prmt_122;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_122) : "r"(prmt_4_19), "r"(prmt_2_19 | prmt_6_19 & 2155905152u));
        dp_lo[3] = _prmt_122;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_43;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_43) : "r"(w_hi_20 & 2004318071), "r"(w_hi_20 & 2004318071));
#else
        uint32_t prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_45) : "r"(w_hi_21 & 2004318071), "r"(w_hi_21 & 2004318071));
#endif
        uint32_t prmt_1_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_21) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_20 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_21) : "r"(2071689984), "r"(2088533116), "r"(w_hi_21 & 2004318071));
#endif
        uint32_t prmt_2_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_20) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_43));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_20) : "r"(2071689984), "r"(2088533116), "r"(prmt_45));
#endif
        uint32_t prmt_3_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_20) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_20 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_20) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_21 & 2004318071));
#endif
        uint32_t prmt_4_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_20) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_43));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_20) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_45));
#endif
        uint32_t prmt_5_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_20) : "r"(w_hi_20 << 4), "r"(w_hi_20));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_20) : "r"(w_hi_21 << 4), "r"(w_hi_21));
#endif
        uint32_t prmt_6_20;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_20) : "r"(w_hi_20 << 4), "r"(w_hi_20));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_20) : "r"(w_hi_21 << 4), "r"(w_hi_21));
#endif
        uint32_t _prmt_123;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_123) : "r"(prmt_3_20), "r"(prmt_1_21 | prmt_5_20 & 2155905152u));
        dp_hi[0] = _prmt_123;
        uint32_t _prmt_124;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_124) : "r"(prmt_3_20), "r"(prmt_1_21 | prmt_5_20 & 2155905152u));
        dp_hi[1] = _prmt_124;
        uint32_t _prmt_125;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_125) : "r"(prmt_4_20), "r"(prmt_2_20 | prmt_6_20 & 2155905152u));
        dp_hi[2] = _prmt_125;
        uint32_t _prmt_126;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_126) : "r"(prmt_4_20), "r"(prmt_2_20 | prmt_6_20 & 2155905152u));
        dp_hi[3] = _prmt_126;
        uint32_t _bf16x2_fma_40;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_40) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_40;
        uint32_t _bf16x2_fma_128_40;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_40) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_128_40;
        uint32_t _bf16x2_fma_41;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_41) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_41;
        uint32_t _bf16x2_fma_128_41;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_41) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_128_41;
        uint32_t _bf16x2_fma_42;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_42) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_42;
        uint32_t _bf16x2_fma_128_42;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_42) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_128_42;
        uint32_t _bf16x2_fma_43;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_43) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_43;
        uint32_t _bf16x2_fma_128_43;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_43) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_128_43;
#if __CUDA_ARCH__ == 900
        uint32_t w_lo_21;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_21) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_22;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_22) : "r"(vw[2]), "r"(vw[3]));
        uint32_t prmt_44;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_44) : "r"(w_lo_21 & 2004318071), "r"(w_lo_21 & 2004318071));
#else
        uint32_t w_lo_22;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_22) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_23;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_23) : "r"(vw[2]), "r"(vw[3]));
        uint32_t prmt_46;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_46) : "r"(w_lo_22 & 2004318071), "r"(w_lo_22 & 2004318071));
#endif
        uint32_t prmt_1_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_22) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_21 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_22) : "r"(2071689984), "r"(2088533116), "r"(w_lo_22 & 2004318071));
#endif
        uint32_t prmt_2_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_21) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_44));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_21) : "r"(2071689984), "r"(2088533116), "r"(prmt_46));
#endif
        uint32_t prmt_3_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_21) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_21 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_21) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_22 & 2004318071));
#endif
        uint32_t prmt_4_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_21) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_44));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_21) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_46));
#endif
        uint32_t prmt_5_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_21) : "r"(w_lo_21 << 4), "r"(w_lo_21));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_21) : "r"(w_lo_22 << 4), "r"(w_lo_22));
#endif
        uint32_t prmt_6_21;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_21) : "r"(w_lo_21 << 4), "r"(w_lo_21));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_21) : "r"(w_lo_22 << 4), "r"(w_lo_22));
#endif
        uint32_t _prmt_127;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_127) : "r"(prmt_3_21), "r"(prmt_1_22 | prmt_5_21 & 2155905152u));
        dp_lo[4] = _prmt_127;
        uint32_t _prmt_128;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_128) : "r"(prmt_3_21), "r"(prmt_1_22 | prmt_5_21 & 2155905152u));
        dp_lo[5] = _prmt_128;
        uint32_t _prmt_129;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_129) : "r"(prmt_4_21), "r"(prmt_2_21 | prmt_6_21 & 2155905152u));
        dp_lo[6] = _prmt_129;
        uint32_t _prmt_130;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_130) : "r"(prmt_4_21), "r"(prmt_2_21 | prmt_6_21 & 2155905152u));
        dp_lo[7] = _prmt_130;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_45) : "r"(w_hi_22 & 2004318071), "r"(w_hi_22 & 2004318071));
#else
        uint32_t prmt_47;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_47) : "r"(w_hi_23 & 2004318071), "r"(w_hi_23 & 2004318071));
#endif
        uint32_t prmt_1_23;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_23) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_22 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_23) : "r"(2071689984), "r"(2088533116), "r"(w_hi_23 & 2004318071));
#endif
        uint32_t prmt_2_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_22) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_45));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_22) : "r"(2071689984), "r"(2088533116), "r"(prmt_47));
#endif
        uint32_t prmt_3_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_22) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_22 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_22) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_23 & 2004318071));
#endif
        uint32_t prmt_4_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_22) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_45));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_22) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_47));
#endif
        uint32_t prmt_5_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_22) : "r"(w_hi_22 << 4), "r"(w_hi_22));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_22) : "r"(w_hi_23 << 4), "r"(w_hi_23));
#endif
        uint32_t prmt_6_22;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_22) : "r"(w_hi_22 << 4), "r"(w_hi_22));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_22) : "r"(w_hi_23 << 4), "r"(w_hi_23));
#endif
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_131) : "r"(prmt_3_22), "r"(prmt_1_23 | prmt_5_22 & 2155905152u));
        dp_hi[4] = _prmt_131;
        uint32_t _prmt_132;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_132) : "r"(prmt_3_22), "r"(prmt_1_23 | prmt_5_22 & 2155905152u));
        dp_hi[5] = _prmt_132;
        uint32_t _prmt_133;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_133) : "r"(prmt_4_22), "r"(prmt_2_22 | prmt_6_22 & 2155905152u));
        dp_hi[6] = _prmt_133;
        uint32_t _prmt_134;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_134) : "r"(prmt_4_22), "r"(prmt_2_22 | prmt_6_22 & 2155905152u));
        dp_hi[7] = _prmt_134;
        uint32_t _bf16x2_fma_44;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_44) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_44;
        uint32_t _bf16x2_fma_128_44;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_44) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_128_44;
        uint32_t _bf16x2_fma_45;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_45) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_45;
        uint32_t _bf16x2_fma_128_45;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_45) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_128_45;
        uint32_t _bf16x2_fma_46;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_46) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_46;
        uint32_t _bf16x2_fma_128_46;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_46) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_128_46;
        uint32_t _bf16x2_fma_47;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_47) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_47;
        uint32_t _bf16x2_fma_128_47;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_47) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_128_47;
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
        uint32_t _prmt_5;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_5) : "r"(scale_byte_24), "r"(scale_byte_26));
        sfv[0] = _prmt_5 & 2147516416u | (_prmt_5 & 2130738944) >> 4;
        uint32_t _prmt_6;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_6) : "r"(scale_byte_28), "r"(scale_byte_30));
        sfv[1] = _prmt_6 & 2147516416u | (_prmt_6 & 2130738944) >> 4;
        uint32_t w_lo_31;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_31) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_32;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_32) : "r"(vw[0]), "r"(vw[1]));
        uint32_t prmt_46;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_46) : "r"(w_lo_31 & 2004318071), "r"(w_lo_31 & 2004318071));
#else
        sf_packed_24 = sf_packed_24 | scale_byte_30 << 16;
        int v_row_31 = v_stage + (32 + 2 * t + 1 + 8) * 144;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vw[3])) : "r"(kv_addr + (unsigned int)(v_row_31 + v_col)));
        unsigned int scale_byte_32 = (unsigned int)kv_u8[v_row_31 + v_scale_col];
        sf_packed_24 = sf_packed_24 | scale_byte_32 << 24;
        uint32_t prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_48) : "r"(sf_packed_24), "r"(sf_packed_24));
        sfv[0] = prmt_48 & 2147516416u | (prmt_48 & 2130738944) >> 4;
        sfv[1] = prmt_48 << 8 & 2147516416u | (prmt_48 << 8 & 2130738944) >> 4;
        uint32_t w_lo_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_33) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_34) : "r"(vw[0]), "r"(vw[1]));
        uint32_t prmt_49;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_49) : "r"(w_lo_33 & 2004318071), "r"(w_lo_33 & 2004318071));
#endif
        uint32_t prmt_1_24;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_24) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_31 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_24) : "r"(2071689984), "r"(2088533116), "r"(w_lo_33 & 2004318071));
#endif
        uint32_t prmt_2_23;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_23) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_46));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_23) : "r"(2071689984), "r"(2088533116), "r"(prmt_49));
#endif
        uint32_t prmt_3_23;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_23) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_31 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_23) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_33 & 2004318071));
#endif
        uint32_t prmt_4_23;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_23) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_46));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_23) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_49));
#endif
        uint32_t prmt_5_23;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_23) : "r"(w_lo_31 << 4), "r"(w_lo_31));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_23) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#endif
        uint32_t prmt_6_23;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_23) : "r"(w_lo_31 << 4), "r"(w_lo_31));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_23) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#endif
        uint32_t _prmt_135;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_135) : "r"(prmt_3_23), "r"(prmt_1_24 | prmt_5_23 & 2155905152u));
        dp_lo[0] = _prmt_135;
        uint32_t _prmt_136;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_136) : "r"(prmt_3_23), "r"(prmt_1_24 | prmt_5_23 & 2155905152u));
        dp_lo[1] = _prmt_136;
        uint32_t _prmt_137;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_137) : "r"(prmt_4_23), "r"(prmt_2_23 | prmt_6_23 & 2155905152u));
        dp_lo[2] = _prmt_137;
        uint32_t _prmt_138;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_138) : "r"(prmt_4_23), "r"(prmt_2_23 | prmt_6_23 & 2155905152u));
        dp_lo[3] = _prmt_138;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_47;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_47) : "r"(w_hi_32 & 2004318071), "r"(w_hi_32 & 2004318071));
#else
        uint32_t prmt_50;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_50) : "r"(w_hi_34 & 2004318071), "r"(w_hi_34 & 2004318071));
#endif
        uint32_t prmt_1_25;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_25) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_32 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_25) : "r"(2071689984), "r"(2088533116), "r"(w_hi_34 & 2004318071));
#endif
        uint32_t prmt_2_24;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_24) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_47));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_24) : "r"(2071689984), "r"(2088533116), "r"(prmt_50));
#endif
        uint32_t prmt_3_24;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_24) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_32 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_24) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_34 & 2004318071));
#endif
        uint32_t prmt_4_24;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_24) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_47));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_24) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_50));
#endif
        uint32_t prmt_5_24;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_24) : "r"(w_hi_32 << 4), "r"(w_hi_32));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_24) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#endif
        uint32_t prmt_6_24;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_24) : "r"(w_hi_32 << 4), "r"(w_hi_32));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_24) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#endif
        uint32_t _prmt_139;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_139) : "r"(prmt_3_24), "r"(prmt_1_25 | prmt_5_24 & 2155905152u));
        dp_hi[0] = _prmt_139;
        uint32_t _prmt_140;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_140) : "r"(prmt_3_24), "r"(prmt_1_25 | prmt_5_24 & 2155905152u));
        dp_hi[1] = _prmt_140;
        uint32_t _prmt_141;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_141) : "r"(prmt_4_24), "r"(prmt_2_24 | prmt_6_24 & 2155905152u));
        dp_hi[2] = _prmt_141;
        uint32_t _prmt_142;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_142) : "r"(prmt_4_24), "r"(prmt_2_24 | prmt_6_24 & 2155905152u));
        dp_hi[3] = _prmt_142;
        uint32_t _bf16x2_fma_48;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_48) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_48;
        uint32_t _bf16x2_fma_128_48;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_48) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_128_48;
        uint32_t _bf16x2_fma_49;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_49) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_49;
        uint32_t _bf16x2_fma_128_49;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_49) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_128_49;
        uint32_t _bf16x2_fma_50;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_50) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_50;
        uint32_t _bf16x2_fma_128_50;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_50) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_128_50;
        uint32_t _bf16x2_fma_51;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_51) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_51;
        uint32_t _bf16x2_fma_128_51;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_51) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_128_51;
#if __CUDA_ARCH__ == 900
        uint32_t w_lo_33;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_33) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_34;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_34) : "r"(vw[2]), "r"(vw[3]));
        uint32_t prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_48) : "r"(w_lo_33 & 2004318071), "r"(w_lo_33 & 2004318071));
#else
        uint32_t w_lo_35;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_35) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_36;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_36) : "r"(vw[2]), "r"(vw[3]));
        uint32_t prmt_51;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_51) : "r"(w_lo_35 & 2004318071), "r"(w_lo_35 & 2004318071));
#endif
        uint32_t prmt_1_26;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_26) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_33 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_26) : "r"(2071689984), "r"(2088533116), "r"(w_lo_35 & 2004318071));
#endif
        uint32_t prmt_2_25;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_25) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_48));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_25) : "r"(2071689984), "r"(2088533116), "r"(prmt_51));
#endif
        uint32_t prmt_3_25;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_25) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_33 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_25) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_35 & 2004318071));
#endif
        uint32_t prmt_4_25;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_25) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_48));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_25) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_51));
#endif
        uint32_t prmt_5_25;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_25) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_25) : "r"(w_lo_35 << 4), "r"(w_lo_35));
#endif
        uint32_t prmt_6_25;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_25) : "r"(w_lo_33 << 4), "r"(w_lo_33));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_25) : "r"(w_lo_35 << 4), "r"(w_lo_35));
#endif
        uint32_t _prmt_143;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_143) : "r"(prmt_3_25), "r"(prmt_1_26 | prmt_5_25 & 2155905152u));
        dp_lo[4] = _prmt_143;
        uint32_t _prmt_144;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_144) : "r"(prmt_3_25), "r"(prmt_1_26 | prmt_5_25 & 2155905152u));
        dp_lo[5] = _prmt_144;
        uint32_t _prmt_145;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_145) : "r"(prmt_4_25), "r"(prmt_2_25 | prmt_6_25 & 2155905152u));
        dp_lo[6] = _prmt_145;
        uint32_t _prmt_146;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_146) : "r"(prmt_4_25), "r"(prmt_2_25 | prmt_6_25 & 2155905152u));
        dp_lo[7] = _prmt_146;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_49;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_49) : "r"(w_hi_34 & 2004318071), "r"(w_hi_34 & 2004318071));
#else
        uint32_t prmt_52;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_52) : "r"(w_hi_36 & 2004318071), "r"(w_hi_36 & 2004318071));
#endif
        uint32_t prmt_1_27;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_27) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_34 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_27) : "r"(2071689984), "r"(2088533116), "r"(w_hi_36 & 2004318071));
#endif
        uint32_t prmt_2_26;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_26) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_49));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_26) : "r"(2071689984), "r"(2088533116), "r"(prmt_52));
#endif
        uint32_t prmt_3_26;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_26) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_34 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_26) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_36 & 2004318071));
#endif
        uint32_t prmt_4_26;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_26) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_49));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_26) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_52));
#endif
        uint32_t prmt_5_26;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_26) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_26) : "r"(w_hi_36 << 4), "r"(w_hi_36));
#endif
        uint32_t prmt_6_26;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_26) : "r"(w_hi_34 << 4), "r"(w_hi_34));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_26) : "r"(w_hi_36 << 4), "r"(w_hi_36));
#endif
        uint32_t _prmt_147;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_147) : "r"(prmt_3_26), "r"(prmt_1_27 | prmt_5_26 & 2155905152u));
        dp_hi[4] = _prmt_147;
        uint32_t _prmt_148;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_148) : "r"(prmt_3_26), "r"(prmt_1_27 | prmt_5_26 & 2155905152u));
        dp_hi[5] = _prmt_148;
        uint32_t _prmt_149;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_149) : "r"(prmt_4_26), "r"(prmt_2_26 | prmt_6_26 & 2155905152u));
        dp_hi[6] = _prmt_149;
        uint32_t _prmt_150;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_150) : "r"(prmt_4_26), "r"(prmt_2_26 | prmt_6_26 & 2155905152u));
        dp_hi[7] = _prmt_150;
        uint32_t _bf16x2_fma_52;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_52) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_52;
        uint32_t _bf16x2_fma_128_52;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_52) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_128_52;
        uint32_t _bf16x2_fma_53;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_53) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_53;
        uint32_t _bf16x2_fma_128_53;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_53) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_128_53;
        uint32_t _bf16x2_fma_54;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_54) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_54;
        uint32_t _bf16x2_fma_128_54;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_54) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_128_54;
        uint32_t _bf16x2_fma_55;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_55) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_55;
        uint32_t _bf16x2_fma_128_55;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_55) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_128_55;
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
        uint32_t _prmt_7;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_7) : "r"(scale_byte_36), "r"(scale_byte_38));
        sfv[0] = _prmt_7 & 2147516416u | (_prmt_7 & 2130738944) >> 4;
        uint32_t _prmt_8;
        asm("prmt.b32 %0, %1, %2, 0x4501;" : "=r"(_prmt_8) : "r"(scale_byte_40), "r"(scale_byte_42));
        sfv[1] = _prmt_8 & 2147516416u | (_prmt_8 & 2130738944) >> 4;
        uint32_t w_lo_43;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_43) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_44;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_44) : "r"(vw[0]), "r"(vw[1]));
        uint32_t prmt_50;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_50) : "r"(w_lo_43 & 2004318071), "r"(w_lo_43 & 2004318071));
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
        uint32_t prmt_53;
        asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_53) : "r"(sf_packed_37), "r"(sf_packed_37));
        sfv[0] = prmt_53 & 2147516416u | (prmt_53 & 2130738944) >> 4;
        sfv[1] = prmt_53 << 8 & 2147516416u | (prmt_53 << 8 & 2130738944) >> 4;
        uint32_t w_lo_46;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_46) : "r"(vw[0]), "r"(vw[1]));
        uint32_t w_hi_47;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_47) : "r"(vw[0]), "r"(vw[1]));
        uint32_t prmt_54;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_54) : "r"(w_lo_46 & 2004318071), "r"(w_lo_46 & 2004318071));
#endif
        uint32_t prmt_1_28;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_28) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_43 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_28) : "r"(2071689984), "r"(2088533116), "r"(w_lo_46 & 2004318071));
#endif
        uint32_t prmt_2_27;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_27) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_50));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_27) : "r"(2071689984), "r"(2088533116), "r"(prmt_54));
#endif
        uint32_t prmt_3_27;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_27) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_43 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_27) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_46 & 2004318071));
#endif
        uint32_t prmt_4_27;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_27) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_50));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_27) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_54));
#endif
        uint32_t prmt_5_27;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_27) : "r"(w_lo_43 << 4), "r"(w_lo_43));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_27) : "r"(w_lo_46 << 4), "r"(w_lo_46));
#endif
        uint32_t prmt_6_27;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_27) : "r"(w_lo_43 << 4), "r"(w_lo_43));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_27) : "r"(w_lo_46 << 4), "r"(w_lo_46));
#endif
        uint32_t _prmt_151;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_151) : "r"(prmt_3_27), "r"(prmt_1_28 | prmt_5_27 & 2155905152u));
        dp_lo[0] = _prmt_151;
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_152) : "r"(prmt_3_27), "r"(prmt_1_28 | prmt_5_27 & 2155905152u));
        dp_lo[1] = _prmt_152;
        uint32_t _prmt_153;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_153) : "r"(prmt_4_27), "r"(prmt_2_27 | prmt_6_27 & 2155905152u));
        dp_lo[2] = _prmt_153;
        uint32_t _prmt_154;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_154) : "r"(prmt_4_27), "r"(prmt_2_27 | prmt_6_27 & 2155905152u));
        dp_lo[3] = _prmt_154;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_51;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_51) : "r"(w_hi_44 & 2004318071), "r"(w_hi_44 & 2004318071));
#else
        uint32_t prmt_55;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_55) : "r"(w_hi_47 & 2004318071), "r"(w_hi_47 & 2004318071));
#endif
        uint32_t prmt_1_29;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_29) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_44 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_29) : "r"(2071689984), "r"(2088533116), "r"(w_hi_47 & 2004318071));
#endif
        uint32_t prmt_2_28;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_28) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_51));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_28) : "r"(2071689984), "r"(2088533116), "r"(prmt_55));
#endif
        uint32_t prmt_3_28;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_28) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_44 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_28) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_47 & 2004318071));
#endif
        uint32_t prmt_4_28;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_28) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_51));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_28) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_55));
#endif
        uint32_t prmt_5_28;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_28) : "r"(w_hi_44 << 4), "r"(w_hi_44));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_28) : "r"(w_hi_47 << 4), "r"(w_hi_47));
#endif
        uint32_t prmt_6_28;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_28) : "r"(w_hi_44 << 4), "r"(w_hi_44));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_28) : "r"(w_hi_47 << 4), "r"(w_hi_47));
#endif
        uint32_t _prmt_155;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_155) : "r"(prmt_3_28), "r"(prmt_1_29 | prmt_5_28 & 2155905152u));
        dp_hi[0] = _prmt_155;
        uint32_t _prmt_156;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_156) : "r"(prmt_3_28), "r"(prmt_1_29 | prmt_5_28 & 2155905152u));
        dp_hi[1] = _prmt_156;
        uint32_t _prmt_157;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_157) : "r"(prmt_4_28), "r"(prmt_2_28 | prmt_6_28 & 2155905152u));
        dp_hi[2] = _prmt_157;
        uint32_t _prmt_158;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_158) : "r"(prmt_4_28), "r"(prmt_2_28 | prmt_6_28 & 2155905152u));
        dp_hi[3] = _prmt_158;
        uint32_t _bf16x2_fma_56;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_56) : "r"(dp_lo[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[0] = _bf16x2_fma_56;
        uint32_t _bf16x2_fma_128_56;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_56) : "r"(dp_hi[0]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[0] = _bf16x2_fma_128_56;
        uint32_t _bf16x2_fma_57;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_57) : "r"(dp_lo[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[1] = _bf16x2_fma_57;
        uint32_t _bf16x2_fma_128_57;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_57) : "r"(dp_hi[1]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[1] = _bf16x2_fma_128_57;
        uint32_t _bf16x2_fma_58;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_58) : "r"(dp_lo[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[2] = _bf16x2_fma_58;
        uint32_t _bf16x2_fma_128_58;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_58) : "r"(dp_hi[2]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[2] = _bf16x2_fma_128_58;
        uint32_t _bf16x2_fma_59;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_59) : "r"(dp_lo[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_lo[3] = _bf16x2_fma_59;
        uint32_t _bf16x2_fma_128_59;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_59) : "r"(dp_hi[3]), "r"(sfv[0]), "r"(2147516416u));
        dp_hi[3] = _bf16x2_fma_128_59;
#if __CUDA_ARCH__ == 900
        uint32_t w_lo_45;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_45) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_46;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_46) : "r"(vw[2]), "r"(vw[3]));
        uint32_t prmt_52;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_52) : "r"(w_lo_45 & 2004318071), "r"(w_lo_45 & 2004318071));
#else
        uint32_t w_lo_48;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w_lo_48) : "r"(vw[2]), "r"(vw[3]));
        uint32_t w_hi_49;
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w_hi_49) : "r"(vw[2]), "r"(vw[3]));
        uint32_t prmt_56;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_56) : "r"(w_lo_48 & 2004318071), "r"(w_lo_48 & 2004318071));
#endif
        uint32_t prmt_1_30;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_30) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_lo_45 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_30) : "r"(2071689984), "r"(2088533116), "r"(w_lo_48 & 2004318071));
#endif
        uint32_t prmt_2_29;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_29) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_52));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_29) : "r"(2071689984), "r"(2088533116), "r"(prmt_56));
#endif
        uint32_t prmt_3_29;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_29) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_lo_45 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_29) : "r"(3229614080u), "r"(3229630464u), "r"(w_lo_48 & 2004318071));
#endif
        uint32_t prmt_4_29;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_29) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_52));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_29) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_56));
#endif
        uint32_t prmt_5_29;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_29) : "r"(w_lo_45 << 4), "r"(w_lo_45));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_29) : "r"(w_lo_48 << 4), "r"(w_lo_48));
#endif
        uint32_t prmt_6_29;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_29) : "r"(w_lo_45 << 4), "r"(w_lo_45));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_29) : "r"(w_lo_48 << 4), "r"(w_lo_48));
#endif
        uint32_t _prmt_159;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_159) : "r"(prmt_3_29), "r"(prmt_1_30 | prmt_5_29 & 2155905152u));
        dp_lo[4] = _prmt_159;
        uint32_t _prmt_160;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_160) : "r"(prmt_3_29), "r"(prmt_1_30 | prmt_5_29 & 2155905152u));
        dp_lo[5] = _prmt_160;
        uint32_t _prmt_161;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_161) : "r"(prmt_4_29), "r"(prmt_2_29 | prmt_6_29 & 2155905152u));
        dp_lo[6] = _prmt_161;
        uint32_t _prmt_162;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_162) : "r"(prmt_4_29), "r"(prmt_2_29 | prmt_6_29 & 2155905152u));
        dp_lo[7] = _prmt_162;
#if __CUDA_ARCH__ == 900
        uint32_t prmt_53;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_53) : "r"(w_hi_46 & 2004318071), "r"(w_hi_46 & 2004318071));
#else
        uint32_t prmt_57;
        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(prmt_57) : "r"(w_hi_49 & 2004318071), "r"(w_hi_49 & 2004318071));
#endif
        uint32_t prmt_1_31;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_31) : "r"(lut_regs[0]), "r"(2088533116), "r"(w_hi_46 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_1_31) : "r"(2071689984), "r"(2088533116), "r"(w_hi_49 & 2004318071));
#endif
        uint32_t prmt_2_30;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_30) : "r"(lut_regs[0]), "r"(2088533116), "r"(prmt_53));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_2_30) : "r"(2071689984), "r"(2088533116), "r"(prmt_57));
#endif
        uint32_t prmt_3_30;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_30) : "r"(lut_regs[1]), "r"(3229630464u), "r"(w_hi_46 & 2004318071));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_3_30) : "r"(3229614080u), "r"(3229630464u), "r"(w_hi_49 & 2004318071));
#endif
        uint32_t prmt_4_30;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_30) : "r"(lut_regs[1]), "r"(3229630464u), "r"(prmt_53));
#else
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(prmt_4_30) : "r"(3229614080u), "r"(3229630464u), "r"(prmt_57));
#endif
        uint32_t prmt_5_30;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_30) : "r"(w_hi_46 << 4), "r"(w_hi_46));
#else
        asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(prmt_5_30) : "r"(w_hi_49 << 4), "r"(w_hi_49));
#endif
        uint32_t prmt_6_30;
#if __CUDA_ARCH__ == 900
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_30) : "r"(w_hi_46 << 4), "r"(w_hi_46));
#else
        asm("prmt.b32 %0, %1, %2, 0xfbea;" : "=r"(prmt_6_30) : "r"(w_hi_49 << 4), "r"(w_hi_49));
#endif
        uint32_t _prmt_163;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_163) : "r"(prmt_3_30), "r"(prmt_1_31 | prmt_5_30 & 2155905152u));
        dp_hi[4] = _prmt_163;
        uint32_t _prmt_164;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_164) : "r"(prmt_3_30), "r"(prmt_1_31 | prmt_5_30 & 2155905152u));
        dp_hi[5] = _prmt_164;
        uint32_t _prmt_165;
        asm("prmt.b32 %0, %1, %2, 0x6240;" : "=r"(_prmt_165) : "r"(prmt_4_30), "r"(prmt_2_30 | prmt_6_30 & 2155905152u));
        dp_hi[6] = _prmt_165;
        uint32_t _prmt_166;
        asm("prmt.b32 %0, %1, %2, 0x7351;" : "=r"(_prmt_166) : "r"(prmt_4_30), "r"(prmt_2_30 | prmt_6_30 & 2155905152u));
        dp_hi[7] = _prmt_166;
        uint32_t _bf16x2_fma_60;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_60) : "r"(dp_lo[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[4] = _bf16x2_fma_60;
        uint32_t _bf16x2_fma_128_60;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_60) : "r"(dp_hi[4]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[4] = _bf16x2_fma_128_60;
        uint32_t _bf16x2_fma_61;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_61) : "r"(dp_lo[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[5] = _bf16x2_fma_61;
        uint32_t _bf16x2_fma_128_61;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_61) : "r"(dp_hi[5]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[5] = _bf16x2_fma_128_61;
        uint32_t _bf16x2_fma_62;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_62) : "r"(dp_lo[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[6] = _bf16x2_fma_62;
        uint32_t _bf16x2_fma_128_62;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_62) : "r"(dp_hi[6]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[6] = _bf16x2_fma_128_62;
        uint32_t _bf16x2_fma_63;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_63) : "r"(dp_lo[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_lo[7] = _bf16x2_fma_63;
        uint32_t _bf16x2_fma_128_63;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_128_63) : "r"(dp_hi[7]), "r"(sfv[1]), "r"(2147516416u));
        dp_hi[7] = _bf16x2_fma_128_63;
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
            unsigned int w0_1_5 = (unsigned int)data_off_7;
            unsigned int w1_2_2 = (unsigned int)(data_off_7 >> 32);
            unsigned int w2_3_2 = (unsigned int)sf_off_5;
            unsigned int w3_4_2 = (unsigned int)(sf_off_5 >> 32);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(write_slot * 1024 + token * 16)), "r"(w0_1_5), "r"(w1_2_2), "r"(w2_3_2), "r"(w3_4_2) : "memory");
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
            float _vec_load_4[8];
            {
                const uint4* _vptr_169 = reinterpret_cast<const uint4*>(gate + gate_off + 0);
                uint4 _vld_169[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_169[_blk] = _vptr_169[_blk];
                    uint32_t* _vpairs_169 = reinterpret_cast<uint32_t*>(&_vld_169[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) << 16);
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_1_0 = (float)(__nv_bfloat16)v[0];
            float _exp2_170 = approx_exp2((-_vec_load_4[0]) * 1.442695f);
            float sigmoid_2_0 = approx_rcp(1.0f + _exp2_170);
            gated[0] = attn_1_0 * sigmoid_2_0;
            float attn_1_1 = (float)(__nv_bfloat16)v[1];
            float _exp2_171 = approx_exp2((-_vec_load_4[1]) * 1.442695f);
            float sigmoid_2_1 = approx_rcp(1.0f + _exp2_171);
            gated[1] = attn_1_1 * sigmoid_2_1;
            float attn_1_2 = (float)(__nv_bfloat16)v[2];
            float _exp2_172 = approx_exp2((-_vec_load_4[2]) * 1.442695f);
            float sigmoid_2_2 = approx_rcp(1.0f + _exp2_172);
            gated[2] = attn_1_2 * sigmoid_2_2;
            float attn_1_3 = (float)(__nv_bfloat16)v[3];
            float _exp2_173 = approx_exp2((-_vec_load_4[3]) * 1.442695f);
            float sigmoid_2_3 = approx_rcp(1.0f + _exp2_173);
            gated[3] = attn_1_3 * sigmoid_2_3;
            float attn_1_4 = (float)(__nv_bfloat16)v[4];
            float _exp2_174 = approx_exp2((-_vec_load_4[4]) * 1.442695f);
            float sigmoid_2_4 = approx_rcp(1.0f + _exp2_174);
            gated[4] = attn_1_4 * sigmoid_2_4;
            float attn_1_5 = (float)(__nv_bfloat16)v[5];
            float _exp2_175 = approx_exp2((-_vec_load_4[5]) * 1.442695f);
            float sigmoid_2_5 = approx_rcp(1.0f + _exp2_175);
            gated[5] = attn_1_5 * sigmoid_2_5;
            float attn_1_6 = (float)(__nv_bfloat16)v[6];
            float _exp2_176 = approx_exp2((-_vec_load_4[6]) * 1.442695f);
            float sigmoid_2_6 = approx_rcp(1.0f + _exp2_176);
            gated[6] = attn_1_6 * sigmoid_2_6;
            float attn_1_7 = (float)(__nv_bfloat16)v[7];
            float _exp2_177 = approx_exp2((-_vec_load_4[7]) * 1.442695f);
            float sigmoid_2_7 = approx_rcp(1.0f + _exp2_177);
            gated[7] = attn_1_7 * sigmoid_2_7;
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
            float _vec_load_5[8];
            {
                const uint4* _vptr_178 = reinterpret_cast<const uint4*>(gate + (gate_off + 8) + 0);
                uint4 _vld_178[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_178[_blk] = _vptr_178[_blk];
                    uint32_t* _vpairs_178 = reinterpret_cast<uint32_t*>(&_vld_178[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_178[_pair]) << 16);
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_178[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_1_8 = (float)(__nv_bfloat16)v[8];
            float _exp2_179 = approx_exp2((-_vec_load_5[0]) * 1.442695f);
            float sigmoid_2_8 = approx_rcp(1.0f + _exp2_179);
            gated[0] = attn_1_8 * sigmoid_2_8;
            float attn_1_9 = (float)(__nv_bfloat16)v[9];
            float _exp2_180 = approx_exp2((-_vec_load_5[1]) * 1.442695f);
            float sigmoid_2_9 = approx_rcp(1.0f + _exp2_180);
            gated[1] = attn_1_9 * sigmoid_2_9;
            float attn_1_10 = (float)(__nv_bfloat16)v[10];
            float _exp2_181 = approx_exp2((-_vec_load_5[2]) * 1.442695f);
            float sigmoid_2_10 = approx_rcp(1.0f + _exp2_181);
            gated[2] = attn_1_10 * sigmoid_2_10;
            float attn_1_11 = (float)(__nv_bfloat16)v[11];
            float _exp2_182 = approx_exp2((-_vec_load_5[3]) * 1.442695f);
            float sigmoid_2_11 = approx_rcp(1.0f + _exp2_182);
            gated[3] = attn_1_11 * sigmoid_2_11;
            float attn_1_12 = (float)(__nv_bfloat16)v[12];
            float _exp2_183 = approx_exp2((-_vec_load_5[4]) * 1.442695f);
            float sigmoid_2_12 = approx_rcp(1.0f + _exp2_183);
            gated[4] = attn_1_12 * sigmoid_2_12;
            float attn_1_13 = (float)(__nv_bfloat16)v[13];
            float _exp2_184 = approx_exp2((-_vec_load_5[5]) * 1.442695f);
            float sigmoid_2_13 = approx_rcp(1.0f + _exp2_184);
            gated[5] = attn_1_13 * sigmoid_2_13;
            float attn_1_14 = (float)(__nv_bfloat16)v[14];
            float _exp2_185 = approx_exp2((-_vec_load_5[6]) * 1.442695f);
            float sigmoid_2_14 = approx_rcp(1.0f + _exp2_185);
            gated[6] = attn_1_14 * sigmoid_2_14;
            float attn_1_15 = (float)(__nv_bfloat16)v[15];
            float _exp2_186 = approx_exp2((-_vec_load_5[7]) * 1.442695f);
            float sigmoid_2_15 = approx_rcp(1.0f + _exp2_186);
            gated[7] = attn_1_15 * sigmoid_2_15;
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
            float _vec_load_6[8];
            {
                const uint4* _vptr_189 = reinterpret_cast<const uint4*>(gate + gate_off_1 + 0);
                uint4 _vld_189[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_189[_blk] = _vptr_189[_blk];
                    uint32_t* _vpairs_189 = reinterpret_cast<uint32_t*>(&_vld_189[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_189[_pair]) << 16);
                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_189[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_1_16 = (float)(__nv_bfloat16)v[0];
            float _exp2_190 = approx_exp2((-_vec_load_6[0]) * 1.442695f);
            float sigmoid_2_16 = approx_rcp(1.0f + _exp2_190);
            gated[0] = attn_1_16 * sigmoid_2_16;
            float attn_1_17 = (float)(__nv_bfloat16)v[1];
            float _exp2_191 = approx_exp2((-_vec_load_6[1]) * 1.442695f);
            float sigmoid_2_17 = approx_rcp(1.0f + _exp2_191);
            gated[1] = attn_1_17 * sigmoid_2_17;
            float attn_1_18 = (float)(__nv_bfloat16)v[2];
            float _exp2_192 = approx_exp2((-_vec_load_6[2]) * 1.442695f);
            float sigmoid_2_18 = approx_rcp(1.0f + _exp2_192);
            gated[2] = attn_1_18 * sigmoid_2_18;
            float attn_1_19 = (float)(__nv_bfloat16)v[3];
            float _exp2_193 = approx_exp2((-_vec_load_6[3]) * 1.442695f);
            float sigmoid_2_19 = approx_rcp(1.0f + _exp2_193);
            gated[3] = attn_1_19 * sigmoid_2_19;
            float attn_1_20 = (float)(__nv_bfloat16)v[4];
            float _exp2_194 = approx_exp2((-_vec_load_6[4]) * 1.442695f);
            float sigmoid_2_20 = approx_rcp(1.0f + _exp2_194);
            gated[4] = attn_1_20 * sigmoid_2_20;
            float attn_1_21 = (float)(__nv_bfloat16)v[5];
            float _exp2_195 = approx_exp2((-_vec_load_6[5]) * 1.442695f);
            float sigmoid_2_21 = approx_rcp(1.0f + _exp2_195);
            gated[5] = attn_1_21 * sigmoid_2_21;
            float attn_1_22 = (float)(__nv_bfloat16)v[6];
            float _exp2_196 = approx_exp2((-_vec_load_6[6]) * 1.442695f);
            float sigmoid_2_22 = approx_rcp(1.0f + _exp2_196);
            gated[6] = attn_1_22 * sigmoid_2_22;
            float attn_1_23 = (float)(__nv_bfloat16)v[7];
            float _exp2_197 = approx_exp2((-_vec_load_6[7]) * 1.442695f);
            float sigmoid_2_23 = approx_rcp(1.0f + _exp2_197);
            gated[7] = attn_1_23 * sigmoid_2_23;
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
            float _vec_load_7[8];
            {
                const uint4* _vptr_198 = reinterpret_cast<const uint4*>(gate + (gate_off_1 + 8) + 0);
                uint4 _vld_198[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_198[_blk] = _vptr_198[_blk];
                    uint32_t* _vpairs_198 = reinterpret_cast<uint32_t*>(&_vld_198[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_198[_pair]) << 16);
                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_198[_pair]) & 0xffff0000u);
                    }
                }
            }
            float attn_1_24 = (float)(__nv_bfloat16)v[8];
            float _exp2_199 = approx_exp2((-_vec_load_7[0]) * 1.442695f);
            float sigmoid_2_24 = approx_rcp(1.0f + _exp2_199);
            gated[0] = attn_1_24 * sigmoid_2_24;
            float attn_1_25 = (float)(__nv_bfloat16)v[9];
            float _exp2_200 = approx_exp2((-_vec_load_7[1]) * 1.442695f);
            float sigmoid_2_25 = approx_rcp(1.0f + _exp2_200);
            gated[1] = attn_1_25 * sigmoid_2_25;
            float attn_1_26 = (float)(__nv_bfloat16)v[10];
            float _exp2_201 = approx_exp2((-_vec_load_7[2]) * 1.442695f);
            float sigmoid_2_26 = approx_rcp(1.0f + _exp2_201);
            gated[2] = attn_1_26 * sigmoid_2_26;
            float attn_1_27 = (float)(__nv_bfloat16)v[11];
            float _exp2_202 = approx_exp2((-_vec_load_7[3]) * 1.442695f);
            float sigmoid_2_27 = approx_rcp(1.0f + _exp2_202);
            gated[3] = attn_1_27 * sigmoid_2_27;
            float attn_1_28 = (float)(__nv_bfloat16)v[12];
            float _exp2_203 = approx_exp2((-_vec_load_7[4]) * 1.442695f);
            float sigmoid_2_28 = approx_rcp(1.0f + _exp2_203);
            gated[4] = attn_1_28 * sigmoid_2_28;
            float attn_1_29 = (float)(__nv_bfloat16)v[13];
            float _exp2_204 = approx_exp2((-_vec_load_7[5]) * 1.442695f);
            float sigmoid_2_29 = approx_rcp(1.0f + _exp2_204);
            gated[5] = attn_1_29 * sigmoid_2_29;
            float attn_1_30 = (float)(__nv_bfloat16)v[14];
            float _exp2_205 = approx_exp2((-_vec_load_7[6]) * 1.442695f);
            float sigmoid_2_30 = approx_rcp(1.0f + _exp2_205);
            gated[6] = attn_1_30 * sigmoid_2_30;
            float attn_1_31 = (float)(__nv_bfloat16)v[15];
            float _exp2_206 = approx_exp2((-_vec_load_7[7]) * 1.442695f);
            float sigmoid_2_31 = approx_rcp(1.0f + _exp2_206);
            gated[7] = attn_1_31 * sigmoid_2_31;
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
#if __CUDA_ARCH__ == 800
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
                            long long part_row_1_0 = (long long)base * split_pitch + row_head[0];
                            float _vec_load_8[2];
                            {
                                float2 _v2_215 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_0 * 256 + dim_1[0]) + 0);
                                _vec_load_8[0] = _v2_215.x;
                                _vec_load_8[0 + 1] = _v2_215.y;
                            }
                            p0[0] = _vec_load_8[0];
                            p1[0] = _vec_load_8[1];
#else
                bool staged = num_splits <= 35 && n_lse <= 240;
                if (staged) {
                    long long nqo_h7 = (long long)num_qo_heads;
                    if (end > p_first) {
                        int head_0 = p_first >> 7;
                        long long dim_1 = (long long)((p_first & 127) * 2);
                        long long row_head = row_i64 * nqo_h7 + (long long)(first_head + head_0);
                        unsigned long long src0 = (unsigned long long)partial_out + (unsigned long long)((row_head * 256 + dim_1) * 4);
                        unsigned long long pitch_b = (unsigned long long)(split_pitch * 1024);
                        unsigned int dst0 = kv_u8_addr + (unsigned int)(tid_0 * 8);
                        #pragma unroll 4
                        for (int s_0 = 0; s_0 < num_splits; s_0++) {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                :: "r"(dst0 + (unsigned int)(s_0 * 1024)), "l"(src0 + (unsigned long long)s_0 * pitch_b));
#endif
                        }
#if __CUDA_ARCH__ == 800
                        if (base + 1 < num_splits) {
                            long long part_row_1_1 = (long long)(base + 1) * split_pitch + row_head[0];
                            float _vec_load_9[2];
                            {
                                float2 _v2_216 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_1 * 256 + dim_1[0]) + 0);
                                _vec_load_9[0] = _v2_216.x;
                                _vec_load_9[0 + 1] = _v2_216.y;
                            }
                            p0[1] = _vec_load_9[0];
                            p1[1] = _vec_load_9[1];
#else
                        asm volatile("cp.async.commit_group;");
                    }
                    #pragma unroll 1
                    for (int i_2 = tid_0; i_2 < n_lse; i_2 += 128) {
                        int h_2 = i_2 / num_splits;
                        int s_3 = i_2 - h_2 * num_splits;
                        lse_tab2[i_2] = partial_lse[(long long)s_3 * split_pitch + row_base + (long long)(head_first + h_2)];
                    }
                    __syncthreads();
                    #pragma unroll 1
                    for (int h_3 = warp_1; h_3 < n_heads; h_3 += 4) {
                        float hmax_2 = -CUDART_INF_F;
                        #pragma unroll 1
                        for (int s_4 = lane_1; s_4 < num_splits; s_4 += 32) {
                            hmax_2 = fmaxf(hmax_2, lse_tab2[h_3 * num_splits + s_4]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        if (base + 2 < num_splits) {
                            long long part_row_1_2 = (long long)(base + 2) * split_pitch + row_head[0];
                            float _vec_load_10[2];
                            {
                                float2 _v2_217 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_2 * 256 + dim_1[0]) + 0);
                                _vec_load_10[0] = _v2_217.x;
                                _vec_load_10[0 + 1] = _v2_217.y;
                            }
                            p0[2] = _vec_load_10[0];
                            p1[2] = _vec_load_10[1];
                        }
                        if (base + 3 < num_splits) {
                            long long part_row_1_3 = (long long)(base + 3) * split_pitch + row_head[0];
                            float _vec_load_11[2];
                            {
                                float2 _v2_218 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_3 * 256 + dim_1[0]) + 0);
                                _vec_load_11[0] = _v2_218.x;
                                _vec_load_11[0 + 1] = _v2_218.y;
                            }
                            p0[3] = _vec_load_11[0];
                            p1[3] = _vec_load_11[1];
                        }
                        if (base + 4 < num_splits) {
                            long long part_row_1_4 = (long long)(base + 4) * split_pitch + row_head[0];
                            float _vec_load_13[2];
                            {
                                float2 _v2_219 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_4 * 256 + dim_1[0]) + 0);
                                _vec_load_13[0] = _v2_219.x;
                                _vec_load_13[0 + 1] = _v2_219.y;
                            }
                            p0[4] = _vec_load_13[0];
                            p1[4] = _vec_load_13[1];
                        }
                        if (base + 5 < num_splits) {
                            long long part_row_1_5 = (long long)(base + 5) * split_pitch + row_head[0];
                            float _vec_load_14[2];
                            {
                                float2 _v2_220 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_5 * 256 + dim_1[0]) + 0);
                                _vec_load_14[0] = _v2_220.x;
                                _vec_load_14[0 + 1] = _v2_220.y;
                            }
                            p0[5] = _vec_load_14[0];
                            p1[5] = _vec_load_14[1];
                        }
                        if (base + 6 < num_splits) {
                            long long part_row_1_6 = (long long)(base + 6) * split_pitch + row_head[0];
                            float _vec_load_15[2];
                            {
                                float2 _v2_221 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_6 * 256 + dim_1[0]) + 0);
                                _vec_load_15[0] = _v2_221.x;
                                _vec_load_15[0 + 1] = _v2_221.y;
                            }
                            p0[6] = _vec_load_15[0];
                            p1[6] = _vec_load_15[1];
                        }
                        if (base + 7 < num_splits) {
                            long long part_row_1_7 = (long long)(base + 7) * split_pitch + row_head[0];
                            float _vec_load_16[2];
                            {
                                float2 _v2_222 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_7 * 256 + dim_1[0]) + 0);
                                _vec_load_16[0] = _v2_222.x;
                                _vec_load_16[0 + 1] = _v2_222.y;
                            }
                            p0[7] = _vec_load_16[0];
                            p1[7] = _vec_load_16[1];
                        }
                        if (base + 8 < num_splits) {
                            long long part_row_1_8 = (long long)(base + 8) * split_pitch + row_head[0];
                            float _vec_load_17[2];
                            {
                                float2 _v2_223 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_8 * 256 + dim_1[0]) + 0);
                                _vec_load_17[0] = _v2_223.x;
                                _vec_load_17[0 + 1] = _v2_223.y;
                            }
                            p0[8] = _vec_load_17[0];
                            p1[8] = _vec_load_17[1];
                        }
                        if (base + 9 < num_splits) {
                            long long part_row_1_9 = (long long)(base + 9) * split_pitch + row_head[0];
                            float _vec_load_18[2];
                            {
                                float2 _v2_224 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_9 * 256 + dim_1[0]) + 0);
                                _vec_load_18[0] = _v2_224.x;
                                _vec_load_18[0 + 1] = _v2_224.y;
                            }
                            p0[9] = _vec_load_18[0];
                            p1[9] = _vec_load_18[1];
                        }
                        if (base + 10 < num_splits) {
                            long long part_row_1_10 = (long long)(base + 10) * split_pitch + row_head[0];
                            float _vec_load_19[2];
                            {
                                float2 _v2_225 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_10 * 256 + dim_1[0]) + 0);
                                _vec_load_19[0] = _v2_225.x;
                                _vec_load_19[0 + 1] = _v2_225.y;
                            }
                            p0[10] = _vec_load_19[0];
                            p1[10] = _vec_load_19[1];
                        }
                        if (base + 11 < num_splits) {
                            long long part_row_1_11 = (long long)(base + 11) * split_pitch + row_head[0];
                            float _vec_load_20[2];
                            {
                                float2 _v2_226 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_11 * 256 + dim_1[0]) + 0);
                                _vec_load_20[0] = _v2_226.x;
                                _vec_load_20[0 + 1] = _v2_226.y;
                            }
                            p0[11] = _vec_load_20[0];
                            p1[11] = _vec_load_20[1];
                        }
                        if (base + 12 < num_splits) {
                            long long part_row_1_12 = (long long)(base + 12) * split_pitch + row_head[0];
                            float _vec_load_21[2];
                            {
                                float2 _v2_227 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_12 * 256 + dim_1[0]) + 0);
                                _vec_load_21[0] = _v2_227.x;
                                _vec_load_21[0 + 1] = _v2_227.y;
                            }
                            p0[12] = _vec_load_21[0];
                            p1[12] = _vec_load_21[1];
                        }
                        if (base + 13 < num_splits) {
                            long long part_row_1_13 = (long long)(base + 13) * split_pitch + row_head[0];
                            float _vec_load_22[2];
                            {
                                float2 _v2_228 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_13 * 256 + dim_1[0]) + 0);
                                _vec_load_22[0] = _v2_228.x;
                                _vec_load_22[0 + 1] = _v2_228.y;
                            }
                            p0[13] = _vec_load_22[0];
                            p1[13] = _vec_load_22[1];
                        }
                        if (base + 14 < num_splits) {
                            long long part_row_1_14 = (long long)(base + 14) * split_pitch + row_head[0];
                            float _vec_load_23[2];
                            {
                                float2 _v2_229 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_14 * 256 + dim_1[0]) + 0);
                                _vec_load_23[0] = _v2_229.x;
                                _vec_load_23[0 + 1] = _v2_229.y;
                            }
                            p0[14] = _vec_load_23[0];
                            p1[14] = _vec_load_23[1];
                        }
                        if (base + 15 < num_splits) {
                            long long part_row_1_15 = (long long)(base + 15) * split_pitch + row_head[0];
                            float _vec_load_24[2];
                            {
                                float2 _v2_230 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_15 * 256 + dim_1[0]) + 0);
                                _vec_load_24[0] = _v2_230.x;
                                _vec_load_24[0 + 1] = _v2_230.y;
                            }
                            p0[15] = _vec_load_24[0];
                            p1[15] = _vec_load_24[1];
                        }
                        bool live_15_0 = base < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_0) {
                            float weight_1_0 = approx_exp2(lse_tab[tab[0] + base] - lse_max[0]);
                            den[0] = den[0] + weight_1_0;
                            acc0[0] = __fmaf_rn(weight_1_0, p0[0], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_0, p1[0], acc1[0]);
                        }
                        bool live_15_1 = base + 1 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_1) {
                            float weight_1_1 = approx_exp2(lse_tab[tab[0] + base + 1] - lse_max[0]);
                            den[0] = den[0] + weight_1_1;
                            acc0[0] = __fmaf_rn(weight_1_1, p0[1], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_1, p1[1], acc1[0]);
                        }
                        bool live_15_2 = base + 2 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_2) {
                            float weight_1_2 = approx_exp2(lse_tab[tab[0] + base + 2] - lse_max[0]);
                            den[0] = den[0] + weight_1_2;
                            acc0[0] = __fmaf_rn(weight_1_2, p0[2], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_2, p1[2], acc1[0]);
                        }
                        bool live_15_3 = base + 3 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_3) {
                            float weight_1_3 = approx_exp2(lse_tab[tab[0] + base + 3] - lse_max[0]);
                            den[0] = den[0] + weight_1_3;
                            acc0[0] = __fmaf_rn(weight_1_3, p0[3], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_3, p1[3], acc1[0]);
                        }
                        bool live_15_4 = base + 4 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_4) {
                            float weight_1_4 = approx_exp2(lse_tab[tab[0] + base + 4] - lse_max[0]);
                            den[0] = den[0] + weight_1_4;
                            acc0[0] = __fmaf_rn(weight_1_4, p0[4], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_4, p1[4], acc1[0]);
                        }
                        bool live_15_5 = base + 5 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_5) {
                            float weight_1_5 = approx_exp2(lse_tab[tab[0] + base + 5] - lse_max[0]);
                            den[0] = den[0] + weight_1_5;
                            acc0[0] = __fmaf_rn(weight_1_5, p0[5], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_5, p1[5], acc1[0]);
                        }
                        bool live_15_6 = base + 6 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_6) {
                            float weight_1_6 = approx_exp2(lse_tab[tab[0] + base + 6] - lse_max[0]);
                            den[0] = den[0] + weight_1_6;
                            acc0[0] = __fmaf_rn(weight_1_6, p0[6], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_6, p1[6], acc1[0]);
                        }
                        bool live_15_7 = base + 7 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_7) {
                            float weight_1_7 = approx_exp2(lse_tab[tab[0] + base + 7] - lse_max[0]);
                            den[0] = den[0] + weight_1_7;
                            acc0[0] = __fmaf_rn(weight_1_7, p0[7], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_7, p1[7], acc1[0]);
                        }
                        bool live_15_8 = base + 8 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_8) {
                            float weight_1_8 = approx_exp2(lse_tab[tab[0] + base + 8] - lse_max[0]);
                            den[0] = den[0] + weight_1_8;
                            acc0[0] = __fmaf_rn(weight_1_8, p0[8], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_8, p1[8], acc1[0]);
                        }
                        bool live_15_9 = base + 9 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_9) {
                            float weight_1_9 = approx_exp2(lse_tab[tab[0] + base + 9] - lse_max[0]);
                            den[0] = den[0] + weight_1_9;
                            acc0[0] = __fmaf_rn(weight_1_9, p0[9], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_9, p1[9], acc1[0]);
                        }
                        bool live_15_10 = base + 10 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_10) {
                            float weight_1_10 = approx_exp2(lse_tab[tab[0] + base + 10] - lse_max[0]);
                            den[0] = den[0] + weight_1_10;
                            acc0[0] = __fmaf_rn(weight_1_10, p0[10], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_10, p1[10], acc1[0]);
                        }
                        bool live_15_11 = base + 11 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_11) {
                            float weight_1_11 = approx_exp2(lse_tab[tab[0] + base + 11] - lse_max[0]);
                            den[0] = den[0] + weight_1_11;
                            acc0[0] = __fmaf_rn(weight_1_11, p0[11], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_11, p1[11], acc1[0]);
                        }
                        bool live_15_12 = base + 12 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_12) {
                            float weight_1_12 = approx_exp2(lse_tab[tab[0] + base + 12] - lse_max[0]);
                            den[0] = den[0] + weight_1_12;
                            acc0[0] = __fmaf_rn(weight_1_12, p0[12], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_12, p1[12], acc1[0]);
                        }
                        bool live_15_13 = base + 13 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_13) {
                            float weight_1_13 = approx_exp2(lse_tab[tab[0] + base + 13] - lse_max[0]);
                            den[0] = den[0] + weight_1_13;
                            acc0[0] = __fmaf_rn(weight_1_13, p0[13], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_13, p1[13], acc1[0]);
                        }
                        bool live_15_14 = base + 14 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_14) {
                            float weight_1_14 = approx_exp2(lse_tab[tab[0] + base + 14] - lse_max[0]);
                            den[0] = den[0] + weight_1_14;
                            acc0[0] = __fmaf_rn(weight_1_14, p0[14], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_14, p1[14], acc1[0]);
                        }
                        bool live_15_15 = base + 15 < num_splits && lse_max[0] > -CUDART_INF_F;
                        if (live_15_15) {
                            float weight_1_15 = approx_exp2(lse_tab[tab[0] + base + 15] - lse_max[0]);
                            den[0] = den[0] + weight_1_15;
                            acc0[0] = __fmaf_rn(weight_1_15, p0[15], acc0[0]);
                            acc1[0] = __fmaf_rn(weight_1_15, p1[15], acc1[0]);
#else
                        float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, hmax_2, 16);
                        hmax_2 = fmaxf(hmax_2, _shfl_xor_sync_209);
                        float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, hmax_2, 8);
                        hmax_2 = fmaxf(hmax_2, _shfl_xor_sync_210);
                        float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, hmax_2, 4);
                        hmax_2 = fmaxf(hmax_2, _shfl_xor_sync_211);
                        float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, hmax_2, 2);
                        hmax_2 = fmaxf(hmax_2, _shfl_xor_sync_212);
                        float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, hmax_2, 1);
                        hmax_2 = fmaxf(hmax_2, _shfl_xor_sync_213);
                        if (lane_1 == 0) {
                            max_tab[h_3] = hmax_2;
#endif
                        }
                    }
#if __CUDA_ARCH__ == 800
                    if (valid_0_4[0] != 0) {
#else
                    __syncthreads();
                    float pr[2];
                    float acc0[1];
                    float acc1[1];
                    float den[1];
                    float lse_max[1];
                    float gate0[1];
                    float gate1[1];
                    unsigned int word[1];
                    long long nqo = (long long)num_qo_heads;
                    #pragma unroll 1
                    for (int p = p_first; p < end; p += 128) {
                        if (p != p_first) {
                            int head_0_1 = p >> 7;
                            long long dim_1_4 = (long long)((p & 127) * 2);
                            long long row_head_1 = row_i64 * nqo + (long long)(first_head + head_0_1);
                            unsigned long long src0_1 = (unsigned long long)partial_out + (unsigned long long)((row_head_1 * 256 + dim_1_4) * 4);
                            unsigned long long pitch_b_1 = (unsigned long long)(split_pitch * 1024);
                            unsigned int dst0_1 = kv_u8_addr + (unsigned int)(tid_0 * 8);
                            #pragma unroll 4
                            for (int s_1 = 0; s_1 < num_splits; s_1++) {
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                    :: "r"(dst0_1 + (unsigned int)(s_1 * 1024)), "l"(src0_1 + (unsigned long long)s_1 * pitch_b_1));
                            }
                            asm volatile("cp.async.commit_group;");
                        }
                        int head_0_2 = p >> 7;
                        long long dim_1_5 = (long long)((p & 127) * 2);
                        long long qo_head_2 = (long long)(first_head + head_0_2);
                        long long row_head_2 = row_i64 * nqo + qo_head_2;
                        acc0[0] = 0.0f;
                        acc1[0] = 0.0f;
                        den[0] = 0.0f;
                        int tab = (head_0_2 - head_first) * num_splits;
                        lse_max[0] = max_tab[head_0_2 - head_first];
                        float _vec_load_g[2];
                        {
                            uint32_t _bf16x2_bits_214;
                            _bf16x2_bits_214 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2 * 256 + dim_1_5) + 0);
                            (&_vec_load_g[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_214) << 16);
                            (&_vec_load_g[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_214) & 0xffff0000u);
                        }
                        gate0[0] = _vec_load_g[0];
                        gate1[0] = _vec_load_g[1];
                        asm volatile("cp.async.wait_group 0;");
                        unsigned int slot0 = kv_u8_addr + (unsigned int)(tid_0 * 8);
                        if (lse_max[0] > -CUDART_INF_F) {
                            #pragma unroll 4
                            for (int s_2 = 0; s_2 < num_splits; s_2++) {
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&pr[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr[(0) + 1]))
                                    : "r"(slot0 + (unsigned int)(s_2 * 1024)));
                                float weight_s = approx_exp2(lse_tab2[tab + s_2] - lse_max[0]);
                                den[0] = den[0] + weight_s;
                                acc0[0] = __fmaf_rn(weight_s, pr[0], acc0[0]);
                                acc1[0] = __fmaf_rn(weight_s, pr[1], acc1[0]);
                            }
                        }
#endif
                        float inv_2 = 0.0f;
                        if (den[0] > 0.0f) {
#if __CUDA_ARCH__ == 800
                            float _rcp_231 = approx_rcp(den[0]);
                            inv_2 = _rcp_231;
#else
                            float _rcp_215 = approx_rcp(den[0]);
                            inv_2 = _rcp_215;
#endif
                        }
                        float attn0 = (float)(__nv_bfloat16)(acc0[0] * inv_2);
                        float attn1 = (float)(__nv_bfloat16)(acc1[0] * inv_2);
#if __CUDA_ARCH__ == 800
                        float _exp2_233 = approx_exp2((-gate0[0]) * 1.442695f);
                        float _rcp_232 = approx_rcp(1.0f + _exp2_233);
                        float result0 = attn0 * _rcp_232;
                        float _exp2_235 = approx_exp2((-gate1[0]) * 1.442695f);
                        float _rcp_234 = approx_rcp(1.0f + _exp2_235);
                        float result1 = attn1 * _rcp_234;
                        __nv_bfloat162 _bit_cast_src_236 = __float22bfloat162_rn(make_float2(result0, result1));
                        word[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_236)[0];
                        long long out_offset = row_i64 * out_stride_row + qo_head_2[0] * out_stride_head + dim_1[0];
#else
                        float _exp2_217 = approx_exp2((-gate0[0]) * 1.442695f);
                        float _rcp_216 = approx_rcp(1.0f + _exp2_217);
                        float result0 = attn0 * _rcp_216;
                        float _exp2_219 = approx_exp2((-gate1[0]) * 1.442695f);
                        float _rcp_218 = approx_rcp(1.0f + _exp2_219);
                        float result1 = attn1 * _rcp_218;
                        __nv_bfloat162 _bit_cast_src_220 = __float22bfloat162_rn(make_float2(result0, result1));
                        word[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_220)[0];
                        long long out_offset = row_i64 * out_stride_row + qo_head_2 * out_stride_head + dim_1_5;
#endif
                        *(reinterpret_cast<unsigned int*>(out + out_offset) + (0)) = word[0];
#if __CUDA_ARCH__ == 800
#else
                    }
                } else {
                    #pragma unroll 1
                    for (int i = tid_0; i < n_lse; i += 128) {
                        int h = i / num_splits;
                        int s_0_1 = i - h * num_splits;
                        lse_tab[i] = partial_lse[(long long)s_0_1 * split_pitch + row_base + (long long)(head_first + h)];
                    }
                    __syncthreads();
                    #pragma unroll 1
                    for (int h_1 = warp_1; h_1 < n_heads; h_1 += 4) {
                        float hmax = -CUDART_INF_F;
                        #pragma unroll 1
                        for (int s_1_1 = lane_1; s_1_1 < num_splits; s_1_1 += 32) {
                            hmax = fmaxf(hmax, lse_tab[h_1 * num_splits + s_1_1]);
                        }
                        float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, hmax, 16);
                        hmax = fmaxf(hmax, _shfl_xor_sync_221);
                        float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, hmax, 8);
                        hmax = fmaxf(hmax, _shfl_xor_sync_222);
                        float _shfl_xor_sync_223 = __shfl_xor_sync(0xFFFFFFFF, hmax, 4);
                        hmax = fmaxf(hmax, _shfl_xor_sync_223);
                        float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, hmax, 2);
                        hmax = fmaxf(hmax, _shfl_xor_sync_224);
                        float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, hmax, 1);
                        hmax = fmaxf(hmax, _shfl_xor_sync_225);
                        if (lane_1 == 0) {
                            max_tab[h_1] = hmax;
                        }
                    }
                    __syncthreads();
                    float lse[16];
                    float p0[16];
                    float p1[16];
                    float acc0_1[1];
                    float acc1_1[1];
                    float den_1[1];
                    float lse_max_1[1];
                    int valid_0_4[1];
                    int tab_1[1];
                    long long dim_1_6[1];
                    long long qo_head_3[1];
                    long long row_head_3[1];
                    float gate0_1[1];
                    float gate1_1[1];
                    unsigned int word_1[1];
                    long long nqo_1 = (long long)num_qo_heads;
                    #pragma unroll 1
                    for (int p_1 = p_first; p_1 < end; p_1 += 128) {
                        int pk = ((p_1) < (end - 1) ? (p_1) : (end - 1));
                        valid_0_4[0] = ((end > p_1) ? 1 : 0);
                        int head_0_3 = pk >> 7;
                        dim_1_6[0] = (long long)((pk & 127) * 2);
                        qo_head_3[0] = (long long)(first_head + head_0_3);
                        row_head_3[0] = row_i64 * nqo_1 + qo_head_3[0];
                        acc0_1[0] = 0.0f;
                        acc1_1[0] = 0.0f;
                        den_1[0] = 0.0f;
                        tab_1[0] = (head_0_3 - head_first) * num_splits;
                        lse_max_1[0] = max_tab[head_0_3 - head_first];
                        float _vec_load_12[2];
                        {
                            uint32_t _bf16x2_bits_226;
                            _bf16x2_bits_226 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[0] * 256 + dim_1_6[0]) + 0);
                            (&_vec_load_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_226) << 16);
                            (&_vec_load_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_226) & 0xffff0000u);
                        }
                        gate0_1[0] = _vec_load_12[0];
                        gate1_1[0] = _vec_load_12[1];
                        #pragma unroll 1
                        for (int base = 0; base < num_splits; base += 16) {
                            if (base < num_splits) {
                                long long part_row_1_0 = (long long)base * split_pitch + row_head_3[0];
                                float _vec_load_8[2];
                                {
                                    float2 _v2_227 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_0 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_8[0] = _v2_227.x;
                                    _vec_load_8[0 + 1] = _v2_227.y;
                                }
                                p0[0] = _vec_load_8[0];
                                p1[0] = _vec_load_8[1];
                            }
                            if (base + 1 < num_splits) {
                                long long part_row_1_1 = (long long)(base + 1) * split_pitch + row_head_3[0];
                                float _vec_load_9[2];
                                {
                                    float2 _v2_228 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_1 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_9[0] = _v2_228.x;
                                    _vec_load_9[0 + 1] = _v2_228.y;
                                }
                                p0[1] = _vec_load_9[0];
                                p1[1] = _vec_load_9[1];
                            }
                            if (base + 2 < num_splits) {
                                long long part_row_1_2 = (long long)(base + 2) * split_pitch + row_head_3[0];
                                float _vec_load_10[2];
                                {
                                    float2 _v2_229 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_2 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_10[0] = _v2_229.x;
                                    _vec_load_10[0 + 1] = _v2_229.y;
                                }
                                p0[2] = _vec_load_10[0];
                                p1[2] = _vec_load_10[1];
                            }
                            if (base + 3 < num_splits) {
                                long long part_row_1_3 = (long long)(base + 3) * split_pitch + row_head_3[0];
                                float _vec_load_11[2];
                                {
                                    float2 _v2_230 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_3 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_11[0] = _v2_230.x;
                                    _vec_load_11[0 + 1] = _v2_230.y;
                                }
                                p0[3] = _vec_load_11[0];
                                p1[3] = _vec_load_11[1];
                            }
                            if (base + 4 < num_splits) {
                                long long part_row_1_4 = (long long)(base + 4) * split_pitch + row_head_3[0];
                                float _vec_load_13[2];
                                {
                                    float2 _v2_231 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_4 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_13[0] = _v2_231.x;
                                    _vec_load_13[0 + 1] = _v2_231.y;
                                }
                                p0[4] = _vec_load_13[0];
                                p1[4] = _vec_load_13[1];
                            }
                            if (base + 5 < num_splits) {
                                long long part_row_1_5 = (long long)(base + 5) * split_pitch + row_head_3[0];
                                float _vec_load_14[2];
                                {
                                    float2 _v2_232 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_5 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_14[0] = _v2_232.x;
                                    _vec_load_14[0 + 1] = _v2_232.y;
                                }
                                p0[5] = _vec_load_14[0];
                                p1[5] = _vec_load_14[1];
                            }
                            if (base + 6 < num_splits) {
                                long long part_row_1_6 = (long long)(base + 6) * split_pitch + row_head_3[0];
                                float _vec_load_15[2];
                                {
                                    float2 _v2_233 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_6 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_15[0] = _v2_233.x;
                                    _vec_load_15[0 + 1] = _v2_233.y;
                                }
                                p0[6] = _vec_load_15[0];
                                p1[6] = _vec_load_15[1];
                            }
                            if (base + 7 < num_splits) {
                                long long part_row_1_7 = (long long)(base + 7) * split_pitch + row_head_3[0];
                                float _vec_load_16[2];
                                {
                                    float2 _v2_234 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_7 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_16[0] = _v2_234.x;
                                    _vec_load_16[0 + 1] = _v2_234.y;
                                }
                                p0[7] = _vec_load_16[0];
                                p1[7] = _vec_load_16[1];
                            }
                            if (base + 8 < num_splits) {
                                long long part_row_1_8 = (long long)(base + 8) * split_pitch + row_head_3[0];
                                float _vec_load_17[2];
                                {
                                    float2 _v2_235 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_8 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_17[0] = _v2_235.x;
                                    _vec_load_17[0 + 1] = _v2_235.y;
                                }
                                p0[8] = _vec_load_17[0];
                                p1[8] = _vec_load_17[1];
                            }
                            if (base + 9 < num_splits) {
                                long long part_row_1_9 = (long long)(base + 9) * split_pitch + row_head_3[0];
                                float _vec_load_18[2];
                                {
                                    float2 _v2_236 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_9 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_18[0] = _v2_236.x;
                                    _vec_load_18[0 + 1] = _v2_236.y;
                                }
                                p0[9] = _vec_load_18[0];
                                p1[9] = _vec_load_18[1];
                            }
                            if (base + 10 < num_splits) {
                                long long part_row_1_10 = (long long)(base + 10) * split_pitch + row_head_3[0];
                                float _vec_load_19[2];
                                {
                                    float2 _v2_237 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_10 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_19[0] = _v2_237.x;
                                    _vec_load_19[0 + 1] = _v2_237.y;
                                }
                                p0[10] = _vec_load_19[0];
                                p1[10] = _vec_load_19[1];
                            }
                            if (base + 11 < num_splits) {
                                long long part_row_1_11 = (long long)(base + 11) * split_pitch + row_head_3[0];
                                float _vec_load_20[2];
                                {
                                    float2 _v2_238 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_11 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_20[0] = _v2_238.x;
                                    _vec_load_20[0 + 1] = _v2_238.y;
                                }
                                p0[11] = _vec_load_20[0];
                                p1[11] = _vec_load_20[1];
                            }
                            if (base + 12 < num_splits) {
                                long long part_row_1_12 = (long long)(base + 12) * split_pitch + row_head_3[0];
                                float _vec_load_21[2];
                                {
                                    float2 _v2_239 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_12 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_21[0] = _v2_239.x;
                                    _vec_load_21[0 + 1] = _v2_239.y;
                                }
                                p0[12] = _vec_load_21[0];
                                p1[12] = _vec_load_21[1];
                            }
                            if (base + 13 < num_splits) {
                                long long part_row_1_13 = (long long)(base + 13) * split_pitch + row_head_3[0];
                                float _vec_load_22[2];
                                {
                                    float2 _v2_240 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_13 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_22[0] = _v2_240.x;
                                    _vec_load_22[0 + 1] = _v2_240.y;
                                }
                                p0[13] = _vec_load_22[0];
                                p1[13] = _vec_load_22[1];
                            }
                            if (base + 14 < num_splits) {
                                long long part_row_1_14 = (long long)(base + 14) * split_pitch + row_head_3[0];
                                float _vec_load_23[2];
                                {
                                    float2 _v2_241 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_14 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_23[0] = _v2_241.x;
                                    _vec_load_23[0 + 1] = _v2_241.y;
                                }
                                p0[14] = _vec_load_23[0];
                                p1[14] = _vec_load_23[1];
                            }
                            if (base + 15 < num_splits) {
                                long long part_row_1_15 = (long long)(base + 15) * split_pitch + row_head_3[0];
                                float _vec_load_24[2];
                                {
                                    float2 _v2_242 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_15 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_24[0] = _v2_242.x;
                                    _vec_load_24[0 + 1] = _v2_242.y;
                                }
                                p0[15] = _vec_load_24[0];
                                p1[15] = _vec_load_24[1];
                            }
                            bool live_15_0 = base < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_0) {
                                float weight_1_0 = approx_exp2(lse_tab[tab_1[0] + base] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_0;
                                acc0_1[0] = __fmaf_rn(weight_1_0, p0[0], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_0, p1[0], acc1_1[0]);
                            }
                            bool live_15_1 = base + 1 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_1) {
                                float weight_1_1 = approx_exp2(lse_tab[tab_1[0] + base + 1] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_1;
                                acc0_1[0] = __fmaf_rn(weight_1_1, p0[1], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_1, p1[1], acc1_1[0]);
                            }
                            bool live_15_2 = base + 2 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_2) {
                                float weight_1_2 = approx_exp2(lse_tab[tab_1[0] + base + 2] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_2;
                                acc0_1[0] = __fmaf_rn(weight_1_2, p0[2], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_2, p1[2], acc1_1[0]);
                            }
                            bool live_15_3 = base + 3 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_3) {
                                float weight_1_3 = approx_exp2(lse_tab[tab_1[0] + base + 3] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_3;
                                acc0_1[0] = __fmaf_rn(weight_1_3, p0[3], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_3, p1[3], acc1_1[0]);
                            }
                            bool live_15_4 = base + 4 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_4) {
                                float weight_1_4 = approx_exp2(lse_tab[tab_1[0] + base + 4] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_4;
                                acc0_1[0] = __fmaf_rn(weight_1_4, p0[4], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_4, p1[4], acc1_1[0]);
                            }
                            bool live_15_5 = base + 5 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_5) {
                                float weight_1_5 = approx_exp2(lse_tab[tab_1[0] + base + 5] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_5;
                                acc0_1[0] = __fmaf_rn(weight_1_5, p0[5], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_5, p1[5], acc1_1[0]);
                            }
                            bool live_15_6 = base + 6 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_6) {
                                float weight_1_6 = approx_exp2(lse_tab[tab_1[0] + base + 6] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_6;
                                acc0_1[0] = __fmaf_rn(weight_1_6, p0[6], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_6, p1[6], acc1_1[0]);
                            }
                            bool live_15_7 = base + 7 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_7) {
                                float weight_1_7 = approx_exp2(lse_tab[tab_1[0] + base + 7] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_7;
                                acc0_1[0] = __fmaf_rn(weight_1_7, p0[7], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_7, p1[7], acc1_1[0]);
                            }
                            bool live_15_8 = base + 8 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_8) {
                                float weight_1_8 = approx_exp2(lse_tab[tab_1[0] + base + 8] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_8;
                                acc0_1[0] = __fmaf_rn(weight_1_8, p0[8], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_8, p1[8], acc1_1[0]);
                            }
                            bool live_15_9 = base + 9 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_9) {
                                float weight_1_9 = approx_exp2(lse_tab[tab_1[0] + base + 9] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_9;
                                acc0_1[0] = __fmaf_rn(weight_1_9, p0[9], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_9, p1[9], acc1_1[0]);
                            }
                            bool live_15_10 = base + 10 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_10) {
                                float weight_1_10 = approx_exp2(lse_tab[tab_1[0] + base + 10] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_10;
                                acc0_1[0] = __fmaf_rn(weight_1_10, p0[10], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_10, p1[10], acc1_1[0]);
                            }
                            bool live_15_11 = base + 11 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_11) {
                                float weight_1_11 = approx_exp2(lse_tab[tab_1[0] + base + 11] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_11;
                                acc0_1[0] = __fmaf_rn(weight_1_11, p0[11], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_11, p1[11], acc1_1[0]);
                            }
                            bool live_15_12 = base + 12 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_12) {
                                float weight_1_12 = approx_exp2(lse_tab[tab_1[0] + base + 12] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_12;
                                acc0_1[0] = __fmaf_rn(weight_1_12, p0[12], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_12, p1[12], acc1_1[0]);
                            }
                            bool live_15_13 = base + 13 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_13) {
                                float weight_1_13 = approx_exp2(lse_tab[tab_1[0] + base + 13] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_13;
                                acc0_1[0] = __fmaf_rn(weight_1_13, p0[13], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_13, p1[13], acc1_1[0]);
                            }
                            bool live_15_14 = base + 14 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_14) {
                                float weight_1_14 = approx_exp2(lse_tab[tab_1[0] + base + 14] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_14;
                                acc0_1[0] = __fmaf_rn(weight_1_14, p0[14], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_14, p1[14], acc1_1[0]);
                            }
                            bool live_15_15 = base + 15 < num_splits && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_15) {
                                float weight_1_15 = approx_exp2(lse_tab[tab_1[0] + base + 15] - lse_max_1[0]);
                                den_1[0] = den_1[0] + weight_1_15;
                                acc0_1[0] = __fmaf_rn(weight_1_15, p0[15], acc0_1[0]);
                                acc1_1[0] = __fmaf_rn(weight_1_15, p1[15], acc1_1[0]);
                            }
                        }
                        if (valid_0_4[0] != 0) {
                            float inv_3 = 0.0f;
                            if (den_1[0] > 0.0f) {
                                float _rcp_243 = approx_rcp(den_1[0]);
                                inv_3 = _rcp_243;
                            }
                            float attn0_1 = (float)(__nv_bfloat16)(acc0_1[0] * inv_3);
                            float attn1_1 = (float)(__nv_bfloat16)(acc1_1[0] * inv_3);
                            float _exp2_245 = approx_exp2((-gate0_1[0]) * 1.442695f);
                            float _rcp_244 = approx_rcp(1.0f + _exp2_245);
                            float result0_1 = attn0_1 * _rcp_244;
                            float _exp2_247 = approx_exp2((-gate1_1[0]) * 1.442695f);
                            float _rcp_246 = approx_rcp(1.0f + _exp2_247);
                            float result1_1 = attn1_1 * _rcp_246;
                            __nv_bfloat162 _bit_cast_src_248 = __float22bfloat162_rn(make_float2(result0_1, result1_1));
                            word_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_248)[0];
                            long long out_offset_1 = row_i64 * out_stride_row + qo_head_3[0] * out_stride_head + dim_1_6[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_1) + (0)) = word_1[0];
                        }
#endif
                    }
                }
            } else if (num_splits > 8) {
                float lse_1[16];
                float p0_1[16];
                float p1_1[16];
#if __CUDA_ARCH__ == 800
                float acc0_1[1];
                float acc1_1[1];
                float den_1[1];
                float lse_max_1[1];
#else
                float acc0_2[1];
                float acc1_2[1];
                float den_2[1];
                float lse_max_2[1];
#endif
                int valid_0_5[1];
#if __CUDA_ARCH__ == 800
                int tab_1[1];
                long long dim_1_4[1];
                long long qo_head_3[1];
                long long row_head_1[1];
                float gate0_1[1];
                float gate1_1[1];
                unsigned int word_1[1];
                long long nqo_1 = (long long)num_qo_heads;
#else
                int tab_2[1];
                long long dim_1_7[1];
                long long qo_head_4[1];
                long long row_head_4[1];
                float gate0_2[1];
                float gate1_2[1];
                unsigned int word_2[1];
                long long nqo_2 = (long long)num_qo_heads;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 800
                for (int p_1 = p_first; p_1 < end; p_1 += 128) {
                    int pk_1 = ((p_1) < (end - 1) ? (p_1) : (end - 1));
                    valid_0_5[0] = ((end > p_1) ? 1 : 0);
                    int head_0_1 = pk_1 >> 7;
                    dim_1_4[0] = (long long)((pk_1 & 127) * 2);
                    qo_head_3[0] = (long long)(first_head + head_0_1);
                    row_head_1[0] = row_i64 * nqo_1 + qo_head_3[0];
                    acc0_1[0] = 0.0f;
                    acc1_1[0] = 0.0f;
                    den_1[0] = 0.0f;
                    lse_max_1[0] = -CUDART_INF_F;
#else
                for (int p_2 = p_first; p_2 < end; p_2 += 128) {
                    int pk_1 = ((p_2) < (end - 1) ? (p_2) : (end - 1));
                    valid_0_5[0] = ((end > p_2) ? 1 : 0);
                    int head_0_4 = pk_1 >> 7;
                    dim_1_7[0] = (long long)((pk_1 & 127) * 2);
                    qo_head_4[0] = (long long)(first_head + head_0_4);
                    row_head_4[0] = row_i64 * nqo_2 + qo_head_4[0];
                    acc0_2[0] = 0.0f;
                    acc1_2[0] = 0.0f;
                    den_2[0] = 0.0f;
                    lse_max_2[0] = -CUDART_INF_F;
#endif
                    float _vec_load_29[2];
                    {
#if __CUDA_ARCH__ == 800
                        uint32_t _bf16x2_bits_237;
                        _bf16x2_bits_237 = *reinterpret_cast<const uint32_t*>(gate + (row_head_1[0] * 256 + dim_1_4[0]) + 0);
                        (&_vec_load_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_237) << 16);
                        (&_vec_load_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_237) & 0xffff0000u);
#else
                        uint32_t _bf16x2_bits_249;
                        _bf16x2_bits_249 = *reinterpret_cast<const uint32_t*>(gate + (row_head_4[0] * 256 + dim_1_7[0]) + 0);
                        (&_vec_load_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_249) << 16);
                        (&_vec_load_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_249) & 0xffff0000u);
#endif
                    }
#if __CUDA_ARCH__ == 800
                    gate0_1[0] = _vec_load_29[0];
                    gate1_1[0] = _vec_load_29[1];
#else
                    gate0_2[0] = _vec_load_29[0];
                    gate1_2[0] = _vec_load_29[1];
#endif
                    #pragma unroll 1
                    for (int base_1 = 0; base_1 < num_splits; base_1 += 16) {
                        lse_1[0] = -CUDART_INF_F;
                        if (base_1 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_16 = (long long)base_1 * split_pitch + row_head_1[0];
#else
                            long long part_row_1_16 = (long long)base_1 * split_pitch + row_head_4[0];
#endif
                            lse_1[0] = partial_lse[part_row_1_16];
                            float _vec_load_25[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_238 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_16 * 256 + dim_1_4[0]) + 0);
                                _vec_load_25[0] = _v2_238.x;
                                _vec_load_25[0 + 1] = _v2_238.y;
#else
                                float2 _v2_250 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_16 * 256 + dim_1_7[0]) + 0);
                                _vec_load_25[0] = _v2_250.x;
                                _vec_load_25[0 + 1] = _v2_250.y;
#endif
                            }
                            p0_1[0] = _vec_load_25[0];
                            p1_1[0] = _vec_load_25[1];
                        }
                        lse_1[1] = -CUDART_INF_F;
                        if (base_1 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_17 = (long long)(base_1 + 1) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_17 = (long long)(base_1 + 1) * split_pitch + row_head_4[0];
#endif
                            lse_1[1] = partial_lse[part_row_1_17];
                            float _vec_load_26[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_239 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_17 * 256 + dim_1_4[0]) + 0);
                                _vec_load_26[0] = _v2_239.x;
                                _vec_load_26[0 + 1] = _v2_239.y;
#else
                                float2 _v2_251 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_17 * 256 + dim_1_7[0]) + 0);
                                _vec_load_26[0] = _v2_251.x;
                                _vec_load_26[0 + 1] = _v2_251.y;
#endif
                            }
                            p0_1[1] = _vec_load_26[0];
                            p1_1[1] = _vec_load_26[1];
                        }
                        lse_1[2] = -CUDART_INF_F;
                        if (base_1 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_18 = (long long)(base_1 + 2) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_18 = (long long)(base_1 + 2) * split_pitch + row_head_4[0];
#endif
                            lse_1[2] = partial_lse[part_row_1_18];
                            float _vec_load_27[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_240 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_18 * 256 + dim_1_4[0]) + 0);
                                _vec_load_27[0] = _v2_240.x;
                                _vec_load_27[0 + 1] = _v2_240.y;
#else
                                float2 _v2_252 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_18 * 256 + dim_1_7[0]) + 0);
                                _vec_load_27[0] = _v2_252.x;
                                _vec_load_27[0 + 1] = _v2_252.y;
#endif
                            }
                            p0_1[2] = _vec_load_27[0];
                            p1_1[2] = _vec_load_27[1];
                        }
                        lse_1[3] = -CUDART_INF_F;
                        if (base_1 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_19 = (long long)(base_1 + 3) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_19 = (long long)(base_1 + 3) * split_pitch + row_head_4[0];
#endif
                            lse_1[3] = partial_lse[part_row_1_19];
                            float _vec_load_28[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_241 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_19 * 256 + dim_1_4[0]) + 0);
                                _vec_load_28[0] = _v2_241.x;
                                _vec_load_28[0 + 1] = _v2_241.y;
#else
                                float2 _v2_253 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_19 * 256 + dim_1_7[0]) + 0);
                                _vec_load_28[0] = _v2_253.x;
                                _vec_load_28[0 + 1] = _v2_253.y;
#endif
                            }
                            p0_1[3] = _vec_load_28[0];
                            p1_1[3] = _vec_load_28[1];
                        }
                        lse_1[4] = -CUDART_INF_F;
                        if (base_1 + 4 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_20 = (long long)(base_1 + 4) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_20 = (long long)(base_1 + 4) * split_pitch + row_head_4[0];
#endif
                            lse_1[4] = partial_lse[part_row_1_20];
                            float _vec_load_30[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_242 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_20 * 256 + dim_1_4[0]) + 0);
                                _vec_load_30[0] = _v2_242.x;
                                _vec_load_30[0 + 1] = _v2_242.y;
#else
                                float2 _v2_254 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_20 * 256 + dim_1_7[0]) + 0);
                                _vec_load_30[0] = _v2_254.x;
                                _vec_load_30[0 + 1] = _v2_254.y;
#endif
                            }
                            p0_1[4] = _vec_load_30[0];
                            p1_1[4] = _vec_load_30[1];
                        }
                        lse_1[5] = -CUDART_INF_F;
                        if (base_1 + 5 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_21 = (long long)(base_1 + 5) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_21 = (long long)(base_1 + 5) * split_pitch + row_head_4[0];
#endif
                            lse_1[5] = partial_lse[part_row_1_21];
                            float _vec_load_31[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_243 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_21 * 256 + dim_1_4[0]) + 0);
                                _vec_load_31[0] = _v2_243.x;
                                _vec_load_31[0 + 1] = _v2_243.y;
#else
                                float2 _v2_255 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_21 * 256 + dim_1_7[0]) + 0);
                                _vec_load_31[0] = _v2_255.x;
                                _vec_load_31[0 + 1] = _v2_255.y;
#endif
                            }
                            p0_1[5] = _vec_load_31[0];
                            p1_1[5] = _vec_load_31[1];
                        }
                        lse_1[6] = -CUDART_INF_F;
                        if (base_1 + 6 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_22 = (long long)(base_1 + 6) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_22 = (long long)(base_1 + 6) * split_pitch + row_head_4[0];
#endif
                            lse_1[6] = partial_lse[part_row_1_22];
                            float _vec_load_32[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_244 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_22 * 256 + dim_1_4[0]) + 0);
                                _vec_load_32[0] = _v2_244.x;
                                _vec_load_32[0 + 1] = _v2_244.y;
#else
                                float2 _v2_256 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_22 * 256 + dim_1_7[0]) + 0);
                                _vec_load_32[0] = _v2_256.x;
                                _vec_load_32[0 + 1] = _v2_256.y;
#endif
                            }
                            p0_1[6] = _vec_load_32[0];
                            p1_1[6] = _vec_load_32[1];
                        }
                        lse_1[7] = -CUDART_INF_F;
                        if (base_1 + 7 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_23 = (long long)(base_1 + 7) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_23 = (long long)(base_1 + 7) * split_pitch + row_head_4[0];
#endif
                            lse_1[7] = partial_lse[part_row_1_23];
                            float _vec_load_33[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_245 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_23 * 256 + dim_1_4[0]) + 0);
                                _vec_load_33[0] = _v2_245.x;
                                _vec_load_33[0 + 1] = _v2_245.y;
#else
                                float2 _v2_257 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_23 * 256 + dim_1_7[0]) + 0);
                                _vec_load_33[0] = _v2_257.x;
                                _vec_load_33[0 + 1] = _v2_257.y;
#endif
                            }
                            p0_1[7] = _vec_load_33[0];
                            p1_1[7] = _vec_load_33[1];
                        }
                        lse_1[8] = -CUDART_INF_F;
                        if (base_1 + 8 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_24 = (long long)(base_1 + 8) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_24 = (long long)(base_1 + 8) * split_pitch + row_head_4[0];
#endif
                            lse_1[8] = partial_lse[part_row_1_24];
                            float _vec_load_34[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_246 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_24 * 256 + dim_1_4[0]) + 0);
                                _vec_load_34[0] = _v2_246.x;
                                _vec_load_34[0 + 1] = _v2_246.y;
#else
                                float2 _v2_258 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_24 * 256 + dim_1_7[0]) + 0);
                                _vec_load_34[0] = _v2_258.x;
                                _vec_load_34[0 + 1] = _v2_258.y;
#endif
                            }
                            p0_1[8] = _vec_load_34[0];
                            p1_1[8] = _vec_load_34[1];
                        }
                        lse_1[9] = -CUDART_INF_F;
                        if (base_1 + 9 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_25 = (long long)(base_1 + 9) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_25 = (long long)(base_1 + 9) * split_pitch + row_head_4[0];
#endif
                            lse_1[9] = partial_lse[part_row_1_25];
                            float _vec_load_35[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_247 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_25 * 256 + dim_1_4[0]) + 0);
                                _vec_load_35[0] = _v2_247.x;
                                _vec_load_35[0 + 1] = _v2_247.y;
#else
                                float2 _v2_259 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_25 * 256 + dim_1_7[0]) + 0);
                                _vec_load_35[0] = _v2_259.x;
                                _vec_load_35[0 + 1] = _v2_259.y;
#endif
                            }
                            p0_1[9] = _vec_load_35[0];
                            p1_1[9] = _vec_load_35[1];
                        }
                        lse_1[10] = -CUDART_INF_F;
                        if (base_1 + 10 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_26 = (long long)(base_1 + 10) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_26 = (long long)(base_1 + 10) * split_pitch + row_head_4[0];
#endif
                            lse_1[10] = partial_lse[part_row_1_26];
                            float _vec_load_36[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_248 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_26 * 256 + dim_1_4[0]) + 0);
                                _vec_load_36[0] = _v2_248.x;
                                _vec_load_36[0 + 1] = _v2_248.y;
#else
                                float2 _v2_260 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_26 * 256 + dim_1_7[0]) + 0);
                                _vec_load_36[0] = _v2_260.x;
                                _vec_load_36[0 + 1] = _v2_260.y;
#endif
                            }
                            p0_1[10] = _vec_load_36[0];
                            p1_1[10] = _vec_load_36[1];
                        }
                        lse_1[11] = -CUDART_INF_F;
                        if (base_1 + 11 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_27 = (long long)(base_1 + 11) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_27 = (long long)(base_1 + 11) * split_pitch + row_head_4[0];
#endif
                            lse_1[11] = partial_lse[part_row_1_27];
                            float _vec_load_37[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_249 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_27 * 256 + dim_1_4[0]) + 0);
                                _vec_load_37[0] = _v2_249.x;
                                _vec_load_37[0 + 1] = _v2_249.y;
#else
                                float2 _v2_261 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_27 * 256 + dim_1_7[0]) + 0);
                                _vec_load_37[0] = _v2_261.x;
                                _vec_load_37[0 + 1] = _v2_261.y;
#endif
                            }
                            p0_1[11] = _vec_load_37[0];
                            p1_1[11] = _vec_load_37[1];
                        }
                        lse_1[12] = -CUDART_INF_F;
                        if (base_1 + 12 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_28 = (long long)(base_1 + 12) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_28 = (long long)(base_1 + 12) * split_pitch + row_head_4[0];
#endif
                            lse_1[12] = partial_lse[part_row_1_28];
                            float _vec_load_38[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_250 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_28 * 256 + dim_1_4[0]) + 0);
                                _vec_load_38[0] = _v2_250.x;
                                _vec_load_38[0 + 1] = _v2_250.y;
#else
                                float2 _v2_262 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_28 * 256 + dim_1_7[0]) + 0);
                                _vec_load_38[0] = _v2_262.x;
                                _vec_load_38[0 + 1] = _v2_262.y;
#endif
                            }
                            p0_1[12] = _vec_load_38[0];
                            p1_1[12] = _vec_load_38[1];
                        }
                        lse_1[13] = -CUDART_INF_F;
                        if (base_1 + 13 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_29 = (long long)(base_1 + 13) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_29 = (long long)(base_1 + 13) * split_pitch + row_head_4[0];
#endif
                            lse_1[13] = partial_lse[part_row_1_29];
                            float _vec_load_39[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_251 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_29 * 256 + dim_1_4[0]) + 0);
                                _vec_load_39[0] = _v2_251.x;
                                _vec_load_39[0 + 1] = _v2_251.y;
#else
                                float2 _v2_263 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_29 * 256 + dim_1_7[0]) + 0);
                                _vec_load_39[0] = _v2_263.x;
                                _vec_load_39[0 + 1] = _v2_263.y;
#endif
                            }
                            p0_1[13] = _vec_load_39[0];
                            p1_1[13] = _vec_load_39[1];
                        }
                        lse_1[14] = -CUDART_INF_F;
                        if (base_1 + 14 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_30 = (long long)(base_1 + 14) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_30 = (long long)(base_1 + 14) * split_pitch + row_head_4[0];
#endif
                            lse_1[14] = partial_lse[part_row_1_30];
                            float _vec_load_40[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_252 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_30 * 256 + dim_1_4[0]) + 0);
                                _vec_load_40[0] = _v2_252.x;
                                _vec_load_40[0 + 1] = _v2_252.y;
#else
                                float2 _v2_264 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_30 * 256 + dim_1_7[0]) + 0);
                                _vec_load_40[0] = _v2_264.x;
                                _vec_load_40[0 + 1] = _v2_264.y;
#endif
                            }
                            p0_1[14] = _vec_load_40[0];
                            p1_1[14] = _vec_load_40[1];
                        }
                        lse_1[15] = -CUDART_INF_F;
                        if (base_1 + 15 < num_splits) {
#if __CUDA_ARCH__ == 800
                            long long part_row_1_31 = (long long)(base_1 + 15) * split_pitch + row_head_1[0];
#else
                            long long part_row_1_31 = (long long)(base_1 + 15) * split_pitch + row_head_4[0];
#endif
                            lse_1[15] = partial_lse[part_row_1_31];
                            float _vec_load_41[2];
                            {
#if __CUDA_ARCH__ == 800
                                float2 _v2_253 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_31 * 256 + dim_1_4[0]) + 0);
                                _vec_load_41[0] = _v2_253.x;
                                _vec_load_41[0 + 1] = _v2_253.y;
#else
                                float2 _v2_265 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_31 * 256 + dim_1_7[0]) + 0);
                                _vec_load_41[0] = _v2_265.x;
                                _vec_load_41[0 + 1] = _v2_265.y;
#endif
                            }
                            p0_1[15] = _vec_load_41[0];
                            p1_1[15] = _vec_load_41[1];
                        }
#if __CUDA_ARCH__ == 800
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
                        bool live_15_16 = base_1 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[0]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[1]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[2]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[3]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[4]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[5]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[6]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[7]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[8]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[9]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[10]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[11]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[12]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[13]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[14]);
                        lse_max_2[0] = fmaxf(lse_max_2[0], lse_1[15]);
                        bool live_15_16 = base_1 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_16) {
#if __CUDA_ARCH__ == 800
                            float weight_1_16 = approx_exp2(lse_1[0] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_16;
                            acc0_1[0] = __fmaf_rn(weight_1_16, p0_1[0], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_16, p1_1[0], acc1_1[0]);
#else
                            float weight_1_16 = approx_exp2(lse_1[0] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_16;
                            acc0_2[0] = __fmaf_rn(weight_1_16, p0_1[0], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_16, p1_1[0], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_17 = base_1 + 1 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_17 = base_1 + 1 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_17) {
#if __CUDA_ARCH__ == 800
                            float weight_1_17 = approx_exp2(lse_1[1] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_17;
                            acc0_1[0] = __fmaf_rn(weight_1_17, p0_1[1], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_17, p1_1[1], acc1_1[0]);
#else
                            float weight_1_17 = approx_exp2(lse_1[1] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_17;
                            acc0_2[0] = __fmaf_rn(weight_1_17, p0_1[1], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_17, p1_1[1], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_18 = base_1 + 2 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_18 = base_1 + 2 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_18) {
#if __CUDA_ARCH__ == 800
                            float weight_1_18 = approx_exp2(lse_1[2] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_18;
                            acc0_1[0] = __fmaf_rn(weight_1_18, p0_1[2], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_18, p1_1[2], acc1_1[0]);
#else
                            float weight_1_18 = approx_exp2(lse_1[2] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_18;
                            acc0_2[0] = __fmaf_rn(weight_1_18, p0_1[2], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_18, p1_1[2], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_19 = base_1 + 3 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_19 = base_1 + 3 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_19) {
#if __CUDA_ARCH__ == 800
                            float weight_1_19 = approx_exp2(lse_1[3] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_19;
                            acc0_1[0] = __fmaf_rn(weight_1_19, p0_1[3], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_19, p1_1[3], acc1_1[0]);
#else
                            float weight_1_19 = approx_exp2(lse_1[3] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_19;
                            acc0_2[0] = __fmaf_rn(weight_1_19, p0_1[3], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_19, p1_1[3], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_20 = base_1 + 4 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_20 = base_1 + 4 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_20) {
#if __CUDA_ARCH__ == 800
                            float weight_1_20 = approx_exp2(lse_1[4] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_20;
                            acc0_1[0] = __fmaf_rn(weight_1_20, p0_1[4], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_20, p1_1[4], acc1_1[0]);
#else
                            float weight_1_20 = approx_exp2(lse_1[4] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_20;
                            acc0_2[0] = __fmaf_rn(weight_1_20, p0_1[4], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_20, p1_1[4], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_21 = base_1 + 5 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_21 = base_1 + 5 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_21) {
#if __CUDA_ARCH__ == 800
                            float weight_1_21 = approx_exp2(lse_1[5] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_21;
                            acc0_1[0] = __fmaf_rn(weight_1_21, p0_1[5], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_21, p1_1[5], acc1_1[0]);
#else
                            float weight_1_21 = approx_exp2(lse_1[5] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_21;
                            acc0_2[0] = __fmaf_rn(weight_1_21, p0_1[5], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_21, p1_1[5], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_22 = base_1 + 6 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_22 = base_1 + 6 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_22) {
#if __CUDA_ARCH__ == 800
                            float weight_1_22 = approx_exp2(lse_1[6] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_22;
                            acc0_1[0] = __fmaf_rn(weight_1_22, p0_1[6], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_22, p1_1[6], acc1_1[0]);
#else
                            float weight_1_22 = approx_exp2(lse_1[6] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_22;
                            acc0_2[0] = __fmaf_rn(weight_1_22, p0_1[6], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_22, p1_1[6], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_23 = base_1 + 7 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_23 = base_1 + 7 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_23) {
#if __CUDA_ARCH__ == 800
                            float weight_1_23 = approx_exp2(lse_1[7] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_23;
                            acc0_1[0] = __fmaf_rn(weight_1_23, p0_1[7], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_23, p1_1[7], acc1_1[0]);
#else
                            float weight_1_23 = approx_exp2(lse_1[7] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_23;
                            acc0_2[0] = __fmaf_rn(weight_1_23, p0_1[7], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_23, p1_1[7], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_24 = base_1 + 8 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_24 = base_1 + 8 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_24) {
#if __CUDA_ARCH__ == 800
                            float weight_1_24 = approx_exp2(lse_1[8] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_24;
                            acc0_1[0] = __fmaf_rn(weight_1_24, p0_1[8], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_24, p1_1[8], acc1_1[0]);
#else
                            float weight_1_24 = approx_exp2(lse_1[8] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_24;
                            acc0_2[0] = __fmaf_rn(weight_1_24, p0_1[8], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_24, p1_1[8], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_25 = base_1 + 9 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_25 = base_1 + 9 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_25) {
#if __CUDA_ARCH__ == 800
                            float weight_1_25 = approx_exp2(lse_1[9] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_25;
                            acc0_1[0] = __fmaf_rn(weight_1_25, p0_1[9], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_25, p1_1[9], acc1_1[0]);
#else
                            float weight_1_25 = approx_exp2(lse_1[9] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_25;
                            acc0_2[0] = __fmaf_rn(weight_1_25, p0_1[9], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_25, p1_1[9], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_26 = base_1 + 10 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_26 = base_1 + 10 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_26) {
#if __CUDA_ARCH__ == 800
                            float weight_1_26 = approx_exp2(lse_1[10] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_26;
                            acc0_1[0] = __fmaf_rn(weight_1_26, p0_1[10], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_26, p1_1[10], acc1_1[0]);
#else
                            float weight_1_26 = approx_exp2(lse_1[10] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_26;
                            acc0_2[0] = __fmaf_rn(weight_1_26, p0_1[10], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_26, p1_1[10], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_27 = base_1 + 11 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_27 = base_1 + 11 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_27) {
#if __CUDA_ARCH__ == 800
                            float weight_1_27 = approx_exp2(lse_1[11] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_27;
                            acc0_1[0] = __fmaf_rn(weight_1_27, p0_1[11], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_27, p1_1[11], acc1_1[0]);
#else
                            float weight_1_27 = approx_exp2(lse_1[11] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_27;
                            acc0_2[0] = __fmaf_rn(weight_1_27, p0_1[11], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_27, p1_1[11], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_28 = base_1 + 12 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_28 = base_1 + 12 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_28) {
#if __CUDA_ARCH__ == 800
                            float weight_1_28 = approx_exp2(lse_1[12] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_28;
                            acc0_1[0] = __fmaf_rn(weight_1_28, p0_1[12], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_28, p1_1[12], acc1_1[0]);
#else
                            float weight_1_28 = approx_exp2(lse_1[12] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_28;
                            acc0_2[0] = __fmaf_rn(weight_1_28, p0_1[12], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_28, p1_1[12], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_29 = base_1 + 13 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_29 = base_1 + 13 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_29) {
#if __CUDA_ARCH__ == 800
                            float weight_1_29 = approx_exp2(lse_1[13] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_29;
                            acc0_1[0] = __fmaf_rn(weight_1_29, p0_1[13], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_29, p1_1[13], acc1_1[0]);
#else
                            float weight_1_29 = approx_exp2(lse_1[13] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_29;
                            acc0_2[0] = __fmaf_rn(weight_1_29, p0_1[13], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_29, p1_1[13], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_30 = base_1 + 14 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_30 = base_1 + 14 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_30) {
#if __CUDA_ARCH__ == 800
                            float weight_1_30 = approx_exp2(lse_1[14] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_30;
                            acc0_1[0] = __fmaf_rn(weight_1_30, p0_1[14], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_30, p1_1[14], acc1_1[0]);
#else
                            float weight_1_30 = approx_exp2(lse_1[14] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_30;
                            acc0_2[0] = __fmaf_rn(weight_1_30, p0_1[14], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_30, p1_1[14], acc1_2[0]);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        bool live_15_31 = base_1 + 15 < num_splits && lse_max_1[0] > -CUDART_INF_F;
#else
                        bool live_15_31 = base_1 + 15 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#endif
                        if (live_15_31) {
#if __CUDA_ARCH__ == 800
                            float weight_1_31 = approx_exp2(lse_1[15] - lse_max_1[0]);
                            den_1[0] = den_1[0] + weight_1_31;
                            acc0_1[0] = __fmaf_rn(weight_1_31, p0_1[15], acc0_1[0]);
                            acc1_1[0] = __fmaf_rn(weight_1_31, p1_1[15], acc1_1[0]);
#else
                            float weight_1_31 = approx_exp2(lse_1[15] - lse_max_2[0]);
                            den_2[0] = den_2[0] + weight_1_31;
                            acc0_2[0] = __fmaf_rn(weight_1_31, p0_1[15], acc0_2[0]);
                            acc1_2[0] = __fmaf_rn(weight_1_31, p1_1[15], acc1_2[0]);
#endif
                        }
                    }
                    if (valid_0_5[0] != 0) {
#if __CUDA_ARCH__ == 800
                        float inv_3 = 0.0f;
                        if (den_1[0] > 0.0f) {
                            float _rcp_254 = approx_rcp(den_1[0]);
                            inv_3 = _rcp_254;
#else
                        float inv_4 = 0.0f;
                        if (den_2[0] > 0.0f) {
                            float _rcp_266 = approx_rcp(den_2[0]);
                            inv_4 = _rcp_266;
#endif
                        }
#if __CUDA_ARCH__ == 800
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
                        long long out_offset_1 = row_i64 * out_stride_row + qo_head_3[0] * out_stride_head + dim_1_4[0];
                        *(reinterpret_cast<unsigned int*>(out + out_offset_1) + (0)) = word_1[0];
#else
                        float attn0_2 = (float)(__nv_bfloat16)(acc0_2[0] * inv_4);
                        float attn1_2 = (float)(__nv_bfloat16)(acc1_2[0] * inv_4);
                        float _exp2_268 = approx_exp2((-gate0_2[0]) * 1.442695f);
                        float _rcp_267 = approx_rcp(1.0f + _exp2_268);
                        float result0_2 = attn0_2 * _rcp_267;
                        float _exp2_270 = approx_exp2((-gate1_2[0]) * 1.442695f);
                        float _rcp_269 = approx_rcp(1.0f + _exp2_270);
                        float result1_2 = attn1_2 * _rcp_269;
                        __nv_bfloat162 _bit_cast_src_271 = __float22bfloat162_rn(make_float2(result0_2, result1_2));
                        word_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_271)[0];
                        long long out_offset_2 = row_i64 * out_stride_row + qo_head_4[0] * out_stride_head + dim_1_7[0];
                        *(reinterpret_cast<unsigned int*>(out + out_offset_2) + (0)) = word_2[0];
#endif
                    }
                }
            } else {
                if (num_splits > 4) {
                    float lse_2[16];
                    float p0_2[16];
                    float p1_2[16];
#if __CUDA_ARCH__ == 800
                    float acc0_2[2];
                    float acc1_2[2];
                    float den_2[2];
                    float lse_max_2[2];
#else
                    float acc0_3[2];
                    float acc1_3[2];
                    float den_3[2];
                    float lse_max_3[2];
#endif
                    int valid_0_6[2];
#if __CUDA_ARCH__ == 800
                    int tab_2[2];
                    long long dim_1_5[2];
                    long long qo_head_4[2];
                    long long row_head_2[2];
                    float gate0_2[2];
                    float gate1_2[2];
                    unsigned int word_2[1];
                    long long nqo_2 = (long long)num_qo_heads;
#else
                    int tab_3[2];
                    long long dim_1_8[2];
                    long long qo_head_5[2];
                    long long row_head_5[2];
                    float gate0_3[2];
                    float gate1_3[2];
                    unsigned int word_3[1];
                    long long nqo_3 = (long long)num_qo_heads;
#endif
                    #pragma unroll 1
#if __CUDA_ARCH__ == 800
                    for (int p_2 = p_first; p_2 < end; p_2 += 256) {
                        int pk_2_0 = ((p_2) < (end - 1) ? (p_2) : (end - 1));
                        valid_0_6[0] = ((end > p_2) ? 1 : 0);
                        int head_0_2 = pk_2_0 >> 7;
                        dim_1_5[0] = (long long)((pk_2_0 & 127) * 2);
                        qo_head_4[0] = (long long)(first_head + head_0_2);
                        row_head_2[0] = row_i64 * nqo_2 + qo_head_4[0];
                        acc0_2[0] = 0.0f;
                        acc1_2[0] = 0.0f;
                        den_2[0] = 0.0f;
                        lse_max_2[0] = -CUDART_INF_F;
#else
                    for (int p_3 = p_first; p_3 < end; p_3 += 256) {
                        int pk_2_0 = ((p_3) < (end - 1) ? (p_3) : (end - 1));
                        valid_0_6[0] = ((end > p_3) ? 1 : 0);
                        int head_0_5 = pk_2_0 >> 7;
                        dim_1_8[0] = (long long)((pk_2_0 & 127) * 2);
                        qo_head_5[0] = (long long)(first_head + head_0_5);
                        row_head_5[0] = row_i64 * nqo_3 + qo_head_5[0];
                        acc0_3[0] = 0.0f;
                        acc1_3[0] = 0.0f;
                        den_3[0] = 0.0f;
                        lse_max_3[0] = -CUDART_INF_F;
#endif
                        float _vec_load_42[2];
                        {
#if __CUDA_ARCH__ == 800
                            uint32_t _bf16x2_bits_260;
                            _bf16x2_bits_260 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2[0] * 256 + dim_1_5[0]) + 0);
                            (&_vec_load_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_260) << 16);
                            (&_vec_load_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_260) & 0xffff0000u);
#else
                            uint32_t _bf16x2_bits_272;
                            _bf16x2_bits_272 = *reinterpret_cast<const uint32_t*>(gate + (row_head_5[0] * 256 + dim_1_8[0]) + 0);
                            (&_vec_load_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_272) << 16);
                            (&_vec_load_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_272) & 0xffff0000u);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        gate0_2[0] = _vec_load_42[0];
                        gate1_2[0] = _vec_load_42[1];
                        int pk_2_1 = ((p_2 + 128) < (end - 1) ? (p_2 + 128) : (end - 1));
                        valid_0_6[1] = ((end > p_2 + 128) ? 1 : 0);
#else
                        gate0_3[0] = _vec_load_42[0];
                        gate1_3[0] = _vec_load_42[1];
                        int pk_2_1 = ((p_3 + 128) < (end - 1) ? (p_3 + 128) : (end - 1));
                        valid_0_6[1] = ((end > p_3 + 128) ? 1 : 0);
#endif
                        int head_2 = pk_2_1 >> 7;
#if __CUDA_ARCH__ == 800
                        dim_1_5[1] = (long long)((pk_2_1 & 127) * 2);
                        qo_head_4[1] = (long long)(first_head + head_2);
                        row_head_2[1] = row_i64 * nqo_2 + qo_head_4[1];
                        acc0_2[1] = 0.0f;
                        acc1_2[1] = 0.0f;
                        den_2[1] = 0.0f;
                        lse_max_2[1] = -CUDART_INF_F;
#else
                        dim_1_8[1] = (long long)((pk_2_1 & 127) * 2);
                        qo_head_5[1] = (long long)(first_head + head_2);
                        row_head_5[1] = row_i64 * nqo_3 + qo_head_5[1];
                        acc0_3[1] = 0.0f;
                        acc1_3[1] = 0.0f;
                        den_3[1] = 0.0f;
                        lse_max_3[1] = -CUDART_INF_F;
#endif
                        float _vec_load_43[2];
                        {
#if __CUDA_ARCH__ == 800
                            uint32_t _bf16x2_bits_261;
                            _bf16x2_bits_261 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2[1] * 256 + dim_1_5[1]) + 0);
                            (&_vec_load_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_261) << 16);
                            (&_vec_load_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_261) & 0xffff0000u);
#else
                            uint32_t _bf16x2_bits_273;
                            _bf16x2_bits_273 = *reinterpret_cast<const uint32_t*>(gate + (row_head_5[1] * 256 + dim_1_8[1]) + 0);
                            (&_vec_load_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_273) << 16);
                            (&_vec_load_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_273) & 0xffff0000u);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        gate0_2[1] = _vec_load_43[0];
                        gate1_2[1] = _vec_load_43[1];
#else
                        gate0_3[1] = _vec_load_43[0];
                        gate1_3[1] = _vec_load_43[1];
#endif
                        #pragma unroll 1
                        for (int base_2 = 0; base_2 < num_splits; base_2 += 8) {
                            lse_2[0] = -CUDART_INF_F;
                            if (base_2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_32 = (long long)base_2 * split_pitch + row_head_2[0];
#else
                                long long part_row_1_32 = (long long)base_2 * split_pitch + row_head_5[0];
#endif
                                lse_2[0] = partial_lse[part_row_1_32];
                                float _vec_load_44[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_262 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_32 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_44[0] = _v2_262.x;
                                    _vec_load_44[0 + 1] = _v2_262.y;
#else
                                    float2 _v2_274 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_32 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_44[0] = _v2_274.x;
                                    _vec_load_44[0 + 1] = _v2_274.y;
#endif
                                }
                                p0_2[0] = _vec_load_44[0];
                                p1_2[0] = _vec_load_44[1];
                            }
                            lse_2[1] = -CUDART_INF_F;
                            if (base_2 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_33 = (long long)(base_2 + 1) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_33 = (long long)(base_2 + 1) * split_pitch + row_head_5[0];
#endif
                                lse_2[1] = partial_lse[part_row_1_33];
                                float _vec_load_45[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_263 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_33 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_45[0] = _v2_263.x;
                                    _vec_load_45[0 + 1] = _v2_263.y;
#else
                                    float2 _v2_275 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_33 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_45[0] = _v2_275.x;
                                    _vec_load_45[0 + 1] = _v2_275.y;
#endif
                                }
                                p0_2[1] = _vec_load_45[0];
                                p1_2[1] = _vec_load_45[1];
                            }
                            lse_2[2] = -CUDART_INF_F;
                            if (base_2 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_34 = (long long)(base_2 + 2) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_34 = (long long)(base_2 + 2) * split_pitch + row_head_5[0];
#endif
                                lse_2[2] = partial_lse[part_row_1_34];
                                float _vec_load_46[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_264 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_34 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_46[0] = _v2_264.x;
                                    _vec_load_46[0 + 1] = _v2_264.y;
#else
                                    float2 _v2_276 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_34 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_46[0] = _v2_276.x;
                                    _vec_load_46[0 + 1] = _v2_276.y;
#endif
                                }
                                p0_2[2] = _vec_load_46[0];
                                p1_2[2] = _vec_load_46[1];
                            }
                            lse_2[3] = -CUDART_INF_F;
                            if (base_2 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_35 = (long long)(base_2 + 3) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_35 = (long long)(base_2 + 3) * split_pitch + row_head_5[0];
#endif
                                lse_2[3] = partial_lse[part_row_1_35];
                                float _vec_load_47[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_265 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_35 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_47[0] = _v2_265.x;
                                    _vec_load_47[0 + 1] = _v2_265.y;
#else
                                    float2 _v2_277 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_35 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_47[0] = _v2_277.x;
                                    _vec_load_47[0 + 1] = _v2_277.y;
#endif
                                }
                                p0_2[3] = _vec_load_47[0];
                                p1_2[3] = _vec_load_47[1];
                            }
                            lse_2[4] = -CUDART_INF_F;
                            if (base_2 + 4 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_36 = (long long)(base_2 + 4) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_36 = (long long)(base_2 + 4) * split_pitch + row_head_5[0];
#endif
                                lse_2[4] = partial_lse[part_row_1_36];
                                float _vec_load_48[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_266 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_36 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_48[0] = _v2_266.x;
                                    _vec_load_48[0 + 1] = _v2_266.y;
#else
                                    float2 _v2_278 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_36 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_48[0] = _v2_278.x;
                                    _vec_load_48[0 + 1] = _v2_278.y;
#endif
                                }
                                p0_2[4] = _vec_load_48[0];
                                p1_2[4] = _vec_load_48[1];
                            }
                            lse_2[5] = -CUDART_INF_F;
                            if (base_2 + 5 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_37 = (long long)(base_2 + 5) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_37 = (long long)(base_2 + 5) * split_pitch + row_head_5[0];
#endif
                                lse_2[5] = partial_lse[part_row_1_37];
                                float _vec_load_49[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_267 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_37 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_49[0] = _v2_267.x;
                                    _vec_load_49[0 + 1] = _v2_267.y;
#else
                                    float2 _v2_279 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_37 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_49[0] = _v2_279.x;
                                    _vec_load_49[0 + 1] = _v2_279.y;
#endif
                                }
                                p0_2[5] = _vec_load_49[0];
                                p1_2[5] = _vec_load_49[1];
                            }
                            lse_2[6] = -CUDART_INF_F;
                            if (base_2 + 6 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_38 = (long long)(base_2 + 6) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_38 = (long long)(base_2 + 6) * split_pitch + row_head_5[0];
#endif
                                lse_2[6] = partial_lse[part_row_1_38];
                                float _vec_load_50[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_268 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_38 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_50[0] = _v2_268.x;
                                    _vec_load_50[0 + 1] = _v2_268.y;
#else
                                    float2 _v2_280 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_38 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_50[0] = _v2_280.x;
                                    _vec_load_50[0 + 1] = _v2_280.y;
#endif
                                }
                                p0_2[6] = _vec_load_50[0];
                                p1_2[6] = _vec_load_50[1];
                            }
                            lse_2[7] = -CUDART_INF_F;
                            if (base_2 + 7 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_39 = (long long)(base_2 + 7) * split_pitch + row_head_2[0];
#else
                                long long part_row_1_39 = (long long)(base_2 + 7) * split_pitch + row_head_5[0];
#endif
                                lse_2[7] = partial_lse[part_row_1_39];
                                float _vec_load_51[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_269 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_39 * 256 + dim_1_5[0]) + 0);
                                    _vec_load_51[0] = _v2_269.x;
                                    _vec_load_51[0 + 1] = _v2_269.y;
#else
                                    float2 _v2_281 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_39 * 256 + dim_1_8[0]) + 0);
                                    _vec_load_51[0] = _v2_281.x;
                                    _vec_load_51[0 + 1] = _v2_281.y;
#endif
                                }
                                p0_2[7] = _vec_load_51[0];
                                p1_2[7] = _vec_load_51[1];
                            }
                            lse_2[8] = -CUDART_INF_F;
                            if (base_2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_40 = (long long)base_2 * split_pitch + row_head_2[1];
#else
                                long long part_row_1_40 = (long long)base_2 * split_pitch + row_head_5[1];
#endif
                                lse_2[8] = partial_lse[part_row_1_40];
                                float _vec_load_52[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_270 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_40 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_52[0] = _v2_270.x;
                                    _vec_load_52[0 + 1] = _v2_270.y;
#else
                                    float2 _v2_282 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_40 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_52[0] = _v2_282.x;
                                    _vec_load_52[0 + 1] = _v2_282.y;
#endif
                                }
                                p0_2[8] = _vec_load_52[0];
                                p1_2[8] = _vec_load_52[1];
                            }
                            lse_2[9] = -CUDART_INF_F;
                            if (base_2 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_41 = (long long)(base_2 + 1) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_41 = (long long)(base_2 + 1) * split_pitch + row_head_5[1];
#endif
                                lse_2[9] = partial_lse[part_row_1_41];
                                float _vec_load_53[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_271 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_41 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_53[0] = _v2_271.x;
                                    _vec_load_53[0 + 1] = _v2_271.y;
#else
                                    float2 _v2_283 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_41 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_53[0] = _v2_283.x;
                                    _vec_load_53[0 + 1] = _v2_283.y;
#endif
                                }
                                p0_2[9] = _vec_load_53[0];
                                p1_2[9] = _vec_load_53[1];
                            }
                            lse_2[10] = -CUDART_INF_F;
                            if (base_2 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_42 = (long long)(base_2 + 2) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_42 = (long long)(base_2 + 2) * split_pitch + row_head_5[1];
#endif
                                lse_2[10] = partial_lse[part_row_1_42];
                                float _vec_load_54[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_272 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_42 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_54[0] = _v2_272.x;
                                    _vec_load_54[0 + 1] = _v2_272.y;
#else
                                    float2 _v2_284 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_42 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_54[0] = _v2_284.x;
                                    _vec_load_54[0 + 1] = _v2_284.y;
#endif
                                }
                                p0_2[10] = _vec_load_54[0];
                                p1_2[10] = _vec_load_54[1];
                            }
                            lse_2[11] = -CUDART_INF_F;
                            if (base_2 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_43 = (long long)(base_2 + 3) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_43 = (long long)(base_2 + 3) * split_pitch + row_head_5[1];
#endif
                                lse_2[11] = partial_lse[part_row_1_43];
                                float _vec_load_55[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_273 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_43 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_55[0] = _v2_273.x;
                                    _vec_load_55[0 + 1] = _v2_273.y;
#else
                                    float2 _v2_285 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_43 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_55[0] = _v2_285.x;
                                    _vec_load_55[0 + 1] = _v2_285.y;
#endif
                                }
                                p0_2[11] = _vec_load_55[0];
                                p1_2[11] = _vec_load_55[1];
                            }
                            lse_2[12] = -CUDART_INF_F;
                            if (base_2 + 4 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_44 = (long long)(base_2 + 4) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_44 = (long long)(base_2 + 4) * split_pitch + row_head_5[1];
#endif
                                lse_2[12] = partial_lse[part_row_1_44];
                                float _vec_load_56[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_274 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_44 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_56[0] = _v2_274.x;
                                    _vec_load_56[0 + 1] = _v2_274.y;
#else
                                    float2 _v2_286 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_44 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_56[0] = _v2_286.x;
                                    _vec_load_56[0 + 1] = _v2_286.y;
#endif
                                }
                                p0_2[12] = _vec_load_56[0];
                                p1_2[12] = _vec_load_56[1];
                            }
                            lse_2[13] = -CUDART_INF_F;
                            if (base_2 + 5 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_45 = (long long)(base_2 + 5) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_45 = (long long)(base_2 + 5) * split_pitch + row_head_5[1];
#endif
                                lse_2[13] = partial_lse[part_row_1_45];
                                float _vec_load_57[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_275 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_45 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_57[0] = _v2_275.x;
                                    _vec_load_57[0 + 1] = _v2_275.y;
#else
                                    float2 _v2_287 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_45 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_57[0] = _v2_287.x;
                                    _vec_load_57[0 + 1] = _v2_287.y;
#endif
                                }
                                p0_2[13] = _vec_load_57[0];
                                p1_2[13] = _vec_load_57[1];
                            }
                            lse_2[14] = -CUDART_INF_F;
                            if (base_2 + 6 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_46 = (long long)(base_2 + 6) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_46 = (long long)(base_2 + 6) * split_pitch + row_head_5[1];
#endif
                                lse_2[14] = partial_lse[part_row_1_46];
                                float _vec_load_58[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_276 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_46 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_58[0] = _v2_276.x;
                                    _vec_load_58[0 + 1] = _v2_276.y;
#else
                                    float2 _v2_288 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_46 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_58[0] = _v2_288.x;
                                    _vec_load_58[0 + 1] = _v2_288.y;
#endif
                                }
                                p0_2[14] = _vec_load_58[0];
                                p1_2[14] = _vec_load_58[1];
                            }
                            lse_2[15] = -CUDART_INF_F;
                            if (base_2 + 7 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_47 = (long long)(base_2 + 7) * split_pitch + row_head_2[1];
#else
                                long long part_row_1_47 = (long long)(base_2 + 7) * split_pitch + row_head_5[1];
#endif
                                lse_2[15] = partial_lse[part_row_1_47];
                                float _vec_load_59[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_277 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_47 * 256 + dim_1_5[1]) + 0);
                                    _vec_load_59[0] = _v2_277.x;
                                    _vec_load_59[0 + 1] = _v2_277.y;
#else
                                    float2 _v2_289 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_47 * 256 + dim_1_8[1]) + 0);
                                    _vec_load_59[0] = _v2_289.x;
                                    _vec_load_59[0 + 1] = _v2_289.y;
#endif
                                }
                                p0_2[15] = _vec_load_59[0];
                                p1_2[15] = _vec_load_59[1];
                            }
#if __CUDA_ARCH__ == 800
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
                            bool live_15_32 = base_2 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[0]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[1]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[2]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[3]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[4]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[5]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[6]);
                            lse_max_3[0] = fmaxf(lse_max_3[0], lse_2[7]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[8]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[9]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[10]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[11]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[12]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[13]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[14]);
                            lse_max_3[1] = fmaxf(lse_max_3[1], lse_2[15]);
                            bool live_15_32 = base_2 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_32) {
#if __CUDA_ARCH__ == 800
                                float weight_1_32 = approx_exp2(lse_2[0] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_32;
                                acc0_2[0] = __fmaf_rn(weight_1_32, p0_2[0], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_32, p1_2[0], acc1_2[0]);
#else
                                float weight_1_32 = approx_exp2(lse_2[0] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_32;
                                acc0_3[0] = __fmaf_rn(weight_1_32, p0_2[0], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_32, p1_2[0], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_33 = base_2 + 1 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_33 = base_2 + 1 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_33) {
#if __CUDA_ARCH__ == 800
                                float weight_1_33 = approx_exp2(lse_2[1] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_33;
                                acc0_2[0] = __fmaf_rn(weight_1_33, p0_2[1], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_33, p1_2[1], acc1_2[0]);
#else
                                float weight_1_33 = approx_exp2(lse_2[1] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_33;
                                acc0_3[0] = __fmaf_rn(weight_1_33, p0_2[1], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_33, p1_2[1], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_34 = base_2 + 2 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_34 = base_2 + 2 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_34) {
#if __CUDA_ARCH__ == 800
                                float weight_1_34 = approx_exp2(lse_2[2] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_34;
                                acc0_2[0] = __fmaf_rn(weight_1_34, p0_2[2], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_34, p1_2[2], acc1_2[0]);
#else
                                float weight_1_34 = approx_exp2(lse_2[2] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_34;
                                acc0_3[0] = __fmaf_rn(weight_1_34, p0_2[2], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_34, p1_2[2], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_35 = base_2 + 3 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_35 = base_2 + 3 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_35) {
#if __CUDA_ARCH__ == 800
                                float weight_1_35 = approx_exp2(lse_2[3] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_35;
                                acc0_2[0] = __fmaf_rn(weight_1_35, p0_2[3], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_35, p1_2[3], acc1_2[0]);
#else
                                float weight_1_35 = approx_exp2(lse_2[3] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_35;
                                acc0_3[0] = __fmaf_rn(weight_1_35, p0_2[3], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_35, p1_2[3], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_36 = base_2 + 4 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_36 = base_2 + 4 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_36) {
#if __CUDA_ARCH__ == 800
                                float weight_1_36 = approx_exp2(lse_2[4] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_36;
                                acc0_2[0] = __fmaf_rn(weight_1_36, p0_2[4], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_36, p1_2[4], acc1_2[0]);
#else
                                float weight_1_36 = approx_exp2(lse_2[4] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_36;
                                acc0_3[0] = __fmaf_rn(weight_1_36, p0_2[4], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_36, p1_2[4], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_37 = base_2 + 5 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_37 = base_2 + 5 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_37) {
#if __CUDA_ARCH__ == 800
                                float weight_1_37 = approx_exp2(lse_2[5] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_37;
                                acc0_2[0] = __fmaf_rn(weight_1_37, p0_2[5], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_37, p1_2[5], acc1_2[0]);
#else
                                float weight_1_37 = approx_exp2(lse_2[5] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_37;
                                acc0_3[0] = __fmaf_rn(weight_1_37, p0_2[5], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_37, p1_2[5], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_38 = base_2 + 6 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_38 = base_2 + 6 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_38) {
#if __CUDA_ARCH__ == 800
                                float weight_1_38 = approx_exp2(lse_2[6] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_38;
                                acc0_2[0] = __fmaf_rn(weight_1_38, p0_2[6], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_38, p1_2[6], acc1_2[0]);
#else
                                float weight_1_38 = approx_exp2(lse_2[6] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_38;
                                acc0_3[0] = __fmaf_rn(weight_1_38, p0_2[6], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_38, p1_2[6], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_39 = base_2 + 7 < num_splits && lse_max_2[0] > -CUDART_INF_F;
#else
                            bool live_15_39 = base_2 + 7 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#endif
                            if (live_15_39) {
#if __CUDA_ARCH__ == 800
                                float weight_1_39 = approx_exp2(lse_2[7] - lse_max_2[0]);
                                den_2[0] = den_2[0] + weight_1_39;
                                acc0_2[0] = __fmaf_rn(weight_1_39, p0_2[7], acc0_2[0]);
                                acc1_2[0] = __fmaf_rn(weight_1_39, p1_2[7], acc1_2[0]);
#else
                                float weight_1_39 = approx_exp2(lse_2[7] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_39;
                                acc0_3[0] = __fmaf_rn(weight_1_39, p0_2[7], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_39, p1_2[7], acc1_3[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_40 = base_2 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_40 = base_2 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_40) {
#if __CUDA_ARCH__ == 800
                                float weight_1_40 = approx_exp2(lse_2[8] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_40;
                                acc0_2[1] = __fmaf_rn(weight_1_40, p0_2[8], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_40, p1_2[8], acc1_2[1]);
#else
                                float weight_1_40 = approx_exp2(lse_2[8] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_40;
                                acc0_3[1] = __fmaf_rn(weight_1_40, p0_2[8], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_40, p1_2[8], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_41 = base_2 + 1 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_41 = base_2 + 1 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_41) {
#if __CUDA_ARCH__ == 800
                                float weight_1_41 = approx_exp2(lse_2[9] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_41;
                                acc0_2[1] = __fmaf_rn(weight_1_41, p0_2[9], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_41, p1_2[9], acc1_2[1]);
#else
                                float weight_1_41 = approx_exp2(lse_2[9] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_41;
                                acc0_3[1] = __fmaf_rn(weight_1_41, p0_2[9], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_41, p1_2[9], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_42 = base_2 + 2 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_42 = base_2 + 2 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_42) {
#if __CUDA_ARCH__ == 800
                                float weight_1_42 = approx_exp2(lse_2[10] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_42;
                                acc0_2[1] = __fmaf_rn(weight_1_42, p0_2[10], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_42, p1_2[10], acc1_2[1]);
#else
                                float weight_1_42 = approx_exp2(lse_2[10] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_42;
                                acc0_3[1] = __fmaf_rn(weight_1_42, p0_2[10], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_42, p1_2[10], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_43 = base_2 + 3 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_43 = base_2 + 3 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_43) {
#if __CUDA_ARCH__ == 800
                                float weight_1_43 = approx_exp2(lse_2[11] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_43;
                                acc0_2[1] = __fmaf_rn(weight_1_43, p0_2[11], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_43, p1_2[11], acc1_2[1]);
#else
                                float weight_1_43 = approx_exp2(lse_2[11] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_43;
                                acc0_3[1] = __fmaf_rn(weight_1_43, p0_2[11], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_43, p1_2[11], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_44 = base_2 + 4 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_44 = base_2 + 4 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_44) {
#if __CUDA_ARCH__ == 800
                                float weight_1_44 = approx_exp2(lse_2[12] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_44;
                                acc0_2[1] = __fmaf_rn(weight_1_44, p0_2[12], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_44, p1_2[12], acc1_2[1]);
#else
                                float weight_1_44 = approx_exp2(lse_2[12] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_44;
                                acc0_3[1] = __fmaf_rn(weight_1_44, p0_2[12], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_44, p1_2[12], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_45 = base_2 + 5 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_45 = base_2 + 5 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_45) {
#if __CUDA_ARCH__ == 800
                                float weight_1_45 = approx_exp2(lse_2[13] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_45;
                                acc0_2[1] = __fmaf_rn(weight_1_45, p0_2[13], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_45, p1_2[13], acc1_2[1]);
#else
                                float weight_1_45 = approx_exp2(lse_2[13] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_45;
                                acc0_3[1] = __fmaf_rn(weight_1_45, p0_2[13], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_45, p1_2[13], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_46 = base_2 + 6 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_46 = base_2 + 6 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_46) {
#if __CUDA_ARCH__ == 800
                                float weight_1_46 = approx_exp2(lse_2[14] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_46;
                                acc0_2[1] = __fmaf_rn(weight_1_46, p0_2[14], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_46, p1_2[14], acc1_2[1]);
#else
                                float weight_1_46 = approx_exp2(lse_2[14] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_46;
                                acc0_3[1] = __fmaf_rn(weight_1_46, p0_2[14], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_46, p1_2[14], acc1_3[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_47 = base_2 + 7 < num_splits && lse_max_2[1] > -CUDART_INF_F;
#else
                            bool live_15_47 = base_2 + 7 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#endif
                            if (live_15_47) {
#if __CUDA_ARCH__ == 800
                                float weight_1_47 = approx_exp2(lse_2[15] - lse_max_2[1]);
                                den_2[1] = den_2[1] + weight_1_47;
                                acc0_2[1] = __fmaf_rn(weight_1_47, p0_2[15], acc0_2[1]);
                                acc1_2[1] = __fmaf_rn(weight_1_47, p1_2[15], acc1_2[1]);
#else
                                float weight_1_47 = approx_exp2(lse_2[15] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_47;
                                acc0_3[1] = __fmaf_rn(weight_1_47, p0_2[15], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_47, p1_2[15], acc1_3[1]);
#endif
                            }
                        }
                        if (valid_0_6[0] != 0) {
#if __CUDA_ARCH__ == 800
                            float inv_4 = 0.0f;
                            if (den_2[0] > 0.0f) {
                                float _rcp_278 = approx_rcp(den_2[0]);
                                inv_4 = _rcp_278;
#else
                            float inv_5 = 0.0f;
                            if (den_3[0] > 0.0f) {
                                float _rcp_290 = approx_rcp(den_3[0]);
                                inv_5 = _rcp_290;
#endif
                            }
#if __CUDA_ARCH__ == 800
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
                            long long out_offset_2 = row_i64 * out_stride_row + qo_head_4[0] * out_stride_head + dim_1_5[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_2) + (0)) = word_2[0];
#else
                            float attn0_3 = (float)(__nv_bfloat16)(acc0_3[0] * inv_5);
                            float attn1_3 = (float)(__nv_bfloat16)(acc1_3[0] * inv_5);
                            float _exp2_292 = approx_exp2((-gate0_3[0]) * 1.442695f);
                            float _rcp_291 = approx_rcp(1.0f + _exp2_292);
                            float result0_3 = attn0_3 * _rcp_291;
                            float _exp2_294 = approx_exp2((-gate1_3[0]) * 1.442695f);
                            float _rcp_293 = approx_rcp(1.0f + _exp2_294);
                            float result1_3 = attn1_3 * _rcp_293;
                            __nv_bfloat162 _bit_cast_src_295 = __float22bfloat162_rn(make_float2(result0_3, result1_3));
                            word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_295)[0];
                            long long out_offset_3 = row_i64 * out_stride_row + qo_head_5[0] * out_stride_head + dim_1_8[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_3) + (0)) = word_3[0];
#endif
                        }
                        if (valid_0_6[1] != 0) {
#if __CUDA_ARCH__ == 800
                            float inv_5 = 0.0f;
                            if (den_2[1] > 0.0f) {
                                float _rcp_284 = approx_rcp(den_2[1]);
                                inv_5 = _rcp_284;
#else
                            float inv_6 = 0.0f;
                            if (den_3[1] > 0.0f) {
                                float _rcp_296 = approx_rcp(den_3[1]);
                                inv_6 = _rcp_296;
#endif
                            }
#if __CUDA_ARCH__ == 800
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
                            long long out_offset_3 = row_i64 * out_stride_row + qo_head_4[1] * out_stride_head + dim_1_5[1];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_3) + (0)) = word_2[0];
#else
                            float attn0_4 = (float)(__nv_bfloat16)(acc0_3[1] * inv_6);
                            float attn1_4 = (float)(__nv_bfloat16)(acc1_3[1] * inv_6);
                            float _exp2_298 = approx_exp2((-gate0_3[1]) * 1.442695f);
                            float _rcp_297 = approx_rcp(1.0f + _exp2_298);
                            float result0_4 = attn0_4 * _rcp_297;
                            float _exp2_300 = approx_exp2((-gate1_3[1]) * 1.442695f);
                            float _rcp_299 = approx_rcp(1.0f + _exp2_300);
                            float result1_4 = attn1_4 * _rcp_299;
                            __nv_bfloat162 _bit_cast_src_301 = __float22bfloat162_rn(make_float2(result0_4, result1_4));
                            word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_301)[0];
                            long long out_offset_4 = row_i64 * out_stride_row + qo_head_5[1] * out_stride_head + dim_1_8[1];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_4) + (0)) = word_3[0];
#endif
                        }
                    }
                } else {
                    float lse_3[16];
                    float p0_3[16];
                    float p1_3[16];
#if __CUDA_ARCH__ == 800
                    float acc0_3[4];
                    float acc1_3[4];
                    float den_3[4];
                    float lse_max_3[4];
#else
                    float acc0_4[4];
                    float acc1_4[4];
                    float den_4[4];
                    float lse_max_4[4];
#endif
                    int valid_0_7[4];
#if __CUDA_ARCH__ == 800
                    int tab_3[4];
                    long long dim_1_6[4];
                    long long qo_head_5[4];
                    long long row_head_3[4];
                    float gate0_3[4];
                    float gate1_3[4];
                    unsigned int word_3[1];
                    long long nqo_3 = (long long)num_qo_heads;
#else
                    int tab_4[4];
                    long long dim_1_9[4];
                    long long qo_head_6[4];
                    long long row_head_6[4];
                    float gate0_4[4];
                    float gate1_4[4];
                    unsigned int word_4[1];
                    long long nqo_4 = (long long)num_qo_heads;
#endif
                    #pragma unroll 1
#if __CUDA_ARCH__ == 800
                    for (int p_3 = p_first; p_3 < end; p_3 += 512) {
                        int pk_2_2 = ((p_3) < (end - 1) ? (p_3) : (end - 1));
                        valid_0_7[0] = ((end > p_3) ? 1 : 0);
#else
                    for (int p_4 = p_first; p_4 < end; p_4 += 512) {
                        int pk_2_2 = ((p_4) < (end - 1) ? (p_4) : (end - 1));
                        valid_0_7[0] = ((end > p_4) ? 1 : 0);
#endif
                        int head_3 = pk_2_2 >> 7;
#if __CUDA_ARCH__ == 800
                        dim_1_6[0] = (long long)((pk_2_2 & 127) * 2);
                        qo_head_5[0] = (long long)(first_head + head_3);
                        row_head_3[0] = row_i64 * nqo_3 + qo_head_5[0];
                        acc0_3[0] = 0.0f;
                        acc1_3[0] = 0.0f;
                        den_3[0] = 0.0f;
                        lse_max_3[0] = -CUDART_INF_F;
#else
                        dim_1_9[0] = (long long)((pk_2_2 & 127) * 2);
                        qo_head_6[0] = (long long)(first_head + head_3);
                        row_head_6[0] = row_i64 * nqo_4 + qo_head_6[0];
                        acc0_4[0] = 0.0f;
                        acc1_4[0] = 0.0f;
                        den_4[0] = 0.0f;
                        lse_max_4[0] = -CUDART_INF_F;
#endif
                        float _vec_load_60[2];
                        {
#if __CUDA_ARCH__ == 800
                            uint32_t _bf16x2_bits_290;
                            _bf16x2_bits_290 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[0] * 256 + dim_1_6[0]) + 0);
                            (&_vec_load_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_290) << 16);
                            (&_vec_load_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_290) & 0xffff0000u);
#else
                            uint32_t _bf16x2_bits_302;
                            _bf16x2_bits_302 = *reinterpret_cast<const uint32_t*>(gate + (row_head_6[0] * 256 + dim_1_9[0]) + 0);
                            (&_vec_load_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_302) << 16);
                            (&_vec_load_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_302) & 0xffff0000u);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        gate0_3[0] = _vec_load_60[0];
                        gate1_3[0] = _vec_load_60[1];
                        int pk_2_3 = ((p_3 + 128) < (end - 1) ? (p_3 + 128) : (end - 1));
                        valid_0_7[1] = ((end > p_3 + 128) ? 1 : 0);
#else
                        gate0_4[0] = _vec_load_60[0];
                        gate1_4[0] = _vec_load_60[1];
                        int pk_2_3 = ((p_4 + 128) < (end - 1) ? (p_4 + 128) : (end - 1));
                        valid_0_7[1] = ((end > p_4 + 128) ? 1 : 0);
#endif
                        int head_4 = pk_2_3 >> 7;
#if __CUDA_ARCH__ == 800
                        dim_1_6[1] = (long long)((pk_2_3 & 127) * 2);
                        qo_head_5[1] = (long long)(first_head + head_4);
                        row_head_3[1] = row_i64 * nqo_3 + qo_head_5[1];
                        acc0_3[1] = 0.0f;
                        acc1_3[1] = 0.0f;
                        den_3[1] = 0.0f;
                        lse_max_3[1] = -CUDART_INF_F;
#else
                        dim_1_9[1] = (long long)((pk_2_3 & 127) * 2);
                        qo_head_6[1] = (long long)(first_head + head_4);
                        row_head_6[1] = row_i64 * nqo_4 + qo_head_6[1];
                        acc0_4[1] = 0.0f;
                        acc1_4[1] = 0.0f;
                        den_4[1] = 0.0f;
                        lse_max_4[1] = -CUDART_INF_F;
#endif
                        float _vec_load_61[2];
                        {
#if __CUDA_ARCH__ == 800
                            uint32_t _bf16x2_bits_291;
                            _bf16x2_bits_291 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[1] * 256 + dim_1_6[1]) + 0);
                            (&_vec_load_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_291) << 16);
                            (&_vec_load_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_291) & 0xffff0000u);
#else
                            uint32_t _bf16x2_bits_303;
                            _bf16x2_bits_303 = *reinterpret_cast<const uint32_t*>(gate + (row_head_6[1] * 256 + dim_1_9[1]) + 0);
                            (&_vec_load_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_303) << 16);
                            (&_vec_load_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_303) & 0xffff0000u);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        gate0_3[1] = _vec_load_61[0];
                        gate1_3[1] = _vec_load_61[1];
                        int pk_2_4 = ((p_3 + 256) < (end - 1) ? (p_3 + 256) : (end - 1));
                        valid_0_7[2] = ((end > p_3 + 256) ? 1 : 0);
#else
                        gate0_4[1] = _vec_load_61[0];
                        gate1_4[1] = _vec_load_61[1];
                        int pk_2_4 = ((p_4 + 256) < (end - 1) ? (p_4 + 256) : (end - 1));
                        valid_0_7[2] = ((end > p_4 + 256) ? 1 : 0);
#endif
                        int head_5 = pk_2_4 >> 7;
#if __CUDA_ARCH__ == 800
                        dim_1_6[2] = (long long)((pk_2_4 & 127) * 2);
                        qo_head_5[2] = (long long)(first_head + head_5);
                        row_head_3[2] = row_i64 * nqo_3 + qo_head_5[2];
                        acc0_3[2] = 0.0f;
                        acc1_3[2] = 0.0f;
                        den_3[2] = 0.0f;
                        lse_max_3[2] = -CUDART_INF_F;
#else
                        dim_1_9[2] = (long long)((pk_2_4 & 127) * 2);
                        qo_head_6[2] = (long long)(first_head + head_5);
                        row_head_6[2] = row_i64 * nqo_4 + qo_head_6[2];
                        acc0_4[2] = 0.0f;
                        acc1_4[2] = 0.0f;
                        den_4[2] = 0.0f;
                        lse_max_4[2] = -CUDART_INF_F;
#endif
                        float _vec_load_62[2];
                        {
#if __CUDA_ARCH__ == 800
                            uint32_t _bf16x2_bits_292;
                            _bf16x2_bits_292 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[2] * 256 + dim_1_6[2]) + 0);
                            (&_vec_load_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_292) << 16);
                            (&_vec_load_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_292) & 0xffff0000u);
#else
                            uint32_t _bf16x2_bits_304;
                            _bf16x2_bits_304 = *reinterpret_cast<const uint32_t*>(gate + (row_head_6[2] * 256 + dim_1_9[2]) + 0);
                            (&_vec_load_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_304) << 16);
                            (&_vec_load_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_304) & 0xffff0000u);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        gate0_3[2] = _vec_load_62[0];
                        gate1_3[2] = _vec_load_62[1];
                        int pk_2_5 = ((p_3 + 384) < (end - 1) ? (p_3 + 384) : (end - 1));
                        valid_0_7[3] = ((end > p_3 + 384) ? 1 : 0);
#else
                        gate0_4[2] = _vec_load_62[0];
                        gate1_4[2] = _vec_load_62[1];
                        int pk_2_5 = ((p_4 + 384) < (end - 1) ? (p_4 + 384) : (end - 1));
                        valid_0_7[3] = ((end > p_4 + 384) ? 1 : 0);
#endif
                        int head_6 = pk_2_5 >> 7;
#if __CUDA_ARCH__ == 800
                        dim_1_6[3] = (long long)((pk_2_5 & 127) * 2);
                        qo_head_5[3] = (long long)(first_head + head_6);
                        row_head_3[3] = row_i64 * nqo_3 + qo_head_5[3];
                        acc0_3[3] = 0.0f;
                        acc1_3[3] = 0.0f;
                        den_3[3] = 0.0f;
                        lse_max_3[3] = -CUDART_INF_F;
#else
                        dim_1_9[3] = (long long)((pk_2_5 & 127) * 2);
                        qo_head_6[3] = (long long)(first_head + head_6);
                        row_head_6[3] = row_i64 * nqo_4 + qo_head_6[3];
                        acc0_4[3] = 0.0f;
                        acc1_4[3] = 0.0f;
                        den_4[3] = 0.0f;
                        lse_max_4[3] = -CUDART_INF_F;
#endif
                        float _vec_load_63[2];
                        {
#if __CUDA_ARCH__ == 800
                            uint32_t _bf16x2_bits_293;
                            _bf16x2_bits_293 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[3] * 256 + dim_1_6[3]) + 0);
                            (&_vec_load_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_293) << 16);
                            (&_vec_load_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_293) & 0xffff0000u);
#else
                            uint32_t _bf16x2_bits_305;
                            _bf16x2_bits_305 = *reinterpret_cast<const uint32_t*>(gate + (row_head_6[3] * 256 + dim_1_9[3]) + 0);
                            (&_vec_load_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_305) << 16);
                            (&_vec_load_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_305) & 0xffff0000u);
#endif
                        }
#if __CUDA_ARCH__ == 800
                        gate0_3[3] = _vec_load_63[0];
                        gate1_3[3] = _vec_load_63[1];
#else
                        gate0_4[3] = _vec_load_63[0];
                        gate1_4[3] = _vec_load_63[1];
#endif
                        #pragma unroll 1
                        for (int base_3 = 0; base_3 < num_splits; base_3 += 4) {
                            lse_3[0] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_48 = (long long)base_3 * split_pitch + row_head_3[0];
#else
                                long long part_row_1_48 = (long long)base_3 * split_pitch + row_head_6[0];
#endif
                                lse_3[0] = partial_lse[part_row_1_48];
                                float _vec_load_64[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_294 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_48 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_64[0] = _v2_294.x;
                                    _vec_load_64[0 + 1] = _v2_294.y;
#else
                                    float2 _v2_306 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_48 * 256 + dim_1_9[0]) + 0);
                                    _vec_load_64[0] = _v2_306.x;
                                    _vec_load_64[0 + 1] = _v2_306.y;
#endif
                                }
                                p0_3[0] = _vec_load_64[0];
                                p1_3[0] = _vec_load_64[1];
                            }
                            lse_3[1] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_49 = (long long)(base_3 + 1) * split_pitch + row_head_3[0];
#else
                                long long part_row_1_49 = (long long)(base_3 + 1) * split_pitch + row_head_6[0];
#endif
                                lse_3[1] = partial_lse[part_row_1_49];
                                float _vec_load_65[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_295 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_49 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_65[0] = _v2_295.x;
                                    _vec_load_65[0 + 1] = _v2_295.y;
#else
                                    float2 _v2_307 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_49 * 256 + dim_1_9[0]) + 0);
                                    _vec_load_65[0] = _v2_307.x;
                                    _vec_load_65[0 + 1] = _v2_307.y;
#endif
                                }
                                p0_3[1] = _vec_load_65[0];
                                p1_3[1] = _vec_load_65[1];
                            }
                            lse_3[2] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_50 = (long long)(base_3 + 2) * split_pitch + row_head_3[0];
#else
                                long long part_row_1_50 = (long long)(base_3 + 2) * split_pitch + row_head_6[0];
#endif
                                lse_3[2] = partial_lse[part_row_1_50];
                                float _vec_load_66[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_296 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_50 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_66[0] = _v2_296.x;
                                    _vec_load_66[0 + 1] = _v2_296.y;
#else
                                    float2 _v2_308 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_50 * 256 + dim_1_9[0]) + 0);
                                    _vec_load_66[0] = _v2_308.x;
                                    _vec_load_66[0 + 1] = _v2_308.y;
#endif
                                }
                                p0_3[2] = _vec_load_66[0];
                                p1_3[2] = _vec_load_66[1];
                            }
                            lse_3[3] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_51 = (long long)(base_3 + 3) * split_pitch + row_head_3[0];
#else
                                long long part_row_1_51 = (long long)(base_3 + 3) * split_pitch + row_head_6[0];
#endif
                                lse_3[3] = partial_lse[part_row_1_51];
                                float _vec_load_67[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_297 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_51 * 256 + dim_1_6[0]) + 0);
                                    _vec_load_67[0] = _v2_297.x;
                                    _vec_load_67[0 + 1] = _v2_297.y;
#else
                                    float2 _v2_309 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_51 * 256 + dim_1_9[0]) + 0);
                                    _vec_load_67[0] = _v2_309.x;
                                    _vec_load_67[0 + 1] = _v2_309.y;
#endif
                                }
                                p0_3[3] = _vec_load_67[0];
                                p1_3[3] = _vec_load_67[1];
                            }
                            lse_3[4] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_52 = (long long)base_3 * split_pitch + row_head_3[1];
#else
                                long long part_row_1_52 = (long long)base_3 * split_pitch + row_head_6[1];
#endif
                                lse_3[4] = partial_lse[part_row_1_52];
                                float _vec_load_68[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_298 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_52 * 256 + dim_1_6[1]) + 0);
                                    _vec_load_68[0] = _v2_298.x;
                                    _vec_load_68[0 + 1] = _v2_298.y;
#else
                                    float2 _v2_310 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_52 * 256 + dim_1_9[1]) + 0);
                                    _vec_load_68[0] = _v2_310.x;
                                    _vec_load_68[0 + 1] = _v2_310.y;
#endif
                                }
                                p0_3[4] = _vec_load_68[0];
                                p1_3[4] = _vec_load_68[1];
                            }
                            lse_3[5] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_53 = (long long)(base_3 + 1) * split_pitch + row_head_3[1];
#else
                                long long part_row_1_53 = (long long)(base_3 + 1) * split_pitch + row_head_6[1];
#endif
                                lse_3[5] = partial_lse[part_row_1_53];
                                float _vec_load_69[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_299 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_53 * 256 + dim_1_6[1]) + 0);
                                    _vec_load_69[0] = _v2_299.x;
                                    _vec_load_69[0 + 1] = _v2_299.y;
#else
                                    float2 _v2_311 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_53 * 256 + dim_1_9[1]) + 0);
                                    _vec_load_69[0] = _v2_311.x;
                                    _vec_load_69[0 + 1] = _v2_311.y;
#endif
                                }
                                p0_3[5] = _vec_load_69[0];
                                p1_3[5] = _vec_load_69[1];
                            }
                            lse_3[6] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_54 = (long long)(base_3 + 2) * split_pitch + row_head_3[1];
#else
                                long long part_row_1_54 = (long long)(base_3 + 2) * split_pitch + row_head_6[1];
#endif
                                lse_3[6] = partial_lse[part_row_1_54];
                                float _vec_load_70[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_300 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_54 * 256 + dim_1_6[1]) + 0);
                                    _vec_load_70[0] = _v2_300.x;
                                    _vec_load_70[0 + 1] = _v2_300.y;
#else
                                    float2 _v2_312 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_54 * 256 + dim_1_9[1]) + 0);
                                    _vec_load_70[0] = _v2_312.x;
                                    _vec_load_70[0 + 1] = _v2_312.y;
#endif
                                }
                                p0_3[6] = _vec_load_70[0];
                                p1_3[6] = _vec_load_70[1];
                            }
                            lse_3[7] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_55 = (long long)(base_3 + 3) * split_pitch + row_head_3[1];
#else
                                long long part_row_1_55 = (long long)(base_3 + 3) * split_pitch + row_head_6[1];
#endif
                                lse_3[7] = partial_lse[part_row_1_55];
                                float _vec_load_71[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_301 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_55 * 256 + dim_1_6[1]) + 0);
                                    _vec_load_71[0] = _v2_301.x;
                                    _vec_load_71[0 + 1] = _v2_301.y;
#else
                                    float2 _v2_313 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_55 * 256 + dim_1_9[1]) + 0);
                                    _vec_load_71[0] = _v2_313.x;
                                    _vec_load_71[0 + 1] = _v2_313.y;
#endif
                                }
                                p0_3[7] = _vec_load_71[0];
                                p1_3[7] = _vec_load_71[1];
                            }
                            lse_3[8] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_56 = (long long)base_3 * split_pitch + row_head_3[2];
#else
                                long long part_row_1_56 = (long long)base_3 * split_pitch + row_head_6[2];
#endif
                                lse_3[8] = partial_lse[part_row_1_56];
                                float _vec_load_72[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_302 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_56 * 256 + dim_1_6[2]) + 0);
                                    _vec_load_72[0] = _v2_302.x;
                                    _vec_load_72[0 + 1] = _v2_302.y;
#else
                                    float2 _v2_314 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_56 * 256 + dim_1_9[2]) + 0);
                                    _vec_load_72[0] = _v2_314.x;
                                    _vec_load_72[0 + 1] = _v2_314.y;
#endif
                                }
                                p0_3[8] = _vec_load_72[0];
                                p1_3[8] = _vec_load_72[1];
                            }
                            lse_3[9] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_57 = (long long)(base_3 + 1) * split_pitch + row_head_3[2];
#else
                                long long part_row_1_57 = (long long)(base_3 + 1) * split_pitch + row_head_6[2];
#endif
                                lse_3[9] = partial_lse[part_row_1_57];
                                float _vec_load_73[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_303 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_57 * 256 + dim_1_6[2]) + 0);
                                    _vec_load_73[0] = _v2_303.x;
                                    _vec_load_73[0 + 1] = _v2_303.y;
#else
                                    float2 _v2_315 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_57 * 256 + dim_1_9[2]) + 0);
                                    _vec_load_73[0] = _v2_315.x;
                                    _vec_load_73[0 + 1] = _v2_315.y;
#endif
                                }
                                p0_3[9] = _vec_load_73[0];
                                p1_3[9] = _vec_load_73[1];
                            }
                            lse_3[10] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_58 = (long long)(base_3 + 2) * split_pitch + row_head_3[2];
#else
                                long long part_row_1_58 = (long long)(base_3 + 2) * split_pitch + row_head_6[2];
#endif
                                lse_3[10] = partial_lse[part_row_1_58];
                                float _vec_load_74[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_304 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_58 * 256 + dim_1_6[2]) + 0);
                                    _vec_load_74[0] = _v2_304.x;
                                    _vec_load_74[0 + 1] = _v2_304.y;
#else
                                    float2 _v2_316 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_58 * 256 + dim_1_9[2]) + 0);
                                    _vec_load_74[0] = _v2_316.x;
                                    _vec_load_74[0 + 1] = _v2_316.y;
#endif
                                }
                                p0_3[10] = _vec_load_74[0];
                                p1_3[10] = _vec_load_74[1];
                            }
                            lse_3[11] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_59 = (long long)(base_3 + 3) * split_pitch + row_head_3[2];
#else
                                long long part_row_1_59 = (long long)(base_3 + 3) * split_pitch + row_head_6[2];
#endif
                                lse_3[11] = partial_lse[part_row_1_59];
                                float _vec_load_75[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_305 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_59 * 256 + dim_1_6[2]) + 0);
                                    _vec_load_75[0] = _v2_305.x;
                                    _vec_load_75[0 + 1] = _v2_305.y;
#else
                                    float2 _v2_317 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_59 * 256 + dim_1_9[2]) + 0);
                                    _vec_load_75[0] = _v2_317.x;
                                    _vec_load_75[0 + 1] = _v2_317.y;
#endif
                                }
                                p0_3[11] = _vec_load_75[0];
                                p1_3[11] = _vec_load_75[1];
                            }
                            lse_3[12] = -CUDART_INF_F;
                            if (base_3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_60 = (long long)base_3 * split_pitch + row_head_3[3];
#else
                                long long part_row_1_60 = (long long)base_3 * split_pitch + row_head_6[3];
#endif
                                lse_3[12] = partial_lse[part_row_1_60];
                                float _vec_load_76[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_306 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_60 * 256 + dim_1_6[3]) + 0);
                                    _vec_load_76[0] = _v2_306.x;
                                    _vec_load_76[0 + 1] = _v2_306.y;
#else
                                    float2 _v2_318 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_60 * 256 + dim_1_9[3]) + 0);
                                    _vec_load_76[0] = _v2_318.x;
                                    _vec_load_76[0 + 1] = _v2_318.y;
#endif
                                }
                                p0_3[12] = _vec_load_76[0];
                                p1_3[12] = _vec_load_76[1];
                            }
                            lse_3[13] = -CUDART_INF_F;
                            if (base_3 + 1 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_61 = (long long)(base_3 + 1) * split_pitch + row_head_3[3];
#else
                                long long part_row_1_61 = (long long)(base_3 + 1) * split_pitch + row_head_6[3];
#endif
                                lse_3[13] = partial_lse[part_row_1_61];
                                float _vec_load_77[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_307 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_61 * 256 + dim_1_6[3]) + 0);
                                    _vec_load_77[0] = _v2_307.x;
                                    _vec_load_77[0 + 1] = _v2_307.y;
#else
                                    float2 _v2_319 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_61 * 256 + dim_1_9[3]) + 0);
                                    _vec_load_77[0] = _v2_319.x;
                                    _vec_load_77[0 + 1] = _v2_319.y;
#endif
                                }
                                p0_3[13] = _vec_load_77[0];
                                p1_3[13] = _vec_load_77[1];
                            }
                            lse_3[14] = -CUDART_INF_F;
                            if (base_3 + 2 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_62 = (long long)(base_3 + 2) * split_pitch + row_head_3[3];
#else
                                long long part_row_1_62 = (long long)(base_3 + 2) * split_pitch + row_head_6[3];
#endif
                                lse_3[14] = partial_lse[part_row_1_62];
                                float _vec_load_78[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_308 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_62 * 256 + dim_1_6[3]) + 0);
                                    _vec_load_78[0] = _v2_308.x;
                                    _vec_load_78[0 + 1] = _v2_308.y;
#else
                                    float2 _v2_320 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_62 * 256 + dim_1_9[3]) + 0);
                                    _vec_load_78[0] = _v2_320.x;
                                    _vec_load_78[0 + 1] = _v2_320.y;
#endif
                                }
                                p0_3[14] = _vec_load_78[0];
                                p1_3[14] = _vec_load_78[1];
                            }
                            lse_3[15] = -CUDART_INF_F;
                            if (base_3 + 3 < num_splits) {
#if __CUDA_ARCH__ == 800
                                long long part_row_1_63 = (long long)(base_3 + 3) * split_pitch + row_head_3[3];
#else
                                long long part_row_1_63 = (long long)(base_3 + 3) * split_pitch + row_head_6[3];
#endif
                                lse_3[15] = partial_lse[part_row_1_63];
                                float _vec_load_79[2];
                                {
#if __CUDA_ARCH__ == 800
                                    float2 _v2_309 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_63 * 256 + dim_1_6[3]) + 0);
                                    _vec_load_79[0] = _v2_309.x;
                                    _vec_load_79[0 + 1] = _v2_309.y;
#else
                                    float2 _v2_321 = *reinterpret_cast<const float2*>(partial_out + (part_row_1_63 * 256 + dim_1_9[3]) + 0);
                                    _vec_load_79[0] = _v2_321.x;
                                    _vec_load_79[0 + 1] = _v2_321.y;
#endif
                                }
                                p0_3[15] = _vec_load_79[0];
                                p1_3[15] = _vec_load_79[1];
                            }
#if __CUDA_ARCH__ == 800
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
                            bool live_15_48 = base_3 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#else
                            lse_max_4[0] = fmaxf(lse_max_4[0], lse_3[0]);
                            lse_max_4[0] = fmaxf(lse_max_4[0], lse_3[1]);
                            lse_max_4[0] = fmaxf(lse_max_4[0], lse_3[2]);
                            lse_max_4[0] = fmaxf(lse_max_4[0], lse_3[3]);
                            lse_max_4[1] = fmaxf(lse_max_4[1], lse_3[4]);
                            lse_max_4[1] = fmaxf(lse_max_4[1], lse_3[5]);
                            lse_max_4[1] = fmaxf(lse_max_4[1], lse_3[6]);
                            lse_max_4[1] = fmaxf(lse_max_4[1], lse_3[7]);
                            lse_max_4[2] = fmaxf(lse_max_4[2], lse_3[8]);
                            lse_max_4[2] = fmaxf(lse_max_4[2], lse_3[9]);
                            lse_max_4[2] = fmaxf(lse_max_4[2], lse_3[10]);
                            lse_max_4[2] = fmaxf(lse_max_4[2], lse_3[11]);
                            lse_max_4[3] = fmaxf(lse_max_4[3], lse_3[12]);
                            lse_max_4[3] = fmaxf(lse_max_4[3], lse_3[13]);
                            lse_max_4[3] = fmaxf(lse_max_4[3], lse_3[14]);
                            lse_max_4[3] = fmaxf(lse_max_4[3], lse_3[15]);
                            bool live_15_48 = base_3 < num_splits && lse_max_4[0] > -CUDART_INF_F;
#endif
                            if (live_15_48) {
#if __CUDA_ARCH__ == 800
                                float weight_1_48 = approx_exp2(lse_3[0] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_48;
                                acc0_3[0] = __fmaf_rn(weight_1_48, p0_3[0], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_48, p1_3[0], acc1_3[0]);
#else
                                float weight_1_48 = approx_exp2(lse_3[0] - lse_max_4[0]);
                                den_4[0] = den_4[0] + weight_1_48;
                                acc0_4[0] = __fmaf_rn(weight_1_48, p0_3[0], acc0_4[0]);
                                acc1_4[0] = __fmaf_rn(weight_1_48, p1_3[0], acc1_4[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_49 = base_3 + 1 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#else
                            bool live_15_49 = base_3 + 1 < num_splits && lse_max_4[0] > -CUDART_INF_F;
#endif
                            if (live_15_49) {
#if __CUDA_ARCH__ == 800
                                float weight_1_49 = approx_exp2(lse_3[1] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_49;
                                acc0_3[0] = __fmaf_rn(weight_1_49, p0_3[1], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_49, p1_3[1], acc1_3[0]);
#else
                                float weight_1_49 = approx_exp2(lse_3[1] - lse_max_4[0]);
                                den_4[0] = den_4[0] + weight_1_49;
                                acc0_4[0] = __fmaf_rn(weight_1_49, p0_3[1], acc0_4[0]);
                                acc1_4[0] = __fmaf_rn(weight_1_49, p1_3[1], acc1_4[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_50 = base_3 + 2 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#else
                            bool live_15_50 = base_3 + 2 < num_splits && lse_max_4[0] > -CUDART_INF_F;
#endif
                            if (live_15_50) {
#if __CUDA_ARCH__ == 800
                                float weight_1_50 = approx_exp2(lse_3[2] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_50;
                                acc0_3[0] = __fmaf_rn(weight_1_50, p0_3[2], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_50, p1_3[2], acc1_3[0]);
#else
                                float weight_1_50 = approx_exp2(lse_3[2] - lse_max_4[0]);
                                den_4[0] = den_4[0] + weight_1_50;
                                acc0_4[0] = __fmaf_rn(weight_1_50, p0_3[2], acc0_4[0]);
                                acc1_4[0] = __fmaf_rn(weight_1_50, p1_3[2], acc1_4[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_51 = base_3 + 3 < num_splits && lse_max_3[0] > -CUDART_INF_F;
#else
                            bool live_15_51 = base_3 + 3 < num_splits && lse_max_4[0] > -CUDART_INF_F;
#endif
                            if (live_15_51) {
#if __CUDA_ARCH__ == 800
                                float weight_1_51 = approx_exp2(lse_3[3] - lse_max_3[0]);
                                den_3[0] = den_3[0] + weight_1_51;
                                acc0_3[0] = __fmaf_rn(weight_1_51, p0_3[3], acc0_3[0]);
                                acc1_3[0] = __fmaf_rn(weight_1_51, p1_3[3], acc1_3[0]);
#else
                                float weight_1_51 = approx_exp2(lse_3[3] - lse_max_4[0]);
                                den_4[0] = den_4[0] + weight_1_51;
                                acc0_4[0] = __fmaf_rn(weight_1_51, p0_3[3], acc0_4[0]);
                                acc1_4[0] = __fmaf_rn(weight_1_51, p1_3[3], acc1_4[0]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_52 = base_3 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#else
                            bool live_15_52 = base_3 < num_splits && lse_max_4[1] > -CUDART_INF_F;
#endif
                            if (live_15_52) {
#if __CUDA_ARCH__ == 800
                                float weight_1_52 = approx_exp2(lse_3[4] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_52;
                                acc0_3[1] = __fmaf_rn(weight_1_52, p0_3[4], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_52, p1_3[4], acc1_3[1]);
#else
                                float weight_1_52 = approx_exp2(lse_3[4] - lse_max_4[1]);
                                den_4[1] = den_4[1] + weight_1_52;
                                acc0_4[1] = __fmaf_rn(weight_1_52, p0_3[4], acc0_4[1]);
                                acc1_4[1] = __fmaf_rn(weight_1_52, p1_3[4], acc1_4[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_53 = base_3 + 1 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#else
                            bool live_15_53 = base_3 + 1 < num_splits && lse_max_4[1] > -CUDART_INF_F;
#endif
                            if (live_15_53) {
#if __CUDA_ARCH__ == 800
                                float weight_1_53 = approx_exp2(lse_3[5] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_53;
                                acc0_3[1] = __fmaf_rn(weight_1_53, p0_3[5], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_53, p1_3[5], acc1_3[1]);
#else
                                float weight_1_53 = approx_exp2(lse_3[5] - lse_max_4[1]);
                                den_4[1] = den_4[1] + weight_1_53;
                                acc0_4[1] = __fmaf_rn(weight_1_53, p0_3[5], acc0_4[1]);
                                acc1_4[1] = __fmaf_rn(weight_1_53, p1_3[5], acc1_4[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_54 = base_3 + 2 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#else
                            bool live_15_54 = base_3 + 2 < num_splits && lse_max_4[1] > -CUDART_INF_F;
#endif
                            if (live_15_54) {
#if __CUDA_ARCH__ == 800
                                float weight_1_54 = approx_exp2(lse_3[6] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_54;
                                acc0_3[1] = __fmaf_rn(weight_1_54, p0_3[6], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_54, p1_3[6], acc1_3[1]);
#else
                                float weight_1_54 = approx_exp2(lse_3[6] - lse_max_4[1]);
                                den_4[1] = den_4[1] + weight_1_54;
                                acc0_4[1] = __fmaf_rn(weight_1_54, p0_3[6], acc0_4[1]);
                                acc1_4[1] = __fmaf_rn(weight_1_54, p1_3[6], acc1_4[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_55 = base_3 + 3 < num_splits && lse_max_3[1] > -CUDART_INF_F;
#else
                            bool live_15_55 = base_3 + 3 < num_splits && lse_max_4[1] > -CUDART_INF_F;
#endif
                            if (live_15_55) {
#if __CUDA_ARCH__ == 800
                                float weight_1_55 = approx_exp2(lse_3[7] - lse_max_3[1]);
                                den_3[1] = den_3[1] + weight_1_55;
                                acc0_3[1] = __fmaf_rn(weight_1_55, p0_3[7], acc0_3[1]);
                                acc1_3[1] = __fmaf_rn(weight_1_55, p1_3[7], acc1_3[1]);
#else
                                float weight_1_55 = approx_exp2(lse_3[7] - lse_max_4[1]);
                                den_4[1] = den_4[1] + weight_1_55;
                                acc0_4[1] = __fmaf_rn(weight_1_55, p0_3[7], acc0_4[1]);
                                acc1_4[1] = __fmaf_rn(weight_1_55, p1_3[7], acc1_4[1]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_56 = base_3 < num_splits && lse_max_3[2] > -CUDART_INF_F;
#else
                            bool live_15_56 = base_3 < num_splits && lse_max_4[2] > -CUDART_INF_F;
#endif
                            if (live_15_56) {
#if __CUDA_ARCH__ == 800
                                float weight_1_56 = approx_exp2(lse_3[8] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_1_56;
                                acc0_3[2] = __fmaf_rn(weight_1_56, p0_3[8], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_1_56, p1_3[8], acc1_3[2]);
#else
                                float weight_1_56 = approx_exp2(lse_3[8] - lse_max_4[2]);
                                den_4[2] = den_4[2] + weight_1_56;
                                acc0_4[2] = __fmaf_rn(weight_1_56, p0_3[8], acc0_4[2]);
                                acc1_4[2] = __fmaf_rn(weight_1_56, p1_3[8], acc1_4[2]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_57 = base_3 + 1 < num_splits && lse_max_3[2] > -CUDART_INF_F;
#else
                            bool live_15_57 = base_3 + 1 < num_splits && lse_max_4[2] > -CUDART_INF_F;
#endif
                            if (live_15_57) {
#if __CUDA_ARCH__ == 800
                                float weight_1_57 = approx_exp2(lse_3[9] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_1_57;
                                acc0_3[2] = __fmaf_rn(weight_1_57, p0_3[9], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_1_57, p1_3[9], acc1_3[2]);
#else
                                float weight_1_57 = approx_exp2(lse_3[9] - lse_max_4[2]);
                                den_4[2] = den_4[2] + weight_1_57;
                                acc0_4[2] = __fmaf_rn(weight_1_57, p0_3[9], acc0_4[2]);
                                acc1_4[2] = __fmaf_rn(weight_1_57, p1_3[9], acc1_4[2]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_58 = base_3 + 2 < num_splits && lse_max_3[2] > -CUDART_INF_F;
#else
                            bool live_15_58 = base_3 + 2 < num_splits && lse_max_4[2] > -CUDART_INF_F;
#endif
                            if (live_15_58) {
#if __CUDA_ARCH__ == 800
                                float weight_1_58 = approx_exp2(lse_3[10] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_1_58;
                                acc0_3[2] = __fmaf_rn(weight_1_58, p0_3[10], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_1_58, p1_3[10], acc1_3[2]);
#else
                                float weight_1_58 = approx_exp2(lse_3[10] - lse_max_4[2]);
                                den_4[2] = den_4[2] + weight_1_58;
                                acc0_4[2] = __fmaf_rn(weight_1_58, p0_3[10], acc0_4[2]);
                                acc1_4[2] = __fmaf_rn(weight_1_58, p1_3[10], acc1_4[2]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_59 = base_3 + 3 < num_splits && lse_max_3[2] > -CUDART_INF_F;
#else
                            bool live_15_59 = base_3 + 3 < num_splits && lse_max_4[2] > -CUDART_INF_F;
#endif
                            if (live_15_59) {
#if __CUDA_ARCH__ == 800
                                float weight_1_59 = approx_exp2(lse_3[11] - lse_max_3[2]);
                                den_3[2] = den_3[2] + weight_1_59;
                                acc0_3[2] = __fmaf_rn(weight_1_59, p0_3[11], acc0_3[2]);
                                acc1_3[2] = __fmaf_rn(weight_1_59, p1_3[11], acc1_3[2]);
#else
                                float weight_1_59 = approx_exp2(lse_3[11] - lse_max_4[2]);
                                den_4[2] = den_4[2] + weight_1_59;
                                acc0_4[2] = __fmaf_rn(weight_1_59, p0_3[11], acc0_4[2]);
                                acc1_4[2] = __fmaf_rn(weight_1_59, p1_3[11], acc1_4[2]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_60 = base_3 < num_splits && lse_max_3[3] > -CUDART_INF_F;
#else
                            bool live_15_60 = base_3 < num_splits && lse_max_4[3] > -CUDART_INF_F;
#endif
                            if (live_15_60) {
#if __CUDA_ARCH__ == 800
                                float weight_1_60 = approx_exp2(lse_3[12] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_1_60;
                                acc0_3[3] = __fmaf_rn(weight_1_60, p0_3[12], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_1_60, p1_3[12], acc1_3[3]);
#else
                                float weight_1_60 = approx_exp2(lse_3[12] - lse_max_4[3]);
                                den_4[3] = den_4[3] + weight_1_60;
                                acc0_4[3] = __fmaf_rn(weight_1_60, p0_3[12], acc0_4[3]);
                                acc1_4[3] = __fmaf_rn(weight_1_60, p1_3[12], acc1_4[3]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_61 = base_3 + 1 < num_splits && lse_max_3[3] > -CUDART_INF_F;
#else
                            bool live_15_61 = base_3 + 1 < num_splits && lse_max_4[3] > -CUDART_INF_F;
#endif
                            if (live_15_61) {
#if __CUDA_ARCH__ == 800
                                float weight_1_61 = approx_exp2(lse_3[13] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_1_61;
                                acc0_3[3] = __fmaf_rn(weight_1_61, p0_3[13], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_1_61, p1_3[13], acc1_3[3]);
#else
                                float weight_1_61 = approx_exp2(lse_3[13] - lse_max_4[3]);
                                den_4[3] = den_4[3] + weight_1_61;
                                acc0_4[3] = __fmaf_rn(weight_1_61, p0_3[13], acc0_4[3]);
                                acc1_4[3] = __fmaf_rn(weight_1_61, p1_3[13], acc1_4[3]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_62 = base_3 + 2 < num_splits && lse_max_3[3] > -CUDART_INF_F;
#else
                            bool live_15_62 = base_3 + 2 < num_splits && lse_max_4[3] > -CUDART_INF_F;
#endif
                            if (live_15_62) {
#if __CUDA_ARCH__ == 800
                                float weight_1_62 = approx_exp2(lse_3[14] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_1_62;
                                acc0_3[3] = __fmaf_rn(weight_1_62, p0_3[14], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_1_62, p1_3[14], acc1_3[3]);
#else
                                float weight_1_62 = approx_exp2(lse_3[14] - lse_max_4[3]);
                                den_4[3] = den_4[3] + weight_1_62;
                                acc0_4[3] = __fmaf_rn(weight_1_62, p0_3[14], acc0_4[3]);
                                acc1_4[3] = __fmaf_rn(weight_1_62, p1_3[14], acc1_4[3]);
#endif
                            }
#if __CUDA_ARCH__ == 800
                            bool live_15_63 = base_3 + 3 < num_splits && lse_max_3[3] > -CUDART_INF_F;
#else
                            bool live_15_63 = base_3 + 3 < num_splits && lse_max_4[3] > -CUDART_INF_F;
#endif
                            if (live_15_63) {
#if __CUDA_ARCH__ == 800
                                float weight_1_63 = approx_exp2(lse_3[15] - lse_max_3[3]);
                                den_3[3] = den_3[3] + weight_1_63;
                                acc0_3[3] = __fmaf_rn(weight_1_63, p0_3[15], acc0_3[3]);
                                acc1_3[3] = __fmaf_rn(weight_1_63, p1_3[15], acc1_3[3]);
#else
                                float weight_1_63 = approx_exp2(lse_3[15] - lse_max_4[3]);
                                den_4[3] = den_4[3] + weight_1_63;
                                acc0_4[3] = __fmaf_rn(weight_1_63, p0_3[15], acc0_4[3]);
                                acc1_4[3] = __fmaf_rn(weight_1_63, p1_3[15], acc1_4[3]);
#endif
                            }
                        }
                        if (valid_0_7[0] != 0) {
#if __CUDA_ARCH__ == 800
                            float inv_6 = 0.0f;
                            if (den_3[0] > 0.0f) {
                                float _rcp_310 = approx_rcp(den_3[0]);
                                inv_6 = _rcp_310;
#else
                            float inv_7 = 0.0f;
                            if (den_4[0] > 0.0f) {
                                float _rcp_322 = approx_rcp(den_4[0]);
                                inv_7 = _rcp_322;
#endif
                            }
#if __CUDA_ARCH__ == 800
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
                            long long out_offset_4 = row_i64 * out_stride_row + qo_head_5[0] * out_stride_head + dim_1_6[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_4) + (0)) = word_3[0];
#else
                            float attn0_5 = (float)(__nv_bfloat16)(acc0_4[0] * inv_7);
                            float attn1_5 = (float)(__nv_bfloat16)(acc1_4[0] * inv_7);
                            float _exp2_324 = approx_exp2((-gate0_4[0]) * 1.442695f);
                            float _rcp_323 = approx_rcp(1.0f + _exp2_324);
                            float result0_5 = attn0_5 * _rcp_323;
                            float _exp2_326 = approx_exp2((-gate1_4[0]) * 1.442695f);
                            float _rcp_325 = approx_rcp(1.0f + _exp2_326);
                            float result1_5 = attn1_5 * _rcp_325;
                            __nv_bfloat162 _bit_cast_src_327 = __float22bfloat162_rn(make_float2(result0_5, result1_5));
                            word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_327)[0];
                            long long out_offset_5 = row_i64 * out_stride_row + qo_head_6[0] * out_stride_head + dim_1_9[0];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_5) + (0)) = word_4[0];
#endif
                        }
                        if (valid_0_7[1] != 0) {
#if __CUDA_ARCH__ == 800
                            float inv_7 = 0.0f;
                            if (den_3[1] > 0.0f) {
                                float _rcp_316 = approx_rcp(den_3[1]);
                                inv_7 = _rcp_316;
#else
                            float inv_8 = 0.0f;
                            if (den_4[1] > 0.0f) {
                                float _rcp_328 = approx_rcp(den_4[1]);
                                inv_8 = _rcp_328;
#endif
                            }
#if __CUDA_ARCH__ == 800
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
                            long long out_offset_5 = row_i64 * out_stride_row + qo_head_5[1] * out_stride_head + dim_1_6[1];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_5) + (0)) = word_3[0];
#else
                            float attn0_6 = (float)(__nv_bfloat16)(acc0_4[1] * inv_8);
                            float attn1_6 = (float)(__nv_bfloat16)(acc1_4[1] * inv_8);
                            float _exp2_330 = approx_exp2((-gate0_4[1]) * 1.442695f);
                            float _rcp_329 = approx_rcp(1.0f + _exp2_330);
                            float result0_6 = attn0_6 * _rcp_329;
                            float _exp2_332 = approx_exp2((-gate1_4[1]) * 1.442695f);
                            float _rcp_331 = approx_rcp(1.0f + _exp2_332);
                            float result1_6 = attn1_6 * _rcp_331;
                            __nv_bfloat162 _bit_cast_src_333 = __float22bfloat162_rn(make_float2(result0_6, result1_6));
                            word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_333)[0];
                            long long out_offset_6 = row_i64 * out_stride_row + qo_head_6[1] * out_stride_head + dim_1_9[1];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_6) + (0)) = word_4[0];
#endif
                        }
                        if (valid_0_7[2] != 0) {
#if __CUDA_ARCH__ == 800
                            float inv_8 = 0.0f;
                            if (den_3[2] > 0.0f) {
                                float _rcp_322 = approx_rcp(den_3[2]);
                                inv_8 = _rcp_322;
#else
                            float inv_9 = 0.0f;
                            if (den_4[2] > 0.0f) {
                                float _rcp_334 = approx_rcp(den_4[2]);
                                inv_9 = _rcp_334;
#endif
                            }
#if __CUDA_ARCH__ == 800
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
                            long long out_offset_6 = row_i64 * out_stride_row + qo_head_5[2] * out_stride_head + dim_1_6[2];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_6) + (0)) = word_3[0];
#else
                            float attn0_7 = (float)(__nv_bfloat16)(acc0_4[2] * inv_9);
                            float attn1_7 = (float)(__nv_bfloat16)(acc1_4[2] * inv_9);
                            float _exp2_336 = approx_exp2((-gate0_4[2]) * 1.442695f);
                            float _rcp_335 = approx_rcp(1.0f + _exp2_336);
                            float result0_7 = attn0_7 * _rcp_335;
                            float _exp2_338 = approx_exp2((-gate1_4[2]) * 1.442695f);
                            float _rcp_337 = approx_rcp(1.0f + _exp2_338);
                            float result1_7 = attn1_7 * _rcp_337;
                            __nv_bfloat162 _bit_cast_src_339 = __float22bfloat162_rn(make_float2(result0_7, result1_7));
                            word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_339)[0];
                            long long out_offset_7 = row_i64 * out_stride_row + qo_head_6[2] * out_stride_head + dim_1_9[2];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_7) + (0)) = word_4[0];
#endif
                        }
                        if (valid_0_7[3] != 0) {
#if __CUDA_ARCH__ == 800
                            float inv_9 = 0.0f;
                            if (den_3[3] > 0.0f) {
                                float _rcp_328 = approx_rcp(den_3[3]);
                                inv_9 = _rcp_328;
#else
                            float inv_10 = 0.0f;
                            if (den_4[3] > 0.0f) {
                                float _rcp_340 = approx_rcp(den_4[3]);
                                inv_10 = _rcp_340;
#endif
                            }
#if __CUDA_ARCH__ == 800
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
                            long long out_offset_7 = row_i64 * out_stride_row + qo_head_5[3] * out_stride_head + dim_1_6[3];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_7) + (0)) = word_3[0];
#else
                            float attn0_8 = (float)(__nv_bfloat16)(acc0_4[3] * inv_10);
                            float attn1_8 = (float)(__nv_bfloat16)(acc1_4[3] * inv_10);
                            float _exp2_342 = approx_exp2((-gate0_4[3]) * 1.442695f);
                            float _rcp_341 = approx_rcp(1.0f + _exp2_342);
                            float result0_8 = attn0_8 * _rcp_341;
                            float _exp2_344 = approx_exp2((-gate1_4[3]) * 1.442695f);
                            float _rcp_343 = approx_rcp(1.0f + _exp2_344);
                            float result1_8 = attn1_8 * _rcp_343;
                            __nv_bfloat162 _bit_cast_src_345 = __float22bfloat162_rn(make_float2(result0_8, result1_8));
                            word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_345)[0];
                            long long out_offset_8 = row_i64 * out_stride_row + qo_head_6[3] * out_stride_head + dim_1_9[3];
                            *(reinterpret_cast<unsigned int*>(out + out_offset_8) + (0)) = word_4[0];
#endif
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
