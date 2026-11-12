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
#define SMEM_IP_OFF 0
#define SMEM_IP_STAGE_BYTES 388
#define SMEM_IP_STRIDE 388
#define SMEM_LENS_OFF 400
#define SMEM_LENS_STAGE_BYTES 384
#define SMEM_LENS_STRIDE 384
#define SMEM_QKW_OFF 784
#define SMEM_QKW_STAGE_BYTES 1024
#define SMEM_QKW_STRIDE 1024
#define SMEM_TOTAL 0
#define THREADS 96
#define LAUNCH_MIN_BLOCKS 10

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(96, LAUNCH_MIN_BLOCKS) void
kernel_cake_fused_qk_rope_fp8_append_bc6c52ab41114c98fa85(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cos_sin, const int* __restrict__ seq_lens, const int* __restrict__ q_indptr, const int* __restrict__ page_indices, const float* __restrict__ q_norm_weight, const float* __restrict__ k_norm_weight, const float* __restrict__ k_scale, const float* __restrict__ v_scale, const float* __restrict__ q_scale_inv, uint8_t* __restrict__ out_q, uint8_t* key_cache, uint8_t* value_cache, uint8_t* out_k, uint8_t* out_v, float* __restrict__ q_scale, int* __restrict__ split_k_flag, int num_rows, int num_requests, int num_q_heads, int num_kv_heads, int page_size, int max_pages_per_request, int max_seqlen, int max_seqlen_aligned, int quant_policy, int norm_policy, bool is_prefill, bool has_out_kv, float upper_max)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[1808];

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* ip = reinterpret_cast<int*>(smem_static_raw + 0);
    const int ip_addr = (int)(unsigned long long)__cvta_generic_to_shared(ip);
    int* lens = reinterpret_cast<int*>(smem_static_raw + 400);
    const int lens_addr = (int)(unsigned long long)__cvta_generic_to_shared(lens);
    float* qkw = reinterpret_cast<float*>(smem_static_raw + 784);
    const int qkw_addr = (int)(unsigned long long)__cvta_generic_to_shared(qkw);

    // === Task calls (declaration order) ===
    int tid_0 = tid;
    int slot = tid_0 / 8;
    int lane_1 = tid_0 % 8;
    int grp_base = tid_0 % 32 - lane_1;
    unsigned int grp_mask = (unsigned int)(255 << grp_base);
    int q_per_kv = num_q_heads / num_kv_heads;
    bool dyn_prefill = quant_policy == 1 && is_prefill;
    int width = (num_q_heads + 2 * num_kv_heads) * 128;
    bool is_q = slot < q_per_kv;
    bool is_k = slot == 8;
    bool is_v = slot == 9;
    bool active = is_q || is_k || is_v;
    int lane_lo = ((is_v) ? lane_1 * 16 : lane_1 * 8);
    int lane_hi = ((is_v) ? lane_1 * 16 + 8 : 64 + lane_1 * 8);
    if (norm_policy != 0 && tid_0 < 64) {
        float w_cp[4];
        if (tid_0 < 32) {
            {
                unsigned _v4_0_0;
                unsigned _v4_0_1;
                unsigned _v4_0_2;
                unsigned _v4_0_3;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(q_norm_weight + (tid_0 * 4) + (0))) : "memory");
                w_cp[0 + 0] = __uint_as_float(_v4_0_0);
                w_cp[0 + 1] = __uint_as_float(_v4_0_1);
                w_cp[0 + 2] = __uint_as_float(_v4_0_2);
                w_cp[0 + 3] = __uint_as_float(_v4_0_3);
            }
        } else {
            {
                unsigned _v4_1_0;
                unsigned _v4_1_1;
                unsigned _v4_1_2;
                unsigned _v4_1_3;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(k_norm_weight + ((tid_0 - 32) * 4) + (0))) : "memory");
                w_cp[0 + 0] = __uint_as_float(_v4_1_0);
                w_cp[0 + 1] = __uint_as_float(_v4_1_1);
                w_cp[0 + 2] = __uint_as_float(_v4_1_2);
                w_cp[0 + 3] = __uint_as_float(_v4_1_3);
            }
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(qkw_addr + (unsigned int)(tid_0 * 16)), "f"(w_cp[0]), "f"(w_cp[1]), "f"(w_cp[2]), "f"(w_cp[3]) : "memory");
    }
    unsigned int w_base = qkw_addr + (unsigned int)(((is_q) ? 0 : 512) + lane_1 * 32);
    float mult_fixed = 1.0f;
    if (is_k) {
        float _rcp_2 = __frcp_rn(k_scale[0]);
        mult_fixed = _rcp_2;
    }
    if (is_v) {
        float _rcp_3 = __frcp_rn(v_scale[0]);
        mult_fixed = _rcp_3;
    }
    if (is_q && quant_policy == 2) {
        mult_fixed = q_scale_inv[0];
    }
    bool any_bad = 0;
    if (tid_0 < num_requests) {
        int begin = q_indptr[tid_0];
        int end = q_indptr[tid_0 + 1];
        int seq_len_r = seq_lens[tid_0];
        bool bad = begin < 0 || end < begin || end > num_rows;
        if (dyn_prefill && end - begin > max_seqlen) {
            bad = 1;
        }
        if (tid_0 == 0 && begin != 0) {
            bad = 1;
        }
        if (tid_0 == num_requests - 1 && end != num_rows) {
            bad = 1;
        }
        if (bad) {
            any_bad = 1;
        }
        ip[tid_0 + 1] = end;
        lens[tid_0] = seq_len_r;
        if (tid_0 == 0) {
            ip[0] = begin;
        }
    }
    int bs_n = num_requests;
    int bs_rounds = 0;
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    if (bs_n > 1) {
        bs_rounds = bs_rounds + 1;
        bs_n = bs_n - (bs_n >> 1);
    }
    uint32_t invalid = __syncthreads_or(any_bad);
    int cta = blockIdx.x;
    int per_head = ((gridDim.x / num_kv_heads) > (1) ? (gridDim.x / num_kv_heads) : (1));
    int j_first = cta / num_kv_heads;
    int kv_head = cta - j_first * num_kv_heads;
    int n_items = num_rows + num_requests;
    if (cta >= per_head * num_kv_heads) {
        j_first = n_items;
    }
    #pragma unroll 1
    for (int it_j = j_first; it_j < n_items; it_j += per_head) {
        if (it_j < num_requests) {
            int req_id = ((it_j) > (0) ? (it_j) : (0));
            if (tid_0 == 0) {
                int flag_val = ((invalid != 0) ? -1 : 0);
                *(reinterpret_cast<int*>(split_k_flag + (req_id * num_kv_heads + kv_head)) + (0)) = flag_val;
            }
            if (invalid == 0) {
                int last_pos = lens[req_id] - 1;
                if (last_pos >= 0) {
                    int last_slot = last_pos / page_size;
                    int zero_from = last_pos - last_slot * page_size + 1;
                    if (zero_from < page_size) {
                        int phys_last = page_indices[req_id * max_pages_per_request + last_slot];
                        long long phys_last64 = phys_last;
                        unsigned int zero_w[4];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            zero_w[j] = 0;
                        }
                        int n_rows = page_size - zero_from;
                        int total_items = n_rows * 16;
                        #pragma unroll 1
                        for (int item = tid_0; item < total_items; item += 96) {
                            int r = item / 16;
                            int sub = item - r * 16;
                            int chunk = sub % 8;
                            long long base = (phys_last64 * (long long)page_size + (long long)zero_from + (long long)r) * (long long)(num_kv_heads * 128) + (long long)(kv_head * 128) + (long long)(chunk * 16);
                            if (sub < 8) {
                                reinterpret_cast<int4*>(key_cache + base)[0] = reinterpret_cast<int4*>(zero_w)[0];
                            } else {
                                reinterpret_cast<int4*>(value_cache + base)[0] = reinterpret_cast<int4*>(zero_w)[0];
                            }
                        }
                    }
                }
            }
        } else if (invalid == 0 && active) {
            int row = it_j - num_requests;
            int bs_base = 0;
            int bs_len = num_requests;
            #pragma unroll 1
            for (int bs_k = 0; bs_k < bs_rounds; bs_k++) {
                int bs_half = bs_len >> 1;
                if (ip[bs_base + bs_half] <= row) {
                    bs_base = bs_base + bs_half;
                }
                bs_len = bs_len - bs_half;
            }
            int batch_i = ((ip[bs_base + 1] <= row) ? bs_base + 1 : bs_base);
            int req_begin = ip[batch_i];
            int req_end = ip[batch_i + 1];
            int pos = row + lens[batch_i] - req_end;
            int tok = row - req_begin;
            int head_in_row = ((is_q) ? kv_head * q_per_kv + slot : ((is_k) ? num_q_heads + kv_head : num_q_heads + num_kv_heads + kv_head));
            unsigned int x_lo_w[4];
            unsigned int x_hi_w[4];
            long long src_base = (long long)row * (long long)width + (long long)(head_in_row * 128);
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(qkv + (src_base + (long long)lane_lo) + 0);
                uint4* _vdst_4 = reinterpret_cast<uint4*>(&x_lo_w[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)) : "memory");
                }
            }
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(qkv + (src_base + (long long)lane_hi) + 0);
                uint4* _vdst_5 = reinterpret_cast<uint4*>(&x_hi_w[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)) : "memory");
                }
            }
            float c_lo[8];
            float c_hi[8];
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                c_lo[j_1] = 1.0f;
                c_hi[j_1] = 0.0f;
            }
            long long cache_row = 0;
            if (is_q || is_k) {
                long long cs_base = (long long)pos * 128 + (long long)(lane_1 * 8);
                {
                    unsigned _v4_6_0;
                    unsigned _v4_6_1;
                    unsigned _v4_6_2;
                    unsigned _v4_6_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(cos_sin + cs_base + (0))) : "memory");
                    c_lo[0 + 0] = __uint_as_float(_v4_6_0);
                    c_lo[0 + 1] = __uint_as_float(_v4_6_1);
                    c_lo[0 + 2] = __uint_as_float(_v4_6_2);
                    c_lo[0 + 3] = __uint_as_float(_v4_6_3);
                }
                {
                    unsigned _v4_7_0;
                    unsigned _v4_7_1;
                    unsigned _v4_7_2;
                    unsigned _v4_7_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(cos_sin + (cs_base + 4) + (0))) : "memory");
                    c_lo[4 + 0] = __uint_as_float(_v4_7_0);
                    c_lo[4 + 1] = __uint_as_float(_v4_7_1);
                    c_lo[4 + 2] = __uint_as_float(_v4_7_2);
                    c_lo[4 + 3] = __uint_as_float(_v4_7_3);
                }
                {
                    unsigned _v4_8_0;
                    unsigned _v4_8_1;
                    unsigned _v4_8_2;
                    unsigned _v4_8_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3) : "l"((const void*)(cos_sin + (cs_base + 64) + (0))) : "memory");
                    c_hi[0 + 0] = __uint_as_float(_v4_8_0);
                    c_hi[0 + 1] = __uint_as_float(_v4_8_1);
                    c_hi[0 + 2] = __uint_as_float(_v4_8_2);
                    c_hi[0 + 3] = __uint_as_float(_v4_8_3);
                }
                {
                    unsigned _v4_9_0;
                    unsigned _v4_9_1;
                    unsigned _v4_9_2;
                    unsigned _v4_9_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3) : "l"((const void*)(cos_sin + (cs_base + 64 + 4) + (0))) : "memory");
                    c_hi[4 + 0] = __uint_as_float(_v4_9_0);
                    c_hi[4 + 1] = __uint_as_float(_v4_9_1);
                    c_hi[4 + 2] = __uint_as_float(_v4_9_2);
                    c_hi[4 + 3] = __uint_as_float(_v4_9_3);
                }
            }
            if ((is_k || is_v) && !has_out_kv) {
                int page_slot = pos / page_size;
                int row_in_page = pos - page_slot * page_size;
                int phys = page_indices[batch_i * max_pages_per_request + page_slot];
                cache_row = ((long long)phys * (long long)page_size + (long long)row_in_page) * (long long)(num_kv_heads * 128) + (long long)(kv_head * 128);
            }
            float x_lo[8];
            float x_hi[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x_lo[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_w[_pair]) << 16);
                (&x_lo[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_w[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x_hi[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_w[_pair]) << 16);
                (&x_hi[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_w[_pair]) & 0xffff0000u);
            }
            if (is_q || is_k) {
                if (norm_policy == 2) {
                    float ss = 0.0f;
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        ss = __fmaf_rn(x_lo[j_2], x_lo[j_2], ss);
                        ss = __fmaf_rn(x_hi[j_2], x_hi[j_2], ss);
                    }
                    float _shfl_xor_sync_10 = __shfl_xor_sync(grp_mask, ss, 4);
                    ss = ss + _shfl_xor_sync_10;
                    float _shfl_xor_sync_11 = __shfl_xor_sync(grp_mask, ss, 2);
                    ss = ss + _shfl_xor_sync_11;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(grp_mask, ss, 1);
                    ss = ss + _shfl_xor_sync_12;
                    float inv_rms = rsqrtf(__fmaf_rn(ss, 0.0078125f, 1e-06f));
                    float w_lo[8];
                    float w_hi[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_lo[(0) + 3]))
                        : "r"(w_base));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_lo[(4) + 3]))
                        : "r"(w_base + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_hi[(0) + 3]))
                        : "r"(w_base + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_hi[(4) + 3]))
                        : "r"(w_base + 272));
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        x_lo[j_3] = x_lo[j_3] * (inv_rms * w_lo[j_3]);
                        x_hi[j_3] = x_hi[j_3] * (inv_rms * w_hi[j_3]);
                    }
                }
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    float y_lo = __fmaf_rn(x_lo[j_4], c_lo[j_4], -(x_hi[j_4] * c_hi[j_4]));
                    float y_hi = __fmaf_rn(x_hi[j_4], c_lo[j_4], x_lo[j_4] * c_hi[j_4]);
                    x_lo[j_4] = y_lo;
                    x_hi[j_4] = y_hi;
                }
                if (norm_policy == 1) {
                    float ss1 = 0.0f;
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 8; j_5++) {
                        ss1 = __fmaf_rn(x_lo[j_5], x_lo[j_5], ss1);
                        ss1 = __fmaf_rn(x_hi[j_5], x_hi[j_5], ss1);
                    }
                    float _shfl_xor_sync_13 = __shfl_xor_sync(grp_mask, ss1, 4);
                    ss1 = ss1 + _shfl_xor_sync_13;
                    float _shfl_xor_sync_14 = __shfl_xor_sync(grp_mask, ss1, 2);
                    ss1 = ss1 + _shfl_xor_sync_14;
                    float _shfl_xor_sync_15 = __shfl_xor_sync(grp_mask, ss1, 1);
                    ss1 = ss1 + _shfl_xor_sync_15;
                    float inv_rms1 = rsqrtf(__fmaf_rn(ss1, 0.0078125f, 1e-06f));
                    float w1_lo[8];
                    float w1_hi[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[(0) + 3]))
                        : "r"(w_base));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w1_lo[(4) + 3]))
                        : "r"(w_base + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[(0) + 3]))
                        : "r"(w_base + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w1_hi[(4) + 3]))
                        : "r"(w_base + 272));
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        x_lo[j_6] = x_lo[j_6] * (inv_rms1 * w1_lo[j_6]);
                        x_hi[j_6] = x_hi[j_6] * (inv_rms1 * w1_hi[j_6]);
                    }
                }
            }
            float mult = mult_fixed;
            if (is_q && quant_policy == 1) {
                float m = 1e-06f;
                #pragma unroll
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    m = fmaxf(m, fabsf(x_lo[j_7]));
                    m = fmaxf(m, fabsf(x_hi[j_7]));
                }
                float _shfl_xor_sync_16 = __shfl_xor_sync(grp_mask, m, 4);
                m = fmaxf(m, _shfl_xor_sync_16);
                float _shfl_xor_sync_17 = __shfl_xor_sync(grp_mask, m, 2);
                m = fmaxf(m, _shfl_xor_sync_17);
                float _shfl_xor_sync_18 = __shfl_xor_sync(grp_mask, m, 1);
                m = fmaxf(m, _shfl_xor_sync_18);
                float scale_val = __fdiv_rn(m, upper_max);
                if (lane_1 == 0) {
                    if (is_prefill) {
                        if (tok < max_seqlen_aligned) {
                            long long qs_idx = ((long long)batch_i * (long long)num_q_heads + (long long)head_in_row) * (long long)max_seqlen_aligned + (long long)tok;
                            *(reinterpret_cast<float*>(q_scale + qs_idx) + (0)) = scale_val;
                        }
                    } else {
                        long long qs_idx2 = (long long)row * (long long)num_q_heads + (long long)head_in_row;
                        *(reinterpret_cast<float*>(q_scale + qs_idx2) + (0)) = scale_val;
                    }
                }
                float _rcp_19 = __frcp_rn(scale_val);
                mult = _rcp_19;
            }
            #pragma unroll
            for (int j_8 = 0; j_8 < 8; j_8++) {
                x_lo[j_8] = x_lo[j_8] * mult;
                x_hi[j_8] = x_hi[j_8] * mult;
            }
            long long row64 = row;
            unsigned int _packed_lo[2];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(x_lo[0]), "f"(x_lo[1]),
                                       "f"(x_lo[2]), "f"(x_lo[3]));
                _packed_lo[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(x_lo[4]), "f"(x_lo[5]),
                                       "f"(x_lo[6]), "f"(x_lo[7]));
                _packed_lo[1] = _packed;
            }
            unsigned int _packed_hi[2];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(x_hi[0]), "f"(x_hi[1]),
                                       "f"(x_hi[2]), "f"(x_hi[3]));
                _packed_hi[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(x_hi[4]), "f"(x_hi[5]),
                                       "f"(x_hi[6]), "f"(x_hi[7]));
                _packed_hi[1] = _packed;
            }
            if (is_q) {
                long long q_dst = (row64 * (long long)num_q_heads + (long long)head_in_row) * 128;
                reinterpret_cast<int2*>(out_q + (q_dst + (long long)lane_lo))[0] = reinterpret_cast<int2*>(_packed_lo)[0];
                reinterpret_cast<int2*>(out_q + (q_dst + (long long)lane_hi))[0] = reinterpret_cast<int2*>(_packed_hi)[0];
            } else if (has_out_kv) {
                long long kv_dst = (row64 * (long long)num_kv_heads + (long long)kv_head) * 128;
                if (is_k) {
                    reinterpret_cast<int2*>(out_k + (kv_dst + (long long)lane_lo))[0] = reinterpret_cast<int2*>(_packed_lo)[0];
                    reinterpret_cast<int2*>(out_k + (kv_dst + (long long)lane_hi))[0] = reinterpret_cast<int2*>(_packed_hi)[0];
                } else {
                    reinterpret_cast<int2*>(out_v + (kv_dst + (long long)lane_lo))[0] = reinterpret_cast<int2*>(_packed_lo)[0];
                    reinterpret_cast<int2*>(out_v + (kv_dst + (long long)lane_hi))[0] = reinterpret_cast<int2*>(_packed_hi)[0];
                }
            } else {
                if (is_k) {
                    reinterpret_cast<int2*>(key_cache + (cache_row + (long long)lane_lo))[0] = reinterpret_cast<int2*>(_packed_lo)[0];
                    reinterpret_cast<int2*>(key_cache + (cache_row + (long long)lane_hi))[0] = reinterpret_cast<int2*>(_packed_hi)[0];
                } else {
                    reinterpret_cast<int2*>(value_cache + (cache_row + (long long)lane_lo))[0] = reinterpret_cast<int2*>(_packed_lo)[0];
                    reinterpret_cast<int2*>(value_cache + (cache_row + (long long)lane_hi))[0] = reinterpret_cast<int2*>(_packed_hi)[0];
                }
            }
        }
    }
}

} // extern "C"
