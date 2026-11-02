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
#define SMEM_WARP_TAB_OFF 0
#define SMEM_WARP_TAB_STAGE_BYTES 128
#define SMEM_WARP_TAB_STRIDE 128
#define SMEM_CTA_TAB_OFF 128
#define SMEM_CTA_TAB_STAGE_BYTES 128
#define SMEM_CTA_TAB_STRIDE 128
#define SMEM_FLAG_OFF 256
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 384
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_tp_moe_finalize_rmsnorm_b5b7351771974b74e6b5(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen, float* __restrict__ part, unsigned int* __restrict__ ctr, unsigned long long r_mcast)
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
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* warp_tab = reinterpret_cast<float*>(smem_raw + 0);
    const int warp_tab_addr = smem + 0;
    float* cta_tab = reinterpret_cast<float*>(smem_raw + 128);
    const int cta_tab_addr = smem + 128;
    unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 256);
    const int flag_addr = smem + 256;

    // === Task calls (declaration order) ===
    int band = blockIdx.x;
    int g = blockIdx.y;
    int d = blockIdx.z;
    int cta_rank_0 = cta_rank;
    int rank_i = rank;
    if (d == 0) {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    }
    int col = (band * 128 + tid) * 8;
    long long col64 = (long long)col;
    long long rank64 = (long long)rank;
    long long slot64 = (long long)g;
    long long dest64 = (long long)d;
    int last = n_tok - 1;
    int tok_push = g * 4 + d;
    int tok_p = ((tok_push < n_tok) ? tok_push : last);
    int tok_own = g * 4 + rank_i;
    int tok_o = ((tok_own < n_tok) ? tok_own : last);
    int gidx = g * 4 + d;
    int gn = gen[gidx];
    int gn_old = ((gn == 0) ? 2 : gn - 1);
    long long cur_slot = (long long)gn * (long long)slots + slot64;
    long long old_slot = (long long)gn_old * (long long)slots + slot64;
    long long a_cur = cur_slot * 131072 + col64;
    long long a_old = old_slot * 131072 + col64;
    long long r_cur = cur_slot * 16384 + col64;
    long long r_old = old_slot * 16384 + col64;
    long long a_local_addr = peer_table[rank];
    long long r_local_addr = peer_table[4 + rank];
    __nv_bfloat16* a_local = reinterpret_cast<__nv_bfloat16*>(a_local_addr);
    __nv_bfloat16* r_local = reinterpret_cast<__nv_bfloat16*>(r_local_addr);
    long long a_dest_addr = peer_table[d];
    __nv_bfloat16* a_dest = reinterpret_cast<__nv_bfloat16*>(a_dest_addr);
    uint32_t _smid_read_0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(_smid_read_0));
    int idx8[8];
    {
        const int4* _ivptr_0 = reinterpret_cast<const int4*>(idx + (tok_p * 8) + 0);
        int4 _ivld_0;
        _ivld_0 = *_ivptr_0;
        idx8[0 + 0] = _ivld_0.x;
        idx8[0 + 1] = _ivld_0.y;
        idx8[0 + 2] = _ivld_0.z;
        idx8[0 + 3] = _ivld_0.w;
    }
    {
        const int4* _ivptr_1 = reinterpret_cast<const int4*>(idx + (tok_p * 8 + 4) + 0);
        int4 _ivld_1;
        _ivld_1 = *_ivptr_1;
        idx8[4 + 0] = _ivld_1.x;
        idx8[4 + 1] = _ivld_1.y;
        idx8[4 + 2] = _ivld_1.z;
        idx8[4 + 3] = _ivld_1.w;
    }
    unsigned int _vec_load_0[4];
    {
        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(norm_weight + col + 0);
        _vec_load_0[0 + 0] = _uv4_2.x;
        _vec_load_0[0 + 1] = _uv4_2.y;
        _vec_load_0[0 + 2] = _uv4_2.z;
        _vec_load_0[0 + 3] = _uv4_2.w;
    }
    float _unpack_bf16_0[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_0[_pair * 2])[0]), "=f"((&_unpack_bf16_0[_pair * 2])[1])
            : "r"(_vec_load_0[_pair]));
    }
    float gw8[8];
    gw8[0] = _unpack_bf16_0[0] + gain_center;
    gw8[1] = _unpack_bf16_0[1] + gain_center;
    gw8[2] = _unpack_bf16_0[2] + gain_center;
    gw8[3] = _unpack_bf16_0[3] + gain_center;
    gw8[4] = _unpack_bf16_0[4] + gain_center;
    gw8[5] = _unpack_bf16_0[5] + gain_center;
    gw8[6] = _unpack_bf16_0[6] + gain_center;
    gw8[7] = _unpack_bf16_0[7] + gain_center;
    unsigned int sent4[4];
    sent4[0] = 2147516416;
    sent4[1] = 2147516416;
    sent4[2] = 2147516416;
    sent4[3] = 2147516416;
    {
        auto _gst_3 = r_local + (r_old + dest64 * 4096);
        __builtin_assume(__isGlobal(_gst_3));
        reinterpret_cast<int4*>(_gst_3)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    unsigned int packets[32];
    packets[0] = 0;
    packets[1] = 0;
    packets[2] = 0;
    packets[3] = 0;
    packets[4] = 0;
    packets[5] = 0;
    packets[6] = 0;
    packets[7] = 0;
    packets[8] = 0;
    packets[9] = 0;
    packets[10] = 0;
    packets[11] = 0;
    packets[12] = 0;
    packets[13] = 0;
    packets[14] = 0;
    packets[15] = 0;
    packets[16] = 0;
    packets[17] = 0;
    packets[18] = 0;
    packets[19] = 0;
    packets[20] = 0;
    packets[21] = 0;
    packets[22] = 0;
    packets[23] = 0;
    packets[24] = 0;
    packets[25] = 0;
    packets[26] = 0;
    packets[27] = 0;
    packets[28] = 0;
    packets[29] = 0;
    packets[30] = 0;
    packets[31] = 0;
    int ridx_0 = idx8[0];
    if (ridx_0 >= 0 && ridx_0 < rows) {
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_0 * (long long)row_stride + col64) + 0);
            packets[0 + 0] = _uv4_4.x;
            packets[0 + 1] = _uv4_4.y;
            packets[0 + 2] = _uv4_4.z;
            packets[0 + 3] = _uv4_4.w;
        }
    }
    int ridx_1 = idx8[1];
    if (ridx_1 >= 0 && ridx_1 < rows) {
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_1 * (long long)row_stride + col64) + 0);
            packets[4 + 0] = _uv4_5.x;
            packets[4 + 1] = _uv4_5.y;
            packets[4 + 2] = _uv4_5.z;
            packets[4 + 3] = _uv4_5.w;
        }
    }
    int ridx_2 = idx8[2];
    if (ridx_2 >= 0 && ridx_2 < rows) {
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_2 * (long long)row_stride + col64) + 0);
            packets[8 + 0] = _uv4_6.x;
            packets[8 + 1] = _uv4_6.y;
            packets[8 + 2] = _uv4_6.z;
            packets[8 + 3] = _uv4_6.w;
        }
    }
    int ridx_3 = idx8[3];
    if (ridx_3 >= 0 && ridx_3 < rows) {
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_3 * (long long)row_stride + col64) + 0);
            packets[12 + 0] = _uv4_7.x;
            packets[12 + 1] = _uv4_7.y;
            packets[12 + 2] = _uv4_7.z;
            packets[12 + 3] = _uv4_7.w;
        }
    }
    int ridx_4 = idx8[4];
    if (ridx_4 >= 0 && ridx_4 < rows) {
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_4 * (long long)row_stride + col64) + 0);
            packets[16 + 0] = _uv4_8.x;
            packets[16 + 1] = _uv4_8.y;
            packets[16 + 2] = _uv4_8.z;
            packets[16 + 3] = _uv4_8.w;
        }
    }
    int ridx_5 = idx8[5];
    if (ridx_5 >= 0 && ridx_5 < rows) {
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_5 * (long long)row_stride + col64) + 0);
            packets[20 + 0] = _uv4_9.x;
            packets[20 + 1] = _uv4_9.y;
            packets[20 + 2] = _uv4_9.z;
            packets[20 + 3] = _uv4_9.w;
        }
    }
    int ridx_6 = idx8[6];
    if (ridx_6 >= 0 && ridx_6 < rows) {
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_6 * (long long)row_stride + col64) + 0);
            packets[24 + 0] = _uv4_10.x;
            packets[24 + 1] = _uv4_10.y;
            packets[24 + 2] = _uv4_10.z;
            packets[24 + 3] = _uv4_10.w;
        }
    }
    int ridx_7 = idx8[7];
    if (ridx_7 >= 0 && ridx_7 < rows) {
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_7 * (long long)row_stride + col64) + 0);
            packets[28 + 0] = _uv4_11.x;
            packets[28 + 1] = _uv4_11.y;
            packets[28 + 2] = _uv4_11.z;
            packets[28 + 3] = _uv4_11.w;
        }
    }
    if (tok_push < n_tok) {
        unsigned int raw0_1_0[4];
        raw0_1_0[0] = packets[0];
        raw0_1_0[1] = packets[1];
        raw0_1_0[2] = packets[2];
        raw0_1_0[3] = packets[3];
        uint32_t _packed16_sanitize_0[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_12 = static_cast<uint32_t>(raw0_1_0[_p16s_word]);
            uint32_t _p16s_lo_12 = _p16s_carrier_12 & 0x0000ffffu;
            uint32_t _p16s_hi_12 = (_p16s_carrier_12 >> 16) & 0x0000ffffu;
            if (_p16s_lo_12 == 0x00008000u) _p16s_lo_12 = 0u;
            if (_p16s_hi_12 == 0x00008000u) _p16s_hi_12 = 0u;
            _packed16_sanitize_0[_p16s_word] = _p16s_lo_12 | (_p16s_hi_12 << 16);
        }
        {
            auto _gst_13 = a_dest + (a_cur + rank64 * 8 * 4096);
            __builtin_assume(__isGlobal(_gst_13));
            reinterpret_cast<int4*>(_gst_13)[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
        }
        unsigned int raw0_1_1[4];
        raw0_1_1[0] = packets[4];
        raw0_1_1[1] = packets[5];
        raw0_1_1[2] = packets[6];
        raw0_1_1[3] = packets[7];
        uint32_t _packed16_sanitize_1[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_14 = static_cast<uint32_t>(raw0_1_1[_p16s_word]);
            uint32_t _p16s_lo_14 = _p16s_carrier_14 & 0x0000ffffu;
            uint32_t _p16s_hi_14 = (_p16s_carrier_14 >> 16) & 0x0000ffffu;
            if (_p16s_lo_14 == 0x00008000u) _p16s_lo_14 = 0u;
            if (_p16s_hi_14 == 0x00008000u) _p16s_hi_14 = 0u;
            _packed16_sanitize_1[_p16s_word] = _p16s_lo_14 | (_p16s_hi_14 << 16);
        }
        {
            auto _gst_15 = a_dest + (a_cur + (rank64 * 8 + 1) * 4096);
            __builtin_assume(__isGlobal(_gst_15));
            reinterpret_cast<int4*>(_gst_15)[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
        }
        unsigned int raw0_1_2[4];
        raw0_1_2[0] = packets[8];
        raw0_1_2[1] = packets[9];
        raw0_1_2[2] = packets[10];
        raw0_1_2[3] = packets[11];
        uint32_t _packed16_sanitize_2[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_16 = static_cast<uint32_t>(raw0_1_2[_p16s_word]);
            uint32_t _p16s_lo_16 = _p16s_carrier_16 & 0x0000ffffu;
            uint32_t _p16s_hi_16 = (_p16s_carrier_16 >> 16) & 0x0000ffffu;
            if (_p16s_lo_16 == 0x00008000u) _p16s_lo_16 = 0u;
            if (_p16s_hi_16 == 0x00008000u) _p16s_hi_16 = 0u;
            _packed16_sanitize_2[_p16s_word] = _p16s_lo_16 | (_p16s_hi_16 << 16);
        }
        {
            auto _gst_17 = a_dest + (a_cur + (rank64 * 8 + 2) * 4096);
            __builtin_assume(__isGlobal(_gst_17));
            reinterpret_cast<int4*>(_gst_17)[0] = reinterpret_cast<int4*>(_packed16_sanitize_2)[0];
        }
        unsigned int raw0_1_3[4];
        raw0_1_3[0] = packets[12];
        raw0_1_3[1] = packets[13];
        raw0_1_3[2] = packets[14];
        raw0_1_3[3] = packets[15];
        uint32_t _packed16_sanitize_3[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_18 = static_cast<uint32_t>(raw0_1_3[_p16s_word]);
            uint32_t _p16s_lo_18 = _p16s_carrier_18 & 0x0000ffffu;
            uint32_t _p16s_hi_18 = (_p16s_carrier_18 >> 16) & 0x0000ffffu;
            if (_p16s_lo_18 == 0x00008000u) _p16s_lo_18 = 0u;
            if (_p16s_hi_18 == 0x00008000u) _p16s_hi_18 = 0u;
            _packed16_sanitize_3[_p16s_word] = _p16s_lo_18 | (_p16s_hi_18 << 16);
        }
        {
            auto _gst_19 = a_dest + (a_cur + (rank64 * 8 + 3) * 4096);
            __builtin_assume(__isGlobal(_gst_19));
            reinterpret_cast<int4*>(_gst_19)[0] = reinterpret_cast<int4*>(_packed16_sanitize_3)[0];
        }
        unsigned int raw0_1_4[4];
        raw0_1_4[0] = packets[16];
        raw0_1_4[1] = packets[17];
        raw0_1_4[2] = packets[18];
        raw0_1_4[3] = packets[19];
        uint32_t _packed16_sanitize_4[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_20 = static_cast<uint32_t>(raw0_1_4[_p16s_word]);
            uint32_t _p16s_lo_20 = _p16s_carrier_20 & 0x0000ffffu;
            uint32_t _p16s_hi_20 = (_p16s_carrier_20 >> 16) & 0x0000ffffu;
            if (_p16s_lo_20 == 0x00008000u) _p16s_lo_20 = 0u;
            if (_p16s_hi_20 == 0x00008000u) _p16s_hi_20 = 0u;
            _packed16_sanitize_4[_p16s_word] = _p16s_lo_20 | (_p16s_hi_20 << 16);
        }
        {
            auto _gst_21 = a_dest + (a_cur + (rank64 * 8 + 4) * 4096);
            __builtin_assume(__isGlobal(_gst_21));
            reinterpret_cast<int4*>(_gst_21)[0] = reinterpret_cast<int4*>(_packed16_sanitize_4)[0];
        }
        unsigned int raw0_1_5[4];
        raw0_1_5[0] = packets[20];
        raw0_1_5[1] = packets[21];
        raw0_1_5[2] = packets[22];
        raw0_1_5[3] = packets[23];
        uint32_t _packed16_sanitize_5[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_22 = static_cast<uint32_t>(raw0_1_5[_p16s_word]);
            uint32_t _p16s_lo_22 = _p16s_carrier_22 & 0x0000ffffu;
            uint32_t _p16s_hi_22 = (_p16s_carrier_22 >> 16) & 0x0000ffffu;
            if (_p16s_lo_22 == 0x00008000u) _p16s_lo_22 = 0u;
            if (_p16s_hi_22 == 0x00008000u) _p16s_hi_22 = 0u;
            _packed16_sanitize_5[_p16s_word] = _p16s_lo_22 | (_p16s_hi_22 << 16);
        }
        {
            auto _gst_23 = a_dest + (a_cur + (rank64 * 8 + 5) * 4096);
            __builtin_assume(__isGlobal(_gst_23));
            reinterpret_cast<int4*>(_gst_23)[0] = reinterpret_cast<int4*>(_packed16_sanitize_5)[0];
        }
        unsigned int raw0_1_6[4];
        raw0_1_6[0] = packets[24];
        raw0_1_6[1] = packets[25];
        raw0_1_6[2] = packets[26];
        raw0_1_6[3] = packets[27];
        uint32_t _packed16_sanitize_6[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_24 = static_cast<uint32_t>(raw0_1_6[_p16s_word]);
            uint32_t _p16s_lo_24 = _p16s_carrier_24 & 0x0000ffffu;
            uint32_t _p16s_hi_24 = (_p16s_carrier_24 >> 16) & 0x0000ffffu;
            if (_p16s_lo_24 == 0x00008000u) _p16s_lo_24 = 0u;
            if (_p16s_hi_24 == 0x00008000u) _p16s_hi_24 = 0u;
            _packed16_sanitize_6[_p16s_word] = _p16s_lo_24 | (_p16s_hi_24 << 16);
        }
        {
            auto _gst_25 = a_dest + (a_cur + (rank64 * 8 + 6) * 4096);
            __builtin_assume(__isGlobal(_gst_25));
            reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(_packed16_sanitize_6)[0];
        }
        unsigned int raw0_1_7[4];
        raw0_1_7[0] = packets[28];
        raw0_1_7[1] = packets[29];
        raw0_1_7[2] = packets[30];
        raw0_1_7[3] = packets[31];
        uint32_t _packed16_sanitize_7[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_26 = static_cast<uint32_t>(raw0_1_7[_p16s_word]);
            uint32_t _p16s_lo_26 = _p16s_carrier_26 & 0x0000ffffu;
            uint32_t _p16s_hi_26 = (_p16s_carrier_26 >> 16) & 0x0000ffffu;
            if (_p16s_lo_26 == 0x00008000u) _p16s_lo_26 = 0u;
            if (_p16s_hi_26 == 0x00008000u) _p16s_hi_26 = 0u;
            _packed16_sanitize_7[_p16s_word] = _p16s_lo_26 | (_p16s_hi_26 << 16);
        }
        {
            auto _gst_27 = a_dest + (a_cur + (rank64 * 8 + 7) * 4096);
            __builtin_assume(__isGlobal(_gst_27));
            reinterpret_cast<int4*>(_gst_27)[0] = reinterpret_cast<int4*>(_packed16_sanitize_7)[0];
        }
    }
    long long q_own = (long long)d * 8;
    {
        auto _gst_28 = a_local + (a_old + q_own * 4096);
        __builtin_assume(__isGlobal(_gst_28));
        reinterpret_cast<int4*>(_gst_28)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_29 = a_local + (a_old + (q_own + 1) * 4096);
        __builtin_assume(__isGlobal(_gst_29));
        reinterpret_cast<int4*>(_gst_29)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_30 = a_local + (a_old + (q_own + 2) * 4096);
        __builtin_assume(__isGlobal(_gst_30));
        reinterpret_cast<int4*>(_gst_30)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_31 = a_local + (a_old + (q_own + 3) * 4096);
        __builtin_assume(__isGlobal(_gst_31));
        reinterpret_cast<int4*>(_gst_31)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_32 = a_local + (a_old + (q_own + 4) * 4096);
        __builtin_assume(__isGlobal(_gst_32));
        reinterpret_cast<int4*>(_gst_32)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_33 = a_local + (a_old + (q_own + 5) * 4096);
        __builtin_assume(__isGlobal(_gst_33));
        reinterpret_cast<int4*>(_gst_33)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_34 = a_local + (a_old + (q_own + 6) * 4096);
        __builtin_assume(__isGlobal(_gst_34));
        reinterpret_cast<int4*>(_gst_34)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_35 = a_local + (a_old + (q_own + 7) * 4096);
        __builtin_assume(__isGlobal(_gst_35));
        reinterpret_cast<int4*>(_gst_35)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    if (d == 0 && tok_own < n_tok) {
        long long xf = 0;
        float w8o[8];
        if (weights_fp32 != 0) {
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(expert_weights + (tok_o * 8)) + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(expert_weights + (tok_o * 8 + 4)) + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            w8o[0] = _vec_load_1[0];
            w8o[4] = _vec_load_2[0];
            w8o[1] = _vec_load_1[1];
            w8o[5] = _vec_load_2[1];
            w8o[2] = _vec_load_1[2];
            w8o[6] = _vec_load_2[2];
            w8o[3] = _vec_load_1[3];
            w8o[7] = _vec_load_2[3];
        } else {
            unsigned int _vec_load_3[4];
            {
                uint4 _uv4_38 = *reinterpret_cast<const uint4*>(expert_weights + (tok_o * 4) + 0);
                _vec_load_3[0 + 0] = _uv4_38.x;
                _vec_load_3[0 + 1] = _uv4_38.y;
                _vec_load_3[0 + 2] = _uv4_38.z;
                _vec_load_3[0 + 3] = _uv4_38.w;
            }
            float _unpack_bf16_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
                    : "r"(_vec_load_3[_pair]));
            }
            w8o[0] = _unpack_bf16_1[0];
            w8o[1] = _unpack_bf16_1[1];
            w8o[2] = _unpack_bf16_1[2];
            w8o[3] = _unpack_bf16_1[3];
            w8o[4] = _unpack_bf16_1[4];
            w8o[5] = _unpack_bf16_1[5];
            w8o[6] = _unpack_bf16_1[6];
            w8o[7] = _unpack_bf16_1[7];
        }
        float wk[8];
        wk[0] = w8o[0];
        wk[1] = w8o[1];
        wk[2] = w8o[2];
        wk[3] = w8o[3];
        wk[4] = w8o[4];
        wk[5] = w8o[5];
        wk[6] = w8o[6];
        wk[7] = w8o[7];
        unsigned int rw[32];
        float ssq_k[8];
        uint32_t _sysv_poll_group_0[64];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + xf * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + xf) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + xf) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + xf) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 1) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 1)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 1)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 1)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[32]), "=r"(_sysv_poll_group_0[33]), "=r"(_sysv_poll_group_0[34]), "=r"(_sysv_poll_group_0[35]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 2) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[36]), "=r"(_sysv_poll_group_0[37]), "=r"(_sysv_poll_group_0[38]), "=r"(_sysv_poll_group_0[39]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 2)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[40]), "=r"(_sysv_poll_group_0[41]), "=r"(_sysv_poll_group_0[42]), "=r"(_sysv_poll_group_0[43]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 2)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[44]), "=r"(_sysv_poll_group_0[45]), "=r"(_sysv_poll_group_0[46]), "=r"(_sysv_poll_group_0[47]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 2)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[48]), "=r"(_sysv_poll_group_0[49]), "=r"(_sysv_poll_group_0[50]), "=r"(_sysv_poll_group_0[51]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 3) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[52]), "=r"(_sysv_poll_group_0[53]), "=r"(_sysv_poll_group_0[54]), "=r"(_sysv_poll_group_0[55]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 3)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[56]), "=r"(_sysv_poll_group_0[57]), "=r"(_sysv_poll_group_0[58]), "=r"(_sysv_poll_group_0[59]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 3)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[60]), "=r"(_sysv_poll_group_0[61]), "=r"(_sysv_poll_group_0[62]), "=r"(_sysv_poll_group_0[63]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 3)) * 4096))) : "memory");
        } while (
            (((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[33] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[35] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[37] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[39] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[41] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[43] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[45] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[47] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[49] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[51] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[53] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[55] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[57] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[59] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[61] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[63] >> 16) & 0xffffu) == 0x8000u)
        );
        float u_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_0[_pair * 2])[0]), "=f"((&u_0[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[_pair]));
        }
        float u_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_1[_pair * 2])[0]), "=f"((&u_1[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[4 + _pair]));
        }
        float u_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_2[_pair * 2])[0]), "=f"((&u_2[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[8 + _pair]));
        }
        float u_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_3[_pair * 2])[0]), "=f"((&u_3[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[12 + _pair]));
        }
        float s8[8];
        float t_0 = (float)u_0[0];
        s8[0] = t_0;
        float t_1 = (float)u_1[0];
        s8[0] = s8[0] + t_1;
        float t_2 = (float)u_2[0];
        s8[0] = s8[0] + t_2;
        float t_3 = (float)u_3[0];
        s8[0] = s8[0] + t_3;
        float t_4 = (float)u_0[1];
        s8[1] = t_4;
        float t_5 = (float)u_1[1];
        s8[1] = s8[1] + t_5;
        float t_6 = (float)u_2[1];
        s8[1] = s8[1] + t_6;
        float t_7 = (float)u_3[1];
        s8[1] = s8[1] + t_7;
        float t_8 = (float)u_0[2];
        s8[2] = t_8;
        float t_9 = (float)u_1[2];
        s8[2] = s8[2] + t_9;
        float t_10 = (float)u_2[2];
        s8[2] = s8[2] + t_10;
        float t_11 = (float)u_3[2];
        s8[2] = s8[2] + t_11;
        float t_12 = (float)u_0[3];
        s8[3] = t_12;
        float t_13 = (float)u_1[3];
        s8[3] = s8[3] + t_13;
        float t_14 = (float)u_2[3];
        s8[3] = s8[3] + t_14;
        float t_15 = (float)u_3[3];
        s8[3] = s8[3] + t_15;
        float t_16 = (float)u_0[4];
        s8[4] = t_16;
        float t_17 = (float)u_1[4];
        s8[4] = s8[4] + t_17;
        float t_18 = (float)u_2[4];
        s8[4] = s8[4] + t_18;
        float t_19 = (float)u_3[4];
        s8[4] = s8[4] + t_19;
        float t_20 = (float)u_0[5];
        s8[5] = t_20;
        float t_21 = (float)u_1[5];
        s8[5] = s8[5] + t_21;
        float t_22 = (float)u_2[5];
        s8[5] = s8[5] + t_22;
        float t_23 = (float)u_3[5];
        s8[5] = s8[5] + t_23;
        float t_24 = (float)u_0[6];
        s8[6] = t_24;
        float t_25 = (float)u_1[6];
        s8[6] = s8[6] + t_25;
        float t_26 = (float)u_2[6];
        s8[6] = s8[6] + t_26;
        float t_27 = (float)u_3[6];
        s8[6] = s8[6] + t_27;
        float t_28 = (float)u_0[7];
        s8[7] = t_28;
        float t_29 = (float)u_1[7];
        s8[7] = s8[7] + t_29;
        float t_30 = (float)u_2[7];
        s8[7] = s8[7] + t_30;
        float t_31 = (float)u_3[7];
        s8[7] = s8[7] + t_31;
        unsigned int s8_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8[_lp*2 + 0], s8[_lp*2+1 + 0]));
            s8_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        rw[0] = s8_bf16[0];
        rw[1] = s8_bf16[1];
        rw[2] = s8_bf16[2];
        rw[3] = s8_bf16[3];
        float y8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8[_pair * 2])[0]), "=f"((&y8[_pair * 2])[1])
                : "r"(s8_bf16[_pair]));
        }
        float ssq = 0.0f;
        float y_0 = ((mode != 0) ? y8[0] * gw8[0] : y8[0]);
        ssq = __fmaf_rn(y_0, y_0, ssq);
        float y_1 = ((mode != 0) ? y8[1] * gw8[1] : y8[1]);
        ssq = __fmaf_rn(y_1, y_1, ssq);
        float y_2 = ((mode != 0) ? y8[2] * gw8[2] : y8[2]);
        ssq = __fmaf_rn(y_2, y_2, ssq);
        float y_3 = ((mode != 0) ? y8[3] * gw8[3] : y8[3]);
        ssq = __fmaf_rn(y_3, y_3, ssq);
        float y_4 = ((mode != 0) ? y8[4] * gw8[4] : y8[4]);
        ssq = __fmaf_rn(y_4, y_4, ssq);
        float y_5 = ((mode != 0) ? y8[5] * gw8[5] : y8[5]);
        ssq = __fmaf_rn(y_5, y_5, ssq);
        float y_6 = ((mode != 0) ? y8[6] * gw8[6] : y8[6]);
        ssq = __fmaf_rn(y_6, y_6, ssq);
        float y_7 = ((mode != 0) ? y8[7] * gw8[7] : y8[7]);
        ssq = __fmaf_rn(y_7, y_7, ssq);
        ssq_k[0] = ssq;
        float u_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_4[_pair * 2])[0]), "=f"((&u_4[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[16 + _pair]));
        }
        float u_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_5[_pair * 2])[0]), "=f"((&u_5[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[20 + _pair]));
        }
        float u_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_6[_pair * 2])[0]), "=f"((&u_6[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[24 + _pair]));
        }
        float u_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_7[_pair * 2])[0]), "=f"((&u_7[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[28 + _pair]));
        }
        float s8_0[8];
        float t_32 = (float)u_4[0];
        s8_0[0] = t_32;
        float t_33 = (float)u_5[0];
        s8_0[0] = s8_0[0] + t_33;
        float t_34 = (float)u_6[0];
        s8_0[0] = s8_0[0] + t_34;
        float t_35 = (float)u_7[0];
        s8_0[0] = s8_0[0] + t_35;
        float t_36 = (float)u_4[1];
        s8_0[1] = t_36;
        float t_37 = (float)u_5[1];
        s8_0[1] = s8_0[1] + t_37;
        float t_38 = (float)u_6[1];
        s8_0[1] = s8_0[1] + t_38;
        float t_39 = (float)u_7[1];
        s8_0[1] = s8_0[1] + t_39;
        float t_40 = (float)u_4[2];
        s8_0[2] = t_40;
        float t_41 = (float)u_5[2];
        s8_0[2] = s8_0[2] + t_41;
        float t_42 = (float)u_6[2];
        s8_0[2] = s8_0[2] + t_42;
        float t_43 = (float)u_7[2];
        s8_0[2] = s8_0[2] + t_43;
        float t_44 = (float)u_4[3];
        s8_0[3] = t_44;
        float t_45 = (float)u_5[3];
        s8_0[3] = s8_0[3] + t_45;
        float t_46 = (float)u_6[3];
        s8_0[3] = s8_0[3] + t_46;
        float t_47 = (float)u_7[3];
        s8_0[3] = s8_0[3] + t_47;
        float t_48 = (float)u_4[4];
        s8_0[4] = t_48;
        float t_49 = (float)u_5[4];
        s8_0[4] = s8_0[4] + t_49;
        float t_50 = (float)u_6[4];
        s8_0[4] = s8_0[4] + t_50;
        float t_51 = (float)u_7[4];
        s8_0[4] = s8_0[4] + t_51;
        float t_52 = (float)u_4[5];
        s8_0[5] = t_52;
        float t_53 = (float)u_5[5];
        s8_0[5] = s8_0[5] + t_53;
        float t_54 = (float)u_6[5];
        s8_0[5] = s8_0[5] + t_54;
        float t_55 = (float)u_7[5];
        s8_0[5] = s8_0[5] + t_55;
        float t_56 = (float)u_4[6];
        s8_0[6] = t_56;
        float t_57 = (float)u_5[6];
        s8_0[6] = s8_0[6] + t_57;
        float t_58 = (float)u_6[6];
        s8_0[6] = s8_0[6] + t_58;
        float t_59 = (float)u_7[6];
        s8_0[6] = s8_0[6] + t_59;
        float t_60 = (float)u_4[7];
        s8_0[7] = t_60;
        float t_61 = (float)u_5[7];
        s8_0[7] = s8_0[7] + t_61;
        float t_62 = (float)u_6[7];
        s8_0[7] = s8_0[7] + t_62;
        float t_63 = (float)u_7[7];
        s8_0[7] = s8_0[7] + t_63;
        unsigned int s8_bf16_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_0[_lp*2 + 0], s8_0[_lp*2+1 + 0]));
            s8_bf16_0[_lp] = *(uint32_t*)&_bf2;
        }
        rw[4] = s8_bf16_0[0];
        rw[5] = s8_bf16_0[1];
        rw[6] = s8_bf16_0[2];
        rw[7] = s8_bf16_0[3];
        float y8_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_0[_pair * 2])[0]), "=f"((&y8_0[_pair * 2])[1])
                : "r"(s8_bf16_0[_pair]));
        }
        float ssq_0 = 0.0f;
        float y_8 = ((mode != 0) ? y8_0[0] * gw8[0] : y8_0[0]);
        ssq_0 = __fmaf_rn(y_8, y_8, ssq_0);
        float y_9 = ((mode != 0) ? y8_0[1] * gw8[1] : y8_0[1]);
        ssq_0 = __fmaf_rn(y_9, y_9, ssq_0);
        float y_10 = ((mode != 0) ? y8_0[2] * gw8[2] : y8_0[2]);
        ssq_0 = __fmaf_rn(y_10, y_10, ssq_0);
        float y_11 = ((mode != 0) ? y8_0[3] * gw8[3] : y8_0[3]);
        ssq_0 = __fmaf_rn(y_11, y_11, ssq_0);
        float y_12 = ((mode != 0) ? y8_0[4] * gw8[4] : y8_0[4]);
        ssq_0 = __fmaf_rn(y_12, y_12, ssq_0);
        float y_13 = ((mode != 0) ? y8_0[5] * gw8[5] : y8_0[5]);
        ssq_0 = __fmaf_rn(y_13, y_13, ssq_0);
        float y_14 = ((mode != 0) ? y8_0[6] * gw8[6] : y8_0[6]);
        ssq_0 = __fmaf_rn(y_14, y_14, ssq_0);
        float y_15 = ((mode != 0) ? y8_0[7] * gw8[7] : y8_0[7]);
        ssq_0 = __fmaf_rn(y_15, y_15, ssq_0);
        ssq_k[1] = ssq_0;
        float u_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_8[_pair * 2])[0]), "=f"((&u_8[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[32 + _pair]));
        }
        float u_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_9[_pair * 2])[0]), "=f"((&u_9[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[36 + _pair]));
        }
        float u_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_10[_pair * 2])[0]), "=f"((&u_10[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[40 + _pair]));
        }
        float u_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_11[_pair * 2])[0]), "=f"((&u_11[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[44 + _pair]));
        }
        float s8_1[8];
        float t_64 = (float)u_8[0];
        s8_1[0] = t_64;
        float t_65 = (float)u_9[0];
        s8_1[0] = s8_1[0] + t_65;
        float t_66 = (float)u_10[0];
        s8_1[0] = s8_1[0] + t_66;
        float t_67 = (float)u_11[0];
        s8_1[0] = s8_1[0] + t_67;
        float t_68 = (float)u_8[1];
        s8_1[1] = t_68;
        float t_69 = (float)u_9[1];
        s8_1[1] = s8_1[1] + t_69;
        float t_70 = (float)u_10[1];
        s8_1[1] = s8_1[1] + t_70;
        float t_71 = (float)u_11[1];
        s8_1[1] = s8_1[1] + t_71;
        float t_72 = (float)u_8[2];
        s8_1[2] = t_72;
        float t_73 = (float)u_9[2];
        s8_1[2] = s8_1[2] + t_73;
        float t_74 = (float)u_10[2];
        s8_1[2] = s8_1[2] + t_74;
        float t_75 = (float)u_11[2];
        s8_1[2] = s8_1[2] + t_75;
        float t_76 = (float)u_8[3];
        s8_1[3] = t_76;
        float t_77 = (float)u_9[3];
        s8_1[3] = s8_1[3] + t_77;
        float t_78 = (float)u_10[3];
        s8_1[3] = s8_1[3] + t_78;
        float t_79 = (float)u_11[3];
        s8_1[3] = s8_1[3] + t_79;
        float t_80 = (float)u_8[4];
        s8_1[4] = t_80;
        float t_81 = (float)u_9[4];
        s8_1[4] = s8_1[4] + t_81;
        float t_82 = (float)u_10[4];
        s8_1[4] = s8_1[4] + t_82;
        float t_83 = (float)u_11[4];
        s8_1[4] = s8_1[4] + t_83;
        float t_84 = (float)u_8[5];
        s8_1[5] = t_84;
        float t_85 = (float)u_9[5];
        s8_1[5] = s8_1[5] + t_85;
        float t_86 = (float)u_10[5];
        s8_1[5] = s8_1[5] + t_86;
        float t_87 = (float)u_11[5];
        s8_1[5] = s8_1[5] + t_87;
        float t_88 = (float)u_8[6];
        s8_1[6] = t_88;
        float t_89 = (float)u_9[6];
        s8_1[6] = s8_1[6] + t_89;
        float t_90 = (float)u_10[6];
        s8_1[6] = s8_1[6] + t_90;
        float t_91 = (float)u_11[6];
        s8_1[6] = s8_1[6] + t_91;
        float t_92 = (float)u_8[7];
        s8_1[7] = t_92;
        float t_93 = (float)u_9[7];
        s8_1[7] = s8_1[7] + t_93;
        float t_94 = (float)u_10[7];
        s8_1[7] = s8_1[7] + t_94;
        float t_95 = (float)u_11[7];
        s8_1[7] = s8_1[7] + t_95;
        unsigned int s8_bf16_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_1[_lp*2 + 0], s8_1[_lp*2+1 + 0]));
            s8_bf16_1[_lp] = *(uint32_t*)&_bf2;
        }
        rw[8] = s8_bf16_1[0];
        rw[9] = s8_bf16_1[1];
        rw[10] = s8_bf16_1[2];
        rw[11] = s8_bf16_1[3];
        float y8_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_1[_pair * 2])[0]), "=f"((&y8_1[_pair * 2])[1])
                : "r"(s8_bf16_1[_pair]));
        }
        float ssq_1 = 0.0f;
        float y_16 = ((mode != 0) ? y8_1[0] * gw8[0] : y8_1[0]);
        ssq_1 = __fmaf_rn(y_16, y_16, ssq_1);
        float y_17 = ((mode != 0) ? y8_1[1] * gw8[1] : y8_1[1]);
        ssq_1 = __fmaf_rn(y_17, y_17, ssq_1);
        float y_18 = ((mode != 0) ? y8_1[2] * gw8[2] : y8_1[2]);
        ssq_1 = __fmaf_rn(y_18, y_18, ssq_1);
        float y_19 = ((mode != 0) ? y8_1[3] * gw8[3] : y8_1[3]);
        ssq_1 = __fmaf_rn(y_19, y_19, ssq_1);
        float y_20 = ((mode != 0) ? y8_1[4] * gw8[4] : y8_1[4]);
        ssq_1 = __fmaf_rn(y_20, y_20, ssq_1);
        float y_21 = ((mode != 0) ? y8_1[5] * gw8[5] : y8_1[5]);
        ssq_1 = __fmaf_rn(y_21, y_21, ssq_1);
        float y_22 = ((mode != 0) ? y8_1[6] * gw8[6] : y8_1[6]);
        ssq_1 = __fmaf_rn(y_22, y_22, ssq_1);
        float y_23 = ((mode != 0) ? y8_1[7] * gw8[7] : y8_1[7]);
        ssq_1 = __fmaf_rn(y_23, y_23, ssq_1);
        ssq_k[2] = ssq_1;
        float u_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_12[_pair * 2])[0]), "=f"((&u_12[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[48 + _pair]));
        }
        float u_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_13[_pair * 2])[0]), "=f"((&u_13[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[52 + _pair]));
        }
        float u_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_14[_pair * 2])[0]), "=f"((&u_14[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[56 + _pair]));
        }
        float u_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_15[_pair * 2])[0]), "=f"((&u_15[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[60 + _pair]));
        }
        float s8_2[8];
        float t_96 = (float)u_12[0];
        s8_2[0] = t_96;
        float t_97 = (float)u_13[0];
        s8_2[0] = s8_2[0] + t_97;
        float t_98 = (float)u_14[0];
        s8_2[0] = s8_2[0] + t_98;
        float t_99 = (float)u_15[0];
        s8_2[0] = s8_2[0] + t_99;
        float t_100 = (float)u_12[1];
        s8_2[1] = t_100;
        float t_101 = (float)u_13[1];
        s8_2[1] = s8_2[1] + t_101;
        float t_102 = (float)u_14[1];
        s8_2[1] = s8_2[1] + t_102;
        float t_103 = (float)u_15[1];
        s8_2[1] = s8_2[1] + t_103;
        float t_104 = (float)u_12[2];
        s8_2[2] = t_104;
        float t_105 = (float)u_13[2];
        s8_2[2] = s8_2[2] + t_105;
        float t_106 = (float)u_14[2];
        s8_2[2] = s8_2[2] + t_106;
        float t_107 = (float)u_15[2];
        s8_2[2] = s8_2[2] + t_107;
        float t_108 = (float)u_12[3];
        s8_2[3] = t_108;
        float t_109 = (float)u_13[3];
        s8_2[3] = s8_2[3] + t_109;
        float t_110 = (float)u_14[3];
        s8_2[3] = s8_2[3] + t_110;
        float t_111 = (float)u_15[3];
        s8_2[3] = s8_2[3] + t_111;
        float t_112 = (float)u_12[4];
        s8_2[4] = t_112;
        float t_113 = (float)u_13[4];
        s8_2[4] = s8_2[4] + t_113;
        float t_114 = (float)u_14[4];
        s8_2[4] = s8_2[4] + t_114;
        float t_115 = (float)u_15[4];
        s8_2[4] = s8_2[4] + t_115;
        float t_116 = (float)u_12[5];
        s8_2[5] = t_116;
        float t_117 = (float)u_13[5];
        s8_2[5] = s8_2[5] + t_117;
        float t_118 = (float)u_14[5];
        s8_2[5] = s8_2[5] + t_118;
        float t_119 = (float)u_15[5];
        s8_2[5] = s8_2[5] + t_119;
        float t_120 = (float)u_12[6];
        s8_2[6] = t_120;
        float t_121 = (float)u_13[6];
        s8_2[6] = s8_2[6] + t_121;
        float t_122 = (float)u_14[6];
        s8_2[6] = s8_2[6] + t_122;
        float t_123 = (float)u_15[6];
        s8_2[6] = s8_2[6] + t_123;
        float t_124 = (float)u_12[7];
        s8_2[7] = t_124;
        float t_125 = (float)u_13[7];
        s8_2[7] = s8_2[7] + t_125;
        float t_126 = (float)u_14[7];
        s8_2[7] = s8_2[7] + t_126;
        float t_127 = (float)u_15[7];
        s8_2[7] = s8_2[7] + t_127;
        unsigned int s8_bf16_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_2[_lp*2 + 0], s8_2[_lp*2+1 + 0]));
            s8_bf16_2[_lp] = *(uint32_t*)&_bf2;
        }
        rw[12] = s8_bf16_2[0];
        rw[13] = s8_bf16_2[1];
        rw[14] = s8_bf16_2[2];
        rw[15] = s8_bf16_2[3];
        float y8_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_2[_pair * 2])[0]), "=f"((&y8_2[_pair * 2])[1])
                : "r"(s8_bf16_2[_pair]));
        }
        float ssq_2 = 0.0f;
        float y_24 = ((mode != 0) ? y8_2[0] * gw8[0] : y8_2[0]);
        ssq_2 = __fmaf_rn(y_24, y_24, ssq_2);
        float y_25 = ((mode != 0) ? y8_2[1] * gw8[1] : y8_2[1]);
        ssq_2 = __fmaf_rn(y_25, y_25, ssq_2);
        float y_26 = ((mode != 0) ? y8_2[2] * gw8[2] : y8_2[2]);
        ssq_2 = __fmaf_rn(y_26, y_26, ssq_2);
        float y_27 = ((mode != 0) ? y8_2[3] * gw8[3] : y8_2[3]);
        ssq_2 = __fmaf_rn(y_27, y_27, ssq_2);
        float y_28 = ((mode != 0) ? y8_2[4] * gw8[4] : y8_2[4]);
        ssq_2 = __fmaf_rn(y_28, y_28, ssq_2);
        float y_29 = ((mode != 0) ? y8_2[5] * gw8[5] : y8_2[5]);
        ssq_2 = __fmaf_rn(y_29, y_29, ssq_2);
        float y_30 = ((mode != 0) ? y8_2[6] * gw8[6] : y8_2[6]);
        ssq_2 = __fmaf_rn(y_30, y_30, ssq_2);
        float y_31 = ((mode != 0) ? y8_2[7] * gw8[7] : y8_2[7]);
        ssq_2 = __fmaf_rn(y_31, y_31, ssq_2);
        ssq_k[3] = ssq_2;
        uint32_t _sysv_poll_group_1[64];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 4) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 4)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 4)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 4)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 5) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 5)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 5)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 5)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[32]), "=r"(_sysv_poll_group_1[33]), "=r"(_sysv_poll_group_1[34]), "=r"(_sysv_poll_group_1[35]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 6) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[36]), "=r"(_sysv_poll_group_1[37]), "=r"(_sysv_poll_group_1[38]), "=r"(_sysv_poll_group_1[39]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 6)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[40]), "=r"(_sysv_poll_group_1[41]), "=r"(_sysv_poll_group_1[42]), "=r"(_sysv_poll_group_1[43]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 6)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[44]), "=r"(_sysv_poll_group_1[45]), "=r"(_sysv_poll_group_1[46]), "=r"(_sysv_poll_group_1[47]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 6)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[48]), "=r"(_sysv_poll_group_1[49]), "=r"(_sysv_poll_group_1[50]), "=r"(_sysv_poll_group_1[51]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 7) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[52]), "=r"(_sysv_poll_group_1[53]), "=r"(_sysv_poll_group_1[54]), "=r"(_sysv_poll_group_1[55]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 7)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[56]), "=r"(_sysv_poll_group_1[57]), "=r"(_sysv_poll_group_1[58]), "=r"(_sysv_poll_group_1[59]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (16 + (xf + 7)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[60]), "=r"(_sysv_poll_group_1[61]), "=r"(_sysv_poll_group_1[62]), "=r"(_sysv_poll_group_1[63]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (24 + (xf + 7)) * 4096))) : "memory");
        } while (
            (((_sysv_poll_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[7] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[9] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[11] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[13] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[15] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[17] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[19] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[21] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[23] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[25] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[27] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[29] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[31] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[33] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[35] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[37] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[39] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[41] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[43] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[45] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[47] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[49] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[51] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[53] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[55] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[57] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[59] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[61] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[63] >> 16) & 0xffffu) == 0x8000u)
        );
        float u_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_16[_pair * 2])[0]), "=f"((&u_16[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[_pair]));
        }
        float u_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_17[_pair * 2])[0]), "=f"((&u_17[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[4 + _pair]));
        }
        float u_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_18[_pair * 2])[0]), "=f"((&u_18[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[8 + _pair]));
        }
        float u_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_19[_pair * 2])[0]), "=f"((&u_19[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[12 + _pair]));
        }
        float s8_3[8];
        float t_128 = (float)u_16[0];
        s8_3[0] = t_128;
        float t_129 = (float)u_17[0];
        s8_3[0] = s8_3[0] + t_129;
        float t_130 = (float)u_18[0];
        s8_3[0] = s8_3[0] + t_130;
        float t_131 = (float)u_19[0];
        s8_3[0] = s8_3[0] + t_131;
        float t_132 = (float)u_16[1];
        s8_3[1] = t_132;
        float t_133 = (float)u_17[1];
        s8_3[1] = s8_3[1] + t_133;
        float t_134 = (float)u_18[1];
        s8_3[1] = s8_3[1] + t_134;
        float t_135 = (float)u_19[1];
        s8_3[1] = s8_3[1] + t_135;
        float t_136 = (float)u_16[2];
        s8_3[2] = t_136;
        float t_137 = (float)u_17[2];
        s8_3[2] = s8_3[2] + t_137;
        float t_138 = (float)u_18[2];
        s8_3[2] = s8_3[2] + t_138;
        float t_139 = (float)u_19[2];
        s8_3[2] = s8_3[2] + t_139;
        float t_140 = (float)u_16[3];
        s8_3[3] = t_140;
        float t_141 = (float)u_17[3];
        s8_3[3] = s8_3[3] + t_141;
        float t_142 = (float)u_18[3];
        s8_3[3] = s8_3[3] + t_142;
        float t_143 = (float)u_19[3];
        s8_3[3] = s8_3[3] + t_143;
        float t_144 = (float)u_16[4];
        s8_3[4] = t_144;
        float t_145 = (float)u_17[4];
        s8_3[4] = s8_3[4] + t_145;
        float t_146 = (float)u_18[4];
        s8_3[4] = s8_3[4] + t_146;
        float t_147 = (float)u_19[4];
        s8_3[4] = s8_3[4] + t_147;
        float t_148 = (float)u_16[5];
        s8_3[5] = t_148;
        float t_149 = (float)u_17[5];
        s8_3[5] = s8_3[5] + t_149;
        float t_150 = (float)u_18[5];
        s8_3[5] = s8_3[5] + t_150;
        float t_151 = (float)u_19[5];
        s8_3[5] = s8_3[5] + t_151;
        float t_152 = (float)u_16[6];
        s8_3[6] = t_152;
        float t_153 = (float)u_17[6];
        s8_3[6] = s8_3[6] + t_153;
        float t_154 = (float)u_18[6];
        s8_3[6] = s8_3[6] + t_154;
        float t_155 = (float)u_19[6];
        s8_3[6] = s8_3[6] + t_155;
        float t_156 = (float)u_16[7];
        s8_3[7] = t_156;
        float t_157 = (float)u_17[7];
        s8_3[7] = s8_3[7] + t_157;
        float t_158 = (float)u_18[7];
        s8_3[7] = s8_3[7] + t_158;
        float t_159 = (float)u_19[7];
        s8_3[7] = s8_3[7] + t_159;
        unsigned int s8_bf16_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_3[_lp*2 + 0], s8_3[_lp*2+1 + 0]));
            s8_bf16_3[_lp] = *(uint32_t*)&_bf2;
        }
        rw[16] = s8_bf16_3[0];
        rw[17] = s8_bf16_3[1];
        rw[18] = s8_bf16_3[2];
        rw[19] = s8_bf16_3[3];
        float y8_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_3[_pair * 2])[0]), "=f"((&y8_3[_pair * 2])[1])
                : "r"(s8_bf16_3[_pair]));
        }
        float ssq_3 = 0.0f;
        float y_32 = ((mode != 0) ? y8_3[0] * gw8[0] : y8_3[0]);
        ssq_3 = __fmaf_rn(y_32, y_32, ssq_3);
        float y_33 = ((mode != 0) ? y8_3[1] * gw8[1] : y8_3[1]);
        ssq_3 = __fmaf_rn(y_33, y_33, ssq_3);
        float y_34 = ((mode != 0) ? y8_3[2] * gw8[2] : y8_3[2]);
        ssq_3 = __fmaf_rn(y_34, y_34, ssq_3);
        float y_35 = ((mode != 0) ? y8_3[3] * gw8[3] : y8_3[3]);
        ssq_3 = __fmaf_rn(y_35, y_35, ssq_3);
        float y_36 = ((mode != 0) ? y8_3[4] * gw8[4] : y8_3[4]);
        ssq_3 = __fmaf_rn(y_36, y_36, ssq_3);
        float y_37 = ((mode != 0) ? y8_3[5] * gw8[5] : y8_3[5]);
        ssq_3 = __fmaf_rn(y_37, y_37, ssq_3);
        float y_38 = ((mode != 0) ? y8_3[6] * gw8[6] : y8_3[6]);
        ssq_3 = __fmaf_rn(y_38, y_38, ssq_3);
        float y_39 = ((mode != 0) ? y8_3[7] * gw8[7] : y8_3[7]);
        ssq_3 = __fmaf_rn(y_39, y_39, ssq_3);
        ssq_k[4] = ssq_3;
        float u_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_20[_pair * 2])[0]), "=f"((&u_20[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[16 + _pair]));
        }
        float u_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_21[_pair * 2])[0]), "=f"((&u_21[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[20 + _pair]));
        }
        float u_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_22[_pair * 2])[0]), "=f"((&u_22[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[24 + _pair]));
        }
        float u_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_23[_pair * 2])[0]), "=f"((&u_23[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[28 + _pair]));
        }
        float s8_4[8];
        float t_160 = (float)u_20[0];
        s8_4[0] = t_160;
        float t_161 = (float)u_21[0];
        s8_4[0] = s8_4[0] + t_161;
        float t_162 = (float)u_22[0];
        s8_4[0] = s8_4[0] + t_162;
        float t_163 = (float)u_23[0];
        s8_4[0] = s8_4[0] + t_163;
        float t_164 = (float)u_20[1];
        s8_4[1] = t_164;
        float t_165 = (float)u_21[1];
        s8_4[1] = s8_4[1] + t_165;
        float t_166 = (float)u_22[1];
        s8_4[1] = s8_4[1] + t_166;
        float t_167 = (float)u_23[1];
        s8_4[1] = s8_4[1] + t_167;
        float t_168 = (float)u_20[2];
        s8_4[2] = t_168;
        float t_169 = (float)u_21[2];
        s8_4[2] = s8_4[2] + t_169;
        float t_170 = (float)u_22[2];
        s8_4[2] = s8_4[2] + t_170;
        float t_171 = (float)u_23[2];
        s8_4[2] = s8_4[2] + t_171;
        float t_172 = (float)u_20[3];
        s8_4[3] = t_172;
        float t_173 = (float)u_21[3];
        s8_4[3] = s8_4[3] + t_173;
        float t_174 = (float)u_22[3];
        s8_4[3] = s8_4[3] + t_174;
        float t_175 = (float)u_23[3];
        s8_4[3] = s8_4[3] + t_175;
        float t_176 = (float)u_20[4];
        s8_4[4] = t_176;
        float t_177 = (float)u_21[4];
        s8_4[4] = s8_4[4] + t_177;
        float t_178 = (float)u_22[4];
        s8_4[4] = s8_4[4] + t_178;
        float t_179 = (float)u_23[4];
        s8_4[4] = s8_4[4] + t_179;
        float t_180 = (float)u_20[5];
        s8_4[5] = t_180;
        float t_181 = (float)u_21[5];
        s8_4[5] = s8_4[5] + t_181;
        float t_182 = (float)u_22[5];
        s8_4[5] = s8_4[5] + t_182;
        float t_183 = (float)u_23[5];
        s8_4[5] = s8_4[5] + t_183;
        float t_184 = (float)u_20[6];
        s8_4[6] = t_184;
        float t_185 = (float)u_21[6];
        s8_4[6] = s8_4[6] + t_185;
        float t_186 = (float)u_22[6];
        s8_4[6] = s8_4[6] + t_186;
        float t_187 = (float)u_23[6];
        s8_4[6] = s8_4[6] + t_187;
        float t_188 = (float)u_20[7];
        s8_4[7] = t_188;
        float t_189 = (float)u_21[7];
        s8_4[7] = s8_4[7] + t_189;
        float t_190 = (float)u_22[7];
        s8_4[7] = s8_4[7] + t_190;
        float t_191 = (float)u_23[7];
        s8_4[7] = s8_4[7] + t_191;
        unsigned int s8_bf16_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_4[_lp*2 + 0], s8_4[_lp*2+1 + 0]));
            s8_bf16_4[_lp] = *(uint32_t*)&_bf2;
        }
        rw[20] = s8_bf16_4[0];
        rw[21] = s8_bf16_4[1];
        rw[22] = s8_bf16_4[2];
        rw[23] = s8_bf16_4[3];
        float y8_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_4[_pair * 2])[0]), "=f"((&y8_4[_pair * 2])[1])
                : "r"(s8_bf16_4[_pair]));
        }
        float ssq_4 = 0.0f;
        float y_40 = ((mode != 0) ? y8_4[0] * gw8[0] : y8_4[0]);
        ssq_4 = __fmaf_rn(y_40, y_40, ssq_4);
        float y_41 = ((mode != 0) ? y8_4[1] * gw8[1] : y8_4[1]);
        ssq_4 = __fmaf_rn(y_41, y_41, ssq_4);
        float y_42 = ((mode != 0) ? y8_4[2] * gw8[2] : y8_4[2]);
        ssq_4 = __fmaf_rn(y_42, y_42, ssq_4);
        float y_43 = ((mode != 0) ? y8_4[3] * gw8[3] : y8_4[3]);
        ssq_4 = __fmaf_rn(y_43, y_43, ssq_4);
        float y_44 = ((mode != 0) ? y8_4[4] * gw8[4] : y8_4[4]);
        ssq_4 = __fmaf_rn(y_44, y_44, ssq_4);
        float y_45 = ((mode != 0) ? y8_4[5] * gw8[5] : y8_4[5]);
        ssq_4 = __fmaf_rn(y_45, y_45, ssq_4);
        float y_46 = ((mode != 0) ? y8_4[6] * gw8[6] : y8_4[6]);
        ssq_4 = __fmaf_rn(y_46, y_46, ssq_4);
        float y_47 = ((mode != 0) ? y8_4[7] * gw8[7] : y8_4[7]);
        ssq_4 = __fmaf_rn(y_47, y_47, ssq_4);
        ssq_k[5] = ssq_4;
        float u_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_24[_pair * 2])[0]), "=f"((&u_24[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[32 + _pair]));
        }
        float u_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_25[_pair * 2])[0]), "=f"((&u_25[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[36 + _pair]));
        }
        float u_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_26[_pair * 2])[0]), "=f"((&u_26[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[40 + _pair]));
        }
        float u_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_27[_pair * 2])[0]), "=f"((&u_27[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[44 + _pair]));
        }
        float s8_5[8];
        float t_192 = (float)u_24[0];
        s8_5[0] = t_192;
        float t_193 = (float)u_25[0];
        s8_5[0] = s8_5[0] + t_193;
        float t_194 = (float)u_26[0];
        s8_5[0] = s8_5[0] + t_194;
        float t_195 = (float)u_27[0];
        s8_5[0] = s8_5[0] + t_195;
        float t_196 = (float)u_24[1];
        s8_5[1] = t_196;
        float t_197 = (float)u_25[1];
        s8_5[1] = s8_5[1] + t_197;
        float t_198 = (float)u_26[1];
        s8_5[1] = s8_5[1] + t_198;
        float t_199 = (float)u_27[1];
        s8_5[1] = s8_5[1] + t_199;
        float t_200 = (float)u_24[2];
        s8_5[2] = t_200;
        float t_201 = (float)u_25[2];
        s8_5[2] = s8_5[2] + t_201;
        float t_202 = (float)u_26[2];
        s8_5[2] = s8_5[2] + t_202;
        float t_203 = (float)u_27[2];
        s8_5[2] = s8_5[2] + t_203;
        float t_204 = (float)u_24[3];
        s8_5[3] = t_204;
        float t_205 = (float)u_25[3];
        s8_5[3] = s8_5[3] + t_205;
        float t_206 = (float)u_26[3];
        s8_5[3] = s8_5[3] + t_206;
        float t_207 = (float)u_27[3];
        s8_5[3] = s8_5[3] + t_207;
        float t_208 = (float)u_24[4];
        s8_5[4] = t_208;
        float t_209 = (float)u_25[4];
        s8_5[4] = s8_5[4] + t_209;
        float t_210 = (float)u_26[4];
        s8_5[4] = s8_5[4] + t_210;
        float t_211 = (float)u_27[4];
        s8_5[4] = s8_5[4] + t_211;
        float t_212 = (float)u_24[5];
        s8_5[5] = t_212;
        float t_213 = (float)u_25[5];
        s8_5[5] = s8_5[5] + t_213;
        float t_214 = (float)u_26[5];
        s8_5[5] = s8_5[5] + t_214;
        float t_215 = (float)u_27[5];
        s8_5[5] = s8_5[5] + t_215;
        float t_216 = (float)u_24[6];
        s8_5[6] = t_216;
        float t_217 = (float)u_25[6];
        s8_5[6] = s8_5[6] + t_217;
        float t_218 = (float)u_26[6];
        s8_5[6] = s8_5[6] + t_218;
        float t_219 = (float)u_27[6];
        s8_5[6] = s8_5[6] + t_219;
        float t_220 = (float)u_24[7];
        s8_5[7] = t_220;
        float t_221 = (float)u_25[7];
        s8_5[7] = s8_5[7] + t_221;
        float t_222 = (float)u_26[7];
        s8_5[7] = s8_5[7] + t_222;
        float t_223 = (float)u_27[7];
        s8_5[7] = s8_5[7] + t_223;
        unsigned int s8_bf16_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_5[_lp*2 + 0], s8_5[_lp*2+1 + 0]));
            s8_bf16_5[_lp] = *(uint32_t*)&_bf2;
        }
        rw[24] = s8_bf16_5[0];
        rw[25] = s8_bf16_5[1];
        rw[26] = s8_bf16_5[2];
        rw[27] = s8_bf16_5[3];
        float y8_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_5[_pair * 2])[0]), "=f"((&y8_5[_pair * 2])[1])
                : "r"(s8_bf16_5[_pair]));
        }
        float ssq_5 = 0.0f;
        float y_48 = ((mode != 0) ? y8_5[0] * gw8[0] : y8_5[0]);
        ssq_5 = __fmaf_rn(y_48, y_48, ssq_5);
        float y_49 = ((mode != 0) ? y8_5[1] * gw8[1] : y8_5[1]);
        ssq_5 = __fmaf_rn(y_49, y_49, ssq_5);
        float y_50 = ((mode != 0) ? y8_5[2] * gw8[2] : y8_5[2]);
        ssq_5 = __fmaf_rn(y_50, y_50, ssq_5);
        float y_51 = ((mode != 0) ? y8_5[3] * gw8[3] : y8_5[3]);
        ssq_5 = __fmaf_rn(y_51, y_51, ssq_5);
        float y_52 = ((mode != 0) ? y8_5[4] * gw8[4] : y8_5[4]);
        ssq_5 = __fmaf_rn(y_52, y_52, ssq_5);
        float y_53 = ((mode != 0) ? y8_5[5] * gw8[5] : y8_5[5]);
        ssq_5 = __fmaf_rn(y_53, y_53, ssq_5);
        float y_54 = ((mode != 0) ? y8_5[6] * gw8[6] : y8_5[6]);
        ssq_5 = __fmaf_rn(y_54, y_54, ssq_5);
        float y_55 = ((mode != 0) ? y8_5[7] * gw8[7] : y8_5[7]);
        ssq_5 = __fmaf_rn(y_55, y_55, ssq_5);
        ssq_k[6] = ssq_5;
        float u_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_28[_pair * 2])[0]), "=f"((&u_28[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[48 + _pair]));
        }
        float u_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_29[_pair * 2])[0]), "=f"((&u_29[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[52 + _pair]));
        }
        float u_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_30[_pair * 2])[0]), "=f"((&u_30[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[56 + _pair]));
        }
        float u_31[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&u_31[_pair * 2])[0]), "=f"((&u_31[_pair * 2])[1])
                : "r"(_sysv_poll_group_1[60 + _pair]));
        }
        float s8_6[8];
        float t_224 = (float)u_28[0];
        s8_6[0] = t_224;
        float t_225 = (float)u_29[0];
        s8_6[0] = s8_6[0] + t_225;
        float t_226 = (float)u_30[0];
        s8_6[0] = s8_6[0] + t_226;
        float t_227 = (float)u_31[0];
        s8_6[0] = s8_6[0] + t_227;
        float t_228 = (float)u_28[1];
        s8_6[1] = t_228;
        float t_229 = (float)u_29[1];
        s8_6[1] = s8_6[1] + t_229;
        float t_230 = (float)u_30[1];
        s8_6[1] = s8_6[1] + t_230;
        float t_231 = (float)u_31[1];
        s8_6[1] = s8_6[1] + t_231;
        float t_232 = (float)u_28[2];
        s8_6[2] = t_232;
        float t_233 = (float)u_29[2];
        s8_6[2] = s8_6[2] + t_233;
        float t_234 = (float)u_30[2];
        s8_6[2] = s8_6[2] + t_234;
        float t_235 = (float)u_31[2];
        s8_6[2] = s8_6[2] + t_235;
        float t_236 = (float)u_28[3];
        s8_6[3] = t_236;
        float t_237 = (float)u_29[3];
        s8_6[3] = s8_6[3] + t_237;
        float t_238 = (float)u_30[3];
        s8_6[3] = s8_6[3] + t_238;
        float t_239 = (float)u_31[3];
        s8_6[3] = s8_6[3] + t_239;
        float t_240 = (float)u_28[4];
        s8_6[4] = t_240;
        float t_241 = (float)u_29[4];
        s8_6[4] = s8_6[4] + t_241;
        float t_242 = (float)u_30[4];
        s8_6[4] = s8_6[4] + t_242;
        float t_243 = (float)u_31[4];
        s8_6[4] = s8_6[4] + t_243;
        float t_244 = (float)u_28[5];
        s8_6[5] = t_244;
        float t_245 = (float)u_29[5];
        s8_6[5] = s8_6[5] + t_245;
        float t_246 = (float)u_30[5];
        s8_6[5] = s8_6[5] + t_246;
        float t_247 = (float)u_31[5];
        s8_6[5] = s8_6[5] + t_247;
        float t_248 = (float)u_28[6];
        s8_6[6] = t_248;
        float t_249 = (float)u_29[6];
        s8_6[6] = s8_6[6] + t_249;
        float t_250 = (float)u_30[6];
        s8_6[6] = s8_6[6] + t_250;
        float t_251 = (float)u_31[6];
        s8_6[6] = s8_6[6] + t_251;
        float t_252 = (float)u_28[7];
        s8_6[7] = t_252;
        float t_253 = (float)u_29[7];
        s8_6[7] = s8_6[7] + t_253;
        float t_254 = (float)u_30[7];
        s8_6[7] = s8_6[7] + t_254;
        float t_255 = (float)u_31[7];
        s8_6[7] = s8_6[7] + t_255;
        unsigned int s8_bf16_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_6[_lp*2 + 0], s8_6[_lp*2+1 + 0]));
            s8_bf16_6[_lp] = *(uint32_t*)&_bf2;
        }
        rw[28] = s8_bf16_6[0];
        rw[29] = s8_bf16_6[1];
        rw[30] = s8_bf16_6[2];
        rw[31] = s8_bf16_6[3];
        float y8_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&y8_6[_pair * 2])[0]), "=f"((&y8_6[_pair * 2])[1])
                : "r"(s8_bf16_6[_pair]));
        }
        float ssq_6 = 0.0f;
        float y_56 = ((mode != 0) ? y8_6[0] * gw8[0] : y8_6[0]);
        ssq_6 = __fmaf_rn(y_56, y_56, ssq_6);
        float y_57 = ((mode != 0) ? y8_6[1] * gw8[1] : y8_6[1]);
        ssq_6 = __fmaf_rn(y_57, y_57, ssq_6);
        float y_58 = ((mode != 0) ? y8_6[2] * gw8[2] : y8_6[2]);
        ssq_6 = __fmaf_rn(y_58, y_58, ssq_6);
        float y_59 = ((mode != 0) ? y8_6[3] * gw8[3] : y8_6[3]);
        ssq_6 = __fmaf_rn(y_59, y_59, ssq_6);
        float y_60 = ((mode != 0) ? y8_6[4] * gw8[4] : y8_6[4]);
        ssq_6 = __fmaf_rn(y_60, y_60, ssq_6);
        float y_61 = ((mode != 0) ? y8_6[5] * gw8[5] : y8_6[5]);
        ssq_6 = __fmaf_rn(y_61, y_61, ssq_6);
        float y_62 = ((mode != 0) ? y8_6[6] * gw8[6] : y8_6[6]);
        ssq_6 = __fmaf_rn(y_62, y_62, ssq_6);
        float y_63 = ((mode != 0) ? y8_6[7] * gw8[7] : y8_6[7]);
        ssq_6 = __fmaf_rn(y_63, y_63, ssq_6);
        ssq_k[7] = ssq_6;
        float wr_0 = ssq_k[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_0 += __shfl_xor_sync(0xFFFFFFFF, wr_0, offset);
        if (lane == 0) {
            warp_tab[warp * 8] = wr_0;
        }
        float wr_1 = ssq_k[1];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_1 += __shfl_xor_sync(0xFFFFFFFF, wr_1, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 1] = wr_1;
        }
        float wr_2 = ssq_k[2];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_2 += __shfl_xor_sync(0xFFFFFFFF, wr_2, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 2] = wr_2;
        }
        float wr_3 = ssq_k[3];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_3 += __shfl_xor_sync(0xFFFFFFFF, wr_3, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 3] = wr_3;
        }
        float wr_4 = ssq_k[4];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_4 += __shfl_xor_sync(0xFFFFFFFF, wr_4, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 4] = wr_4;
        }
        float wr_5 = ssq_k[5];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_5 += __shfl_xor_sync(0xFFFFFFFF, wr_5, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 5] = wr_5;
        }
        float wr_6 = ssq_k[6];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_6 += __shfl_xor_sync(0xFFFFFFFF, wr_6, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 6] = wr_6;
        }
        float wr_7 = ssq_k[7];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wr_7 += __shfl_xor_sync(0xFFFFFFFF, wr_7, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 7] = wr_7;
        }
        __syncthreads();
        float tot_k[8];
        int kk_t = tid & 7;
        float part_t = warp_tab[kk_t];
        part_t = part_t + warp_tab[8 + kk_t];
        part_t = part_t + warp_tab[16 + kk_t];
        part_t = part_t + warp_tab[24 + kk_t];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (tid < 8) {
            unsigned int slot_addr = cta_tab_addr + (unsigned int)((cta_rank_0 * 8 + tid) * 4);
            uint32_t _cluster_mapa_39;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_39) : "r"(slot_addr), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_39), "f"(part_t) : "memory");
            uint32_t _cluster_mapa_40;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_40) : "r"(slot_addr), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_40), "f"(part_t) : "memory");
            uint32_t _cluster_mapa_41;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_41) : "r"(slot_addr), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_41), "f"(part_t) : "memory");
            uint32_t _cluster_mapa_42;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_42) : "r"(slot_addr), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_42), "f"(part_t) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        tot_k[0] = cta_tab[0];
        tot_k[0] = tot_k[0] + cta_tab[8];
        tot_k[0] = tot_k[0] + cta_tab[16];
        tot_k[0] = tot_k[0] + cta_tab[24];
        tot_k[1] = cta_tab[1];
        tot_k[1] = tot_k[1] + cta_tab[9];
        tot_k[1] = tot_k[1] + cta_tab[17];
        tot_k[1] = tot_k[1] + cta_tab[25];
        tot_k[2] = cta_tab[2];
        tot_k[2] = tot_k[2] + cta_tab[10];
        tot_k[2] = tot_k[2] + cta_tab[18];
        tot_k[2] = tot_k[2] + cta_tab[26];
        tot_k[3] = cta_tab[3];
        tot_k[3] = tot_k[3] + cta_tab[11];
        tot_k[3] = tot_k[3] + cta_tab[19];
        tot_k[3] = tot_k[3] + cta_tab[27];
        tot_k[4] = cta_tab[4];
        tot_k[4] = tot_k[4] + cta_tab[12];
        tot_k[4] = tot_k[4] + cta_tab[20];
        tot_k[4] = tot_k[4] + cta_tab[28];
        tot_k[5] = cta_tab[5];
        tot_k[5] = tot_k[5] + cta_tab[13];
        tot_k[5] = tot_k[5] + cta_tab[21];
        tot_k[5] = tot_k[5] + cta_tab[29];
        tot_k[6] = cta_tab[6];
        tot_k[6] = tot_k[6] + cta_tab[14];
        tot_k[6] = tot_k[6] + cta_tab[22];
        tot_k[6] = tot_k[6] + cta_tab[30];
        tot_k[7] = cta_tab[7];
        tot_k[7] = tot_k[7] + cta_tab[15];
        tot_k[7] = tot_k[7] + cta_tab[23];
        tot_k[7] = tot_k[7] + cta_tab[31];
        float pp[8];
        float acc[8];
        float _sqrt_43;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_43) : "f"(__fmaf_rn(tot_k[0], 0.00024414062f, eps)));
        float rs = __frcp_rn(_sqrt_43);
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float yu[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu[_pair * 2])[0]), "=f"((&yu[_pair * 2])[1])
                : "r"(rw[_pair]));
        }
        float yn_0 = ((mode != 0) ? yu[0] * gw8[0] : yu[0]);
        float o_0 = yn_0 * rs;
        float o2_0 = ((mode != 0) ? o_0 : o_0 * gw8[0]);
        pp[0] = __fmaf_rn(o2_0, wk[0], pp[0]);
        float yn_1 = ((mode != 0) ? yu[1] * gw8[1] : yu[1]);
        float o_1 = yn_1 * rs;
        float o2_1 = ((mode != 0) ? o_1 : o_1 * gw8[1]);
        pp[1] = __fmaf_rn(o2_1, wk[0], pp[1]);
        float yn_2 = ((mode != 0) ? yu[2] * gw8[2] : yu[2]);
        float o_2 = yn_2 * rs;
        float o2_2 = ((mode != 0) ? o_2 : o_2 * gw8[2]);
        pp[2] = __fmaf_rn(o2_2, wk[0], pp[2]);
        float yn_3 = ((mode != 0) ? yu[3] * gw8[3] : yu[3]);
        float o_3 = yn_3 * rs;
        float o2_3 = ((mode != 0) ? o_3 : o_3 * gw8[3]);
        pp[3] = __fmaf_rn(o2_3, wk[0], pp[3]);
        float yn_4 = ((mode != 0) ? yu[4] * gw8[4] : yu[4]);
        float o_4 = yn_4 * rs;
        float o2_4 = ((mode != 0) ? o_4 : o_4 * gw8[4]);
        pp[4] = __fmaf_rn(o2_4, wk[0], pp[4]);
        float yn_5 = ((mode != 0) ? yu[5] * gw8[5] : yu[5]);
        float o_5 = yn_5 * rs;
        float o2_5 = ((mode != 0) ? o_5 : o_5 * gw8[5]);
        pp[5] = __fmaf_rn(o2_5, wk[0], pp[5]);
        float yn_6 = ((mode != 0) ? yu[6] * gw8[6] : yu[6]);
        float o_6 = yn_6 * rs;
        float o2_6 = ((mode != 0) ? o_6 : o_6 * gw8[6]);
        pp[6] = __fmaf_rn(o2_6, wk[0], pp[6]);
        float yn_7 = ((mode != 0) ? yu[7] * gw8[7] : yu[7]);
        float o_7 = yn_7 * rs;
        float o2_7 = ((mode != 0) ? o_7 : o_7 * gw8[7]);
        pp[7] = __fmaf_rn(o2_7, wk[0], pp[7]);
        float _sqrt_44;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_44) : "f"(__fmaf_rn(tot_k[1], 0.00024414062f, eps)));
        float rs_0 = __frcp_rn(_sqrt_44);
        float yu_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_0[_pair * 2])[0]), "=f"((&yu_0[_pair * 2])[1])
                : "r"(rw[4 + _pair]));
        }
        float yn_8 = ((mode != 0) ? yu_0[0] * gw8[0] : yu_0[0]);
        float o_8 = yn_8 * rs_0;
        float o2_8 = ((mode != 0) ? o_8 : o_8 * gw8[0]);
        pp[0] = __fmaf_rn(o2_8, wk[1], pp[0]);
        float yn_9 = ((mode != 0) ? yu_0[1] * gw8[1] : yu_0[1]);
        float o_9 = yn_9 * rs_0;
        float o2_9 = ((mode != 0) ? o_9 : o_9 * gw8[1]);
        pp[1] = __fmaf_rn(o2_9, wk[1], pp[1]);
        float yn_10 = ((mode != 0) ? yu_0[2] * gw8[2] : yu_0[2]);
        float o_10 = yn_10 * rs_0;
        float o2_10 = ((mode != 0) ? o_10 : o_10 * gw8[2]);
        pp[2] = __fmaf_rn(o2_10, wk[1], pp[2]);
        float yn_11 = ((mode != 0) ? yu_0[3] * gw8[3] : yu_0[3]);
        float o_11 = yn_11 * rs_0;
        float o2_11 = ((mode != 0) ? o_11 : o_11 * gw8[3]);
        pp[3] = __fmaf_rn(o2_11, wk[1], pp[3]);
        float yn_12 = ((mode != 0) ? yu_0[4] * gw8[4] : yu_0[4]);
        float o_12 = yn_12 * rs_0;
        float o2_12 = ((mode != 0) ? o_12 : o_12 * gw8[4]);
        pp[4] = __fmaf_rn(o2_12, wk[1], pp[4]);
        float yn_13 = ((mode != 0) ? yu_0[5] * gw8[5] : yu_0[5]);
        float o_13 = yn_13 * rs_0;
        float o2_13 = ((mode != 0) ? o_13 : o_13 * gw8[5]);
        pp[5] = __fmaf_rn(o2_13, wk[1], pp[5]);
        float yn_14 = ((mode != 0) ? yu_0[6] * gw8[6] : yu_0[6]);
        float o_14 = yn_14 * rs_0;
        float o2_14 = ((mode != 0) ? o_14 : o_14 * gw8[6]);
        pp[6] = __fmaf_rn(o2_14, wk[1], pp[6]);
        float yn_15 = ((mode != 0) ? yu_0[7] * gw8[7] : yu_0[7]);
        float o_15 = yn_15 * rs_0;
        float o2_15 = ((mode != 0) ? o_15 : o_15 * gw8[7]);
        pp[7] = __fmaf_rn(o2_15, wk[1], pp[7]);
        acc[0] = pp[0];
        acc[1] = pp[1];
        acc[2] = pp[2];
        acc[3] = pp[3];
        acc[4] = pp[4];
        acc[5] = pp[5];
        acc[6] = pp[6];
        acc[7] = pp[7];
        float _sqrt_45;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_45) : "f"(__fmaf_rn(tot_k[2], 0.00024414062f, eps)));
        float rs_1 = __frcp_rn(_sqrt_45);
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float yu_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_1[_pair * 2])[0]), "=f"((&yu_1[_pair * 2])[1])
                : "r"(rw[8 + _pair]));
        }
        float yn_16 = ((mode != 0) ? yu_1[0] * gw8[0] : yu_1[0]);
        float o_16 = yn_16 * rs_1;
        float o2_16 = ((mode != 0) ? o_16 : o_16 * gw8[0]);
        pp[0] = __fmaf_rn(o2_16, wk[2], pp[0]);
        float yn_17 = ((mode != 0) ? yu_1[1] * gw8[1] : yu_1[1]);
        float o_17 = yn_17 * rs_1;
        float o2_17 = ((mode != 0) ? o_17 : o_17 * gw8[1]);
        pp[1] = __fmaf_rn(o2_17, wk[2], pp[1]);
        float yn_18 = ((mode != 0) ? yu_1[2] * gw8[2] : yu_1[2]);
        float o_18 = yn_18 * rs_1;
        float o2_18 = ((mode != 0) ? o_18 : o_18 * gw8[2]);
        pp[2] = __fmaf_rn(o2_18, wk[2], pp[2]);
        float yn_19 = ((mode != 0) ? yu_1[3] * gw8[3] : yu_1[3]);
        float o_19 = yn_19 * rs_1;
        float o2_19 = ((mode != 0) ? o_19 : o_19 * gw8[3]);
        pp[3] = __fmaf_rn(o2_19, wk[2], pp[3]);
        float yn_20 = ((mode != 0) ? yu_1[4] * gw8[4] : yu_1[4]);
        float o_20 = yn_20 * rs_1;
        float o2_20 = ((mode != 0) ? o_20 : o_20 * gw8[4]);
        pp[4] = __fmaf_rn(o2_20, wk[2], pp[4]);
        float yn_21 = ((mode != 0) ? yu_1[5] * gw8[5] : yu_1[5]);
        float o_21 = yn_21 * rs_1;
        float o2_21 = ((mode != 0) ? o_21 : o_21 * gw8[5]);
        pp[5] = __fmaf_rn(o2_21, wk[2], pp[5]);
        float yn_22 = ((mode != 0) ? yu_1[6] * gw8[6] : yu_1[6]);
        float o_22 = yn_22 * rs_1;
        float o2_22 = ((mode != 0) ? o_22 : o_22 * gw8[6]);
        pp[6] = __fmaf_rn(o2_22, wk[2], pp[6]);
        float yn_23 = ((mode != 0) ? yu_1[7] * gw8[7] : yu_1[7]);
        float o_23 = yn_23 * rs_1;
        float o2_23 = ((mode != 0) ? o_23 : o_23 * gw8[7]);
        pp[7] = __fmaf_rn(o2_23, wk[2], pp[7]);
        float _sqrt_46;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_46) : "f"(__fmaf_rn(tot_k[3], 0.00024414062f, eps)));
        float rs_2 = __frcp_rn(_sqrt_46);
        float yu_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_2[_pair * 2])[0]), "=f"((&yu_2[_pair * 2])[1])
                : "r"(rw[12 + _pair]));
        }
        float yn_24 = ((mode != 0) ? yu_2[0] * gw8[0] : yu_2[0]);
        float o_24 = yn_24 * rs_2;
        float o2_24 = ((mode != 0) ? o_24 : o_24 * gw8[0]);
        pp[0] = __fmaf_rn(o2_24, wk[3], pp[0]);
        float yn_25 = ((mode != 0) ? yu_2[1] * gw8[1] : yu_2[1]);
        float o_25 = yn_25 * rs_2;
        float o2_25 = ((mode != 0) ? o_25 : o_25 * gw8[1]);
        pp[1] = __fmaf_rn(o2_25, wk[3], pp[1]);
        float yn_26 = ((mode != 0) ? yu_2[2] * gw8[2] : yu_2[2]);
        float o_26 = yn_26 * rs_2;
        float o2_26 = ((mode != 0) ? o_26 : o_26 * gw8[2]);
        pp[2] = __fmaf_rn(o2_26, wk[3], pp[2]);
        float yn_27 = ((mode != 0) ? yu_2[3] * gw8[3] : yu_2[3]);
        float o_27 = yn_27 * rs_2;
        float o2_27 = ((mode != 0) ? o_27 : o_27 * gw8[3]);
        pp[3] = __fmaf_rn(o2_27, wk[3], pp[3]);
        float yn_28 = ((mode != 0) ? yu_2[4] * gw8[4] : yu_2[4]);
        float o_28 = yn_28 * rs_2;
        float o2_28 = ((mode != 0) ? o_28 : o_28 * gw8[4]);
        pp[4] = __fmaf_rn(o2_28, wk[3], pp[4]);
        float yn_29 = ((mode != 0) ? yu_2[5] * gw8[5] : yu_2[5]);
        float o_29 = yn_29 * rs_2;
        float o2_29 = ((mode != 0) ? o_29 : o_29 * gw8[5]);
        pp[5] = __fmaf_rn(o2_29, wk[3], pp[5]);
        float yn_30 = ((mode != 0) ? yu_2[6] * gw8[6] : yu_2[6]);
        float o_30 = yn_30 * rs_2;
        float o2_30 = ((mode != 0) ? o_30 : o_30 * gw8[6]);
        pp[6] = __fmaf_rn(o2_30, wk[3], pp[6]);
        float yn_31 = ((mode != 0) ? yu_2[7] * gw8[7] : yu_2[7]);
        float o_31 = yn_31 * rs_2;
        float o2_31 = ((mode != 0) ? o_31 : o_31 * gw8[7]);
        pp[7] = __fmaf_rn(o2_31, wk[3], pp[7]);
        acc[0] = acc[0] + pp[0];
        acc[1] = acc[1] + pp[1];
        acc[2] = acc[2] + pp[2];
        acc[3] = acc[3] + pp[3];
        acc[4] = acc[4] + pp[4];
        acc[5] = acc[5] + pp[5];
        acc[6] = acc[6] + pp[6];
        acc[7] = acc[7] + pp[7];
        float _sqrt_47;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_47) : "f"(__fmaf_rn(tot_k[4], 0.00024414062f, eps)));
        float rs_3 = __frcp_rn(_sqrt_47);
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float yu_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_3[_pair * 2])[0]), "=f"((&yu_3[_pair * 2])[1])
                : "r"(rw[16 + _pair]));
        }
        float yn_32 = ((mode != 0) ? yu_3[0] * gw8[0] : yu_3[0]);
        float o_32 = yn_32 * rs_3;
        float o2_32 = ((mode != 0) ? o_32 : o_32 * gw8[0]);
        pp[0] = __fmaf_rn(o2_32, wk[4], pp[0]);
        float yn_33 = ((mode != 0) ? yu_3[1] * gw8[1] : yu_3[1]);
        float o_33 = yn_33 * rs_3;
        float o2_33 = ((mode != 0) ? o_33 : o_33 * gw8[1]);
        pp[1] = __fmaf_rn(o2_33, wk[4], pp[1]);
        float yn_34 = ((mode != 0) ? yu_3[2] * gw8[2] : yu_3[2]);
        float o_34 = yn_34 * rs_3;
        float o2_34 = ((mode != 0) ? o_34 : o_34 * gw8[2]);
        pp[2] = __fmaf_rn(o2_34, wk[4], pp[2]);
        float yn_35 = ((mode != 0) ? yu_3[3] * gw8[3] : yu_3[3]);
        float o_35 = yn_35 * rs_3;
        float o2_35 = ((mode != 0) ? o_35 : o_35 * gw8[3]);
        pp[3] = __fmaf_rn(o2_35, wk[4], pp[3]);
        float yn_36 = ((mode != 0) ? yu_3[4] * gw8[4] : yu_3[4]);
        float o_36 = yn_36 * rs_3;
        float o2_36 = ((mode != 0) ? o_36 : o_36 * gw8[4]);
        pp[4] = __fmaf_rn(o2_36, wk[4], pp[4]);
        float yn_37 = ((mode != 0) ? yu_3[5] * gw8[5] : yu_3[5]);
        float o_37 = yn_37 * rs_3;
        float o2_37 = ((mode != 0) ? o_37 : o_37 * gw8[5]);
        pp[5] = __fmaf_rn(o2_37, wk[4], pp[5]);
        float yn_38 = ((mode != 0) ? yu_3[6] * gw8[6] : yu_3[6]);
        float o_38 = yn_38 * rs_3;
        float o2_38 = ((mode != 0) ? o_38 : o_38 * gw8[6]);
        pp[6] = __fmaf_rn(o2_38, wk[4], pp[6]);
        float yn_39 = ((mode != 0) ? yu_3[7] * gw8[7] : yu_3[7]);
        float o_39 = yn_39 * rs_3;
        float o2_39 = ((mode != 0) ? o_39 : o_39 * gw8[7]);
        pp[7] = __fmaf_rn(o2_39, wk[4], pp[7]);
        float _sqrt_48;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_48) : "f"(__fmaf_rn(tot_k[5], 0.00024414062f, eps)));
        float rs_4 = __frcp_rn(_sqrt_48);
        float yu_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_4[_pair * 2])[0]), "=f"((&yu_4[_pair * 2])[1])
                : "r"(rw[20 + _pair]));
        }
        float yn_40 = ((mode != 0) ? yu_4[0] * gw8[0] : yu_4[0]);
        float o_40 = yn_40 * rs_4;
        float o2_40 = ((mode != 0) ? o_40 : o_40 * gw8[0]);
        pp[0] = __fmaf_rn(o2_40, wk[5], pp[0]);
        float yn_41 = ((mode != 0) ? yu_4[1] * gw8[1] : yu_4[1]);
        float o_41 = yn_41 * rs_4;
        float o2_41 = ((mode != 0) ? o_41 : o_41 * gw8[1]);
        pp[1] = __fmaf_rn(o2_41, wk[5], pp[1]);
        float yn_42 = ((mode != 0) ? yu_4[2] * gw8[2] : yu_4[2]);
        float o_42 = yn_42 * rs_4;
        float o2_42 = ((mode != 0) ? o_42 : o_42 * gw8[2]);
        pp[2] = __fmaf_rn(o2_42, wk[5], pp[2]);
        float yn_43 = ((mode != 0) ? yu_4[3] * gw8[3] : yu_4[3]);
        float o_43 = yn_43 * rs_4;
        float o2_43 = ((mode != 0) ? o_43 : o_43 * gw8[3]);
        pp[3] = __fmaf_rn(o2_43, wk[5], pp[3]);
        float yn_44 = ((mode != 0) ? yu_4[4] * gw8[4] : yu_4[4]);
        float o_44 = yn_44 * rs_4;
        float o2_44 = ((mode != 0) ? o_44 : o_44 * gw8[4]);
        pp[4] = __fmaf_rn(o2_44, wk[5], pp[4]);
        float yn_45 = ((mode != 0) ? yu_4[5] * gw8[5] : yu_4[5]);
        float o_45 = yn_45 * rs_4;
        float o2_45 = ((mode != 0) ? o_45 : o_45 * gw8[5]);
        pp[5] = __fmaf_rn(o2_45, wk[5], pp[5]);
        float yn_46 = ((mode != 0) ? yu_4[6] * gw8[6] : yu_4[6]);
        float o_46 = yn_46 * rs_4;
        float o2_46 = ((mode != 0) ? o_46 : o_46 * gw8[6]);
        pp[6] = __fmaf_rn(o2_46, wk[5], pp[6]);
        float yn_47 = ((mode != 0) ? yu_4[7] * gw8[7] : yu_4[7]);
        float o_47 = yn_47 * rs_4;
        float o2_47 = ((mode != 0) ? o_47 : o_47 * gw8[7]);
        pp[7] = __fmaf_rn(o2_47, wk[5], pp[7]);
        acc[0] = acc[0] + pp[0];
        acc[1] = acc[1] + pp[1];
        acc[2] = acc[2] + pp[2];
        acc[3] = acc[3] + pp[3];
        acc[4] = acc[4] + pp[4];
        acc[5] = acc[5] + pp[5];
        acc[6] = acc[6] + pp[6];
        acc[7] = acc[7] + pp[7];
        float _sqrt_49;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_49) : "f"(__fmaf_rn(tot_k[6], 0.00024414062f, eps)));
        float rs_5 = __frcp_rn(_sqrt_49);
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float yu_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_5[_pair * 2])[0]), "=f"((&yu_5[_pair * 2])[1])
                : "r"(rw[24 + _pair]));
        }
        float yn_48 = ((mode != 0) ? yu_5[0] * gw8[0] : yu_5[0]);
        float o_48 = yn_48 * rs_5;
        float o2_48 = ((mode != 0) ? o_48 : o_48 * gw8[0]);
        pp[0] = __fmaf_rn(o2_48, wk[6], pp[0]);
        float yn_49 = ((mode != 0) ? yu_5[1] * gw8[1] : yu_5[1]);
        float o_49 = yn_49 * rs_5;
        float o2_49 = ((mode != 0) ? o_49 : o_49 * gw8[1]);
        pp[1] = __fmaf_rn(o2_49, wk[6], pp[1]);
        float yn_50 = ((mode != 0) ? yu_5[2] * gw8[2] : yu_5[2]);
        float o_50 = yn_50 * rs_5;
        float o2_50 = ((mode != 0) ? o_50 : o_50 * gw8[2]);
        pp[2] = __fmaf_rn(o2_50, wk[6], pp[2]);
        float yn_51 = ((mode != 0) ? yu_5[3] * gw8[3] : yu_5[3]);
        float o_51 = yn_51 * rs_5;
        float o2_51 = ((mode != 0) ? o_51 : o_51 * gw8[3]);
        pp[3] = __fmaf_rn(o2_51, wk[6], pp[3]);
        float yn_52 = ((mode != 0) ? yu_5[4] * gw8[4] : yu_5[4]);
        float o_52 = yn_52 * rs_5;
        float o2_52 = ((mode != 0) ? o_52 : o_52 * gw8[4]);
        pp[4] = __fmaf_rn(o2_52, wk[6], pp[4]);
        float yn_53 = ((mode != 0) ? yu_5[5] * gw8[5] : yu_5[5]);
        float o_53 = yn_53 * rs_5;
        float o2_53 = ((mode != 0) ? o_53 : o_53 * gw8[5]);
        pp[5] = __fmaf_rn(o2_53, wk[6], pp[5]);
        float yn_54 = ((mode != 0) ? yu_5[6] * gw8[6] : yu_5[6]);
        float o_54 = yn_54 * rs_5;
        float o2_54 = ((mode != 0) ? o_54 : o_54 * gw8[6]);
        pp[6] = __fmaf_rn(o2_54, wk[6], pp[6]);
        float yn_55 = ((mode != 0) ? yu_5[7] * gw8[7] : yu_5[7]);
        float o_55 = yn_55 * rs_5;
        float o2_55 = ((mode != 0) ? o_55 : o_55 * gw8[7]);
        pp[7] = __fmaf_rn(o2_55, wk[6], pp[7]);
        float _sqrt_50;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_50) : "f"(__fmaf_rn(tot_k[7], 0.00024414062f, eps)));
        float rs_6 = __frcp_rn(_sqrt_50);
        float yu_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&yu_6[_pair * 2])[0]), "=f"((&yu_6[_pair * 2])[1])
                : "r"(rw[28 + _pair]));
        }
        float yn_56 = ((mode != 0) ? yu_6[0] * gw8[0] : yu_6[0]);
        float o_56 = yn_56 * rs_6;
        float o2_56 = ((mode != 0) ? o_56 : o_56 * gw8[0]);
        pp[0] = __fmaf_rn(o2_56, wk[7], pp[0]);
        float yn_57 = ((mode != 0) ? yu_6[1] * gw8[1] : yu_6[1]);
        float o_57 = yn_57 * rs_6;
        float o2_57 = ((mode != 0) ? o_57 : o_57 * gw8[1]);
        pp[1] = __fmaf_rn(o2_57, wk[7], pp[1]);
        float yn_58 = ((mode != 0) ? yu_6[2] * gw8[2] : yu_6[2]);
        float o_58 = yn_58 * rs_6;
        float o2_58 = ((mode != 0) ? o_58 : o_58 * gw8[2]);
        pp[2] = __fmaf_rn(o2_58, wk[7], pp[2]);
        float yn_59 = ((mode != 0) ? yu_6[3] * gw8[3] : yu_6[3]);
        float o_59 = yn_59 * rs_6;
        float o2_59 = ((mode != 0) ? o_59 : o_59 * gw8[3]);
        pp[3] = __fmaf_rn(o2_59, wk[7], pp[3]);
        float yn_60 = ((mode != 0) ? yu_6[4] * gw8[4] : yu_6[4]);
        float o_60 = yn_60 * rs_6;
        float o2_60 = ((mode != 0) ? o_60 : o_60 * gw8[4]);
        pp[4] = __fmaf_rn(o2_60, wk[7], pp[4]);
        float yn_61 = ((mode != 0) ? yu_6[5] * gw8[5] : yu_6[5]);
        float o_61 = yn_61 * rs_6;
        float o2_61 = ((mode != 0) ? o_61 : o_61 * gw8[5]);
        pp[5] = __fmaf_rn(o2_61, wk[7], pp[5]);
        float yn_62 = ((mode != 0) ? yu_6[6] * gw8[6] : yu_6[6]);
        float o_62 = yn_62 * rs_6;
        float o2_62 = ((mode != 0) ? o_62 : o_62 * gw8[6]);
        pp[6] = __fmaf_rn(o2_62, wk[7], pp[6]);
        float yn_63 = ((mode != 0) ? yu_6[7] * gw8[7] : yu_6[7]);
        float o_63 = yn_63 * rs_6;
        float o2_63 = ((mode != 0) ? o_63 : o_63 * gw8[7]);
        pp[7] = __fmaf_rn(o2_63, wk[7], pp[7]);
        acc[0] = acc[0] + pp[0];
        acc[1] = acc[1] + pp[1];
        acc[2] = acc[2] + pp[2];
        acc[3] = acc[3] + pp[3];
        acc[4] = acc[4] + pp[4];
        acc[5] = acc[5] + pp[5];
        acc[6] = acc[6] + pp[6];
        acc[7] = acc[7] + pp[7];
        unsigned int acc_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            acc_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output + ((long long)tok_own * 4096 + col64)))[0] = reinterpret_cast<int4*>(acc_bf16)[0];
        uint32_t _packed16_sanitize_8[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_51 = static_cast<uint32_t>(acc_bf16[_p16s_word]);
            uint32_t _p16s_lo_51 = _p16s_carrier_51 & 0x0000ffffu;
            uint32_t _p16s_hi_51 = (_p16s_carrier_51 >> 16) & 0x0000ffffu;
            if (_p16s_lo_51 == 0x00008000u) _p16s_lo_51 = 0u;
            if (_p16s_hi_51 == 0x00008000u) _p16s_hi_51 = 0u;
            _packed16_sanitize_8[_p16s_word] = _p16s_lo_51 | (_p16s_hi_51 << 16);
        }
        unsigned long long mc = r_mcast + (unsigned long long)((r_cur + rank64 * 4096) * 2);
        asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<float*>(mc)), "r"(_packed16_sanitize_8[0]), "r"(_packed16_sanitize_8[1]), "r"(_packed16_sanitize_8[2]), "r"(_packed16_sanitize_8[3]) : "memory");
    } else {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    if (tok_push < n_tok && d != rank_i) {
        uint32_t _sysv_poll_group_2[4];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[0]), "=r"(_sysv_poll_group_2[1]), "=r"(_sysv_poll_group_2[2]), "=r"(_sysv_poll_group_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(r_local + (r_cur + dest64 * 4096))) : "memory");
        } while (
            (((_sysv_poll_group_2[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_2[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[3] >> 16) & 0xffffu) == 0x8000u)
        );
        unsigned int row4[4];
        row4[0] = _sysv_poll_group_2[0];
        row4[1] = _sysv_poll_group_2[1];
        row4[2] = _sysv_poll_group_2[2];
        row4[3] = _sysv_poll_group_2[3];
        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output + ((long long)tok_push * 4096 + col64)))[0] = reinterpret_cast<int4*>(row4)[0];
    }
    int gn_next = ((gn == 2) ? 0 : gn + 1);
    if (tid == 0 && cta_rank_0 == 0) {
        *(reinterpret_cast<int*>(gen + gidx) + (0)) = gn_next;
    }
}

} // extern "C"
