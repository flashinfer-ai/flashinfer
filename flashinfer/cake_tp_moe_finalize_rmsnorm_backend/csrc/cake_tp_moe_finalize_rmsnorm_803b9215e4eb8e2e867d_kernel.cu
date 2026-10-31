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
kernel_cake_tp_moe_finalize_rmsnorm_803b9215e4eb8e2e867d(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen, float* __restrict__ part, unsigned int* __restrict__ ctr, unsigned long long r_mcast)
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
    int tok_push = g * 2 + d;
    int tok_p = ((tok_push < n_tok) ? tok_push : last);
    int tok_own = g * 2 + rank_i;
    int tok_o = ((tok_own < n_tok) ? tok_own : last);
    int gidx = g * 2 + d;
    int gn = gen[gidx];
    int gn_old = ((gn == 0) ? 2 : gn - 1);
    long long cur_slot = (long long)gn * (long long)slots + slot64;
    long long old_slot = (long long)gn_old * (long long)slots + slot64;
    long long a_cur = cur_slot * 65536 + col64;
    long long a_old = old_slot * 65536 + col64;
    long long r_cur = cur_slot * 8192 + col64;
    long long r_old = old_slot * 8192 + col64;
    long long a_local_addr = peer_table[rank];
    long long r_local_addr = peer_table[2 + rank];
    __nv_bfloat16* a_local = reinterpret_cast<__nv_bfloat16*>(a_local_addr);
    __nv_bfloat16* r_local = reinterpret_cast<__nv_bfloat16*>(r_local_addr);
    long long a_dest_addr = peer_table[d];
    __nv_bfloat16* a_dest = reinterpret_cast<__nv_bfloat16*>(a_dest_addr);
    long long r_peer_addr0 = peer_table[2];
    long long r_peer_addr1 = peer_table[3];
    __nv_bfloat16* r_peer0 = reinterpret_cast<__nv_bfloat16*>(r_peer_addr0);
    __nv_bfloat16* r_peer1 = reinterpret_cast<__nv_bfloat16*>(r_peer_addr1);
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
    int ridx0 = idx8[0];
    if (ridx0 >= 0 && ridx0 < rows) {
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx0 * (long long)row_stride + col64) + 0);
            packets[0 + 0] = _uv4_4.x;
            packets[0 + 1] = _uv4_4.y;
            packets[0 + 2] = _uv4_4.z;
            packets[0 + 3] = _uv4_4.w;
        }
    }
    int ridx_0 = idx8[1];
    if (ridx_0 >= 0 && ridx_0 < rows) {
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_0 * (long long)row_stride + col64) + 0);
            packets[4 + 0] = _uv4_5.x;
            packets[4 + 1] = _uv4_5.y;
            packets[4 + 2] = _uv4_5.z;
            packets[4 + 3] = _uv4_5.w;
        }
    }
    int ridx_1 = idx8[2];
    if (ridx_1 >= 0 && ridx_1 < rows) {
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_1 * (long long)row_stride + col64) + 0);
            packets[8 + 0] = _uv4_6.x;
            packets[8 + 1] = _uv4_6.y;
            packets[8 + 2] = _uv4_6.z;
            packets[8 + 3] = _uv4_6.w;
        }
    }
    int ridx_2 = idx8[3];
    if (ridx_2 >= 0 && ridx_2 < rows) {
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_2 * (long long)row_stride + col64) + 0);
            packets[12 + 0] = _uv4_7.x;
            packets[12 + 1] = _uv4_7.y;
            packets[12 + 2] = _uv4_7.z;
            packets[12 + 3] = _uv4_7.w;
        }
    }
    int ridx_3 = idx8[4];
    if (ridx_3 >= 0 && ridx_3 < rows) {
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_3 * (long long)row_stride + col64) + 0);
            packets[16 + 0] = _uv4_8.x;
            packets[16 + 1] = _uv4_8.y;
            packets[16 + 2] = _uv4_8.z;
            packets[16 + 3] = _uv4_8.w;
        }
    }
    int ridx_4 = idx8[5];
    if (ridx_4 >= 0 && ridx_4 < rows) {
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_4 * (long long)row_stride + col64) + 0);
            packets[20 + 0] = _uv4_9.x;
            packets[20 + 1] = _uv4_9.y;
            packets[20 + 2] = _uv4_9.z;
            packets[20 + 3] = _uv4_9.w;
        }
    }
    int ridx_5 = idx8[6];
    if (ridx_5 >= 0 && ridx_5 < rows) {
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_5 * (long long)row_stride + col64) + 0);
            packets[24 + 0] = _uv4_10.x;
            packets[24 + 1] = _uv4_10.y;
            packets[24 + 2] = _uv4_10.z;
            packets[24 + 3] = _uv4_10.w;
        }
    }
    int ridx_6 = idx8[7];
    if (ridx_6 >= 0 && ridx_6 < rows) {
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_6 * (long long)row_stride + col64) + 0);
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
#if __CUDA_ARCH__ == 1030
#else
        uint32_t _sysv_pending_39 = 65535u;
#endif
        do {
#if __CUDA_ARCH__ == 1030
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + xf * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + xf) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 1) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 1)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 2) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 2)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 3) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 3)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[32]), "=r"(_sysv_poll_group_0[33]), "=r"(_sysv_poll_group_0[34]), "=r"(_sysv_poll_group_0[35]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 4) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[36]), "=r"(_sysv_poll_group_0[37]), "=r"(_sysv_poll_group_0[38]), "=r"(_sysv_poll_group_0[39]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 4)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[40]), "=r"(_sysv_poll_group_0[41]), "=r"(_sysv_poll_group_0[42]), "=r"(_sysv_poll_group_0[43]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 5) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[44]), "=r"(_sysv_poll_group_0[45]), "=r"(_sysv_poll_group_0[46]), "=r"(_sysv_poll_group_0[47]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 5)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[48]), "=r"(_sysv_poll_group_0[49]), "=r"(_sysv_poll_group_0[50]), "=r"(_sysv_poll_group_0[51]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 6) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[52]), "=r"(_sysv_poll_group_0[53]), "=r"(_sysv_poll_group_0[54]), "=r"(_sysv_poll_group_0[55]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 6)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[56]), "=r"(_sysv_poll_group_0[57]), "=r"(_sysv_poll_group_0[58]), "=r"(_sysv_poll_group_0[59]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 7) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[60]), "=r"(_sysv_poll_group_0[61]), "=r"(_sysv_poll_group_0[62]), "=r"(_sysv_poll_group_0[63]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 7)) * 4096))) : "memory");
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
#else
            if (_sysv_pending_39 & 1u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + xf * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 2u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + xf) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 4u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 1) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 8u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 1)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 16u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 2) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 32u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 2)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 64u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 3) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 128u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 3)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 256u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[32]), "=r"(_sysv_poll_group_0[33]), "=r"(_sysv_poll_group_0[34]), "=r"(_sysv_poll_group_0[35]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 4) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 512u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[36]), "=r"(_sysv_poll_group_0[37]), "=r"(_sysv_poll_group_0[38]), "=r"(_sysv_poll_group_0[39]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 4)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 1024u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[40]), "=r"(_sysv_poll_group_0[41]), "=r"(_sysv_poll_group_0[42]), "=r"(_sysv_poll_group_0[43]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 5) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 2048u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[44]), "=r"(_sysv_poll_group_0[45]), "=r"(_sysv_poll_group_0[46]), "=r"(_sysv_poll_group_0[47]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 5)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 4096u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[48]), "=r"(_sysv_poll_group_0[49]), "=r"(_sysv_poll_group_0[50]), "=r"(_sysv_poll_group_0[51]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 6) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 8192u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[52]), "=r"(_sysv_poll_group_0[53]), "=r"(_sysv_poll_group_0[54]), "=r"(_sysv_poll_group_0[55]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 6)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 16384u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[56]), "=r"(_sysv_poll_group_0[57]), "=r"(_sysv_poll_group_0[58]), "=r"(_sysv_poll_group_0[59]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 7) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 32768u) {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[60]), "=r"(_sysv_poll_group_0[61]), "=r"(_sysv_poll_group_0[62]), "=r"(_sysv_poll_group_0[63]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 7)) * 4096))) : "memory");
            }
            if (_sysv_pending_39 & 1u) {
                if (!((((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~1u;
            }
            if (_sysv_pending_39 & 2u) {
                if (!((((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~2u;
            }
            if (_sysv_pending_39 & 4u) {
                if (!((((_sysv_poll_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~4u;
            }
            if (_sysv_pending_39 & 8u) {
                if (!((((_sysv_poll_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~8u;
            }
            if (_sysv_pending_39 & 16u) {
                if (!((((_sysv_poll_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~16u;
            }
            if (_sysv_pending_39 & 32u) {
                if (!((((_sysv_poll_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~32u;
            }
            if (_sysv_pending_39 & 64u) {
                if (!((((_sysv_poll_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~64u;
            }
            if (_sysv_pending_39 & 128u) {
                if (!((((_sysv_poll_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~128u;
            }
            if (_sysv_pending_39 & 256u) {
                if (!((((_sysv_poll_group_0[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[33] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[35] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~256u;
            }
            if (_sysv_pending_39 & 512u) {
                if (!((((_sysv_poll_group_0[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[37] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[39] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~512u;
            }
            if (_sysv_pending_39 & 1024u) {
                if (!((((_sysv_poll_group_0[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[41] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[43] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~1024u;
            }
            if (_sysv_pending_39 & 2048u) {
                if (!((((_sysv_poll_group_0[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[45] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[47] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~2048u;
            }
            if (_sysv_pending_39 & 4096u) {
                if (!((((_sysv_poll_group_0[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[49] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[51] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~4096u;
            }
            if (_sysv_pending_39 & 8192u) {
                if (!((((_sysv_poll_group_0[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[53] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[55] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~8192u;
            }
            if (_sysv_pending_39 & 16384u) {
                if (!((((_sysv_poll_group_0[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[57] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[59] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~16384u;
            }
            if (_sysv_pending_39 & 32768u) {
                if (!((((_sysv_poll_group_0[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[61] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[63] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_39 &= ~32768u;
            }
        } while (_sysv_pending_39 != 0u);
#endif
        float unpack_lo_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_0[_pair * 2])[0]), "=f"((&unpack_lo_0[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[_pair]));
        }
        float unpack_hi_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_0[_pair * 2])[0]), "=f"((&unpack_hi_0[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[4 + _pair]));
        }
        float s8_0[8];
        float t_lo_0 = (float)unpack_lo_0[0];
        s8_0[0] = t_lo_0;
        float t_hi_0 = (float)unpack_hi_0[0];
        s8_0[0] = s8_0[0] + t_hi_0;
        float t_lo_1 = (float)unpack_lo_0[1];
        s8_0[1] = t_lo_1;
        float t_hi_1 = (float)unpack_hi_0[1];
        s8_0[1] = s8_0[1] + t_hi_1;
        float t_lo_2 = (float)unpack_lo_0[2];
        s8_0[2] = t_lo_2;
        float t_hi_2 = (float)unpack_hi_0[2];
        s8_0[2] = s8_0[2] + t_hi_2;
        float t_lo_3 = (float)unpack_lo_0[3];
        s8_0[3] = t_lo_3;
        float t_hi_3 = (float)unpack_hi_0[3];
        s8_0[3] = s8_0[3] + t_hi_3;
        float t_lo_4 = (float)unpack_lo_0[4];
        s8_0[4] = t_lo_4;
        float t_hi_4 = (float)unpack_hi_0[4];
        s8_0[4] = s8_0[4] + t_hi_4;
        float t_lo_5 = (float)unpack_lo_0[5];
        s8_0[5] = t_lo_5;
        float t_hi_5 = (float)unpack_hi_0[5];
        s8_0[5] = s8_0[5] + t_hi_5;
        float t_lo_6 = (float)unpack_lo_0[6];
        s8_0[6] = t_lo_6;
        float t_hi_6 = (float)unpack_hi_0[6];
        s8_0[6] = s8_0[6] + t_hi_6;
        float t_lo_7 = (float)unpack_lo_0[7];
        s8_0[7] = t_lo_7;
        float t_hi_7 = (float)unpack_hi_0[7];
        s8_0[7] = s8_0[7] + t_hi_7;
        unsigned int s8_bf16_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_0[_lp*2 + 0], s8_0[_lp*2+1 + 0]));
            s8_bf16_0[_lp] = *(uint32_t*)&_bf2;
        }
        rw[0] = s8_bf16_0[0];
        rw[1] = s8_bf16_0[1];
        rw[2] = s8_bf16_0[2];
        rw[3] = s8_bf16_0[3];
        float unpack_s_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_0[_pair * 2])[0]), "=f"((&unpack_s_0[_pair * 2])[1])
                : "r"(s8_bf16_0[_pair]));
        }
        float ssq_e_0 = 0.0f;
        float y_e_0 = ((mode != 0) ? unpack_s_0[0] * gw8[0] : unpack_s_0[0]);
        ssq_e_0 = __fmaf_rn(y_e_0, y_e_0, ssq_e_0);
        float y_e_1 = ((mode != 0) ? unpack_s_0[1] * gw8[1] : unpack_s_0[1]);
        ssq_e_0 = __fmaf_rn(y_e_1, y_e_1, ssq_e_0);
        float y_e_2 = ((mode != 0) ? unpack_s_0[2] * gw8[2] : unpack_s_0[2]);
        ssq_e_0 = __fmaf_rn(y_e_2, y_e_2, ssq_e_0);
        float y_e_3 = ((mode != 0) ? unpack_s_0[3] * gw8[3] : unpack_s_0[3]);
        ssq_e_0 = __fmaf_rn(y_e_3, y_e_3, ssq_e_0);
        float y_e_4 = ((mode != 0) ? unpack_s_0[4] * gw8[4] : unpack_s_0[4]);
        ssq_e_0 = __fmaf_rn(y_e_4, y_e_4, ssq_e_0);
        float y_e_5 = ((mode != 0) ? unpack_s_0[5] * gw8[5] : unpack_s_0[5]);
        ssq_e_0 = __fmaf_rn(y_e_5, y_e_5, ssq_e_0);
        float y_e_6 = ((mode != 0) ? unpack_s_0[6] * gw8[6] : unpack_s_0[6]);
        ssq_e_0 = __fmaf_rn(y_e_6, y_e_6, ssq_e_0);
        float y_e_7 = ((mode != 0) ? unpack_s_0[7] * gw8[7] : unpack_s_0[7]);
        ssq_e_0 = __fmaf_rn(y_e_7, y_e_7, ssq_e_0);
        ssq_k[0] = ssq_e_0;
        float unpack_lo_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_1[_pair * 2])[0]), "=f"((&unpack_lo_1[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[8 + _pair]));
        }
        float unpack_hi_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_1[_pair * 2])[0]), "=f"((&unpack_hi_1[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[12 + _pair]));
        }
        float s8_1[8];
        float t_lo_8 = (float)unpack_lo_1[0];
        s8_1[0] = t_lo_8;
        float t_hi_8 = (float)unpack_hi_1[0];
        s8_1[0] = s8_1[0] + t_hi_8;
        float t_lo_9 = (float)unpack_lo_1[1];
        s8_1[1] = t_lo_9;
        float t_hi_9 = (float)unpack_hi_1[1];
        s8_1[1] = s8_1[1] + t_hi_9;
        float t_lo_10 = (float)unpack_lo_1[2];
        s8_1[2] = t_lo_10;
        float t_hi_10 = (float)unpack_hi_1[2];
        s8_1[2] = s8_1[2] + t_hi_10;
        float t_lo_11 = (float)unpack_lo_1[3];
        s8_1[3] = t_lo_11;
        float t_hi_11 = (float)unpack_hi_1[3];
        s8_1[3] = s8_1[3] + t_hi_11;
        float t_lo_12 = (float)unpack_lo_1[4];
        s8_1[4] = t_lo_12;
        float t_hi_12 = (float)unpack_hi_1[4];
        s8_1[4] = s8_1[4] + t_hi_12;
        float t_lo_13 = (float)unpack_lo_1[5];
        s8_1[5] = t_lo_13;
        float t_hi_13 = (float)unpack_hi_1[5];
        s8_1[5] = s8_1[5] + t_hi_13;
        float t_lo_14 = (float)unpack_lo_1[6];
        s8_1[6] = t_lo_14;
        float t_hi_14 = (float)unpack_hi_1[6];
        s8_1[6] = s8_1[6] + t_hi_14;
        float t_lo_15 = (float)unpack_lo_1[7];
        s8_1[7] = t_lo_15;
        float t_hi_15 = (float)unpack_hi_1[7];
        s8_1[7] = s8_1[7] + t_hi_15;
        unsigned int s8_bf16_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_1[_lp*2 + 0], s8_1[_lp*2+1 + 0]));
            s8_bf16_1[_lp] = *(uint32_t*)&_bf2;
        }
        rw[4] = s8_bf16_1[0];
        rw[5] = s8_bf16_1[1];
        rw[6] = s8_bf16_1[2];
        rw[7] = s8_bf16_1[3];
        float unpack_s_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_1[_pair * 2])[0]), "=f"((&unpack_s_1[_pair * 2])[1])
                : "r"(s8_bf16_1[_pair]));
        }
        float ssq_e_1 = 0.0f;
        float y_e_8 = ((mode != 0) ? unpack_s_1[0] * gw8[0] : unpack_s_1[0]);
        ssq_e_1 = __fmaf_rn(y_e_8, y_e_8, ssq_e_1);
        float y_e_9 = ((mode != 0) ? unpack_s_1[1] * gw8[1] : unpack_s_1[1]);
        ssq_e_1 = __fmaf_rn(y_e_9, y_e_9, ssq_e_1);
        float y_e_10 = ((mode != 0) ? unpack_s_1[2] * gw8[2] : unpack_s_1[2]);
        ssq_e_1 = __fmaf_rn(y_e_10, y_e_10, ssq_e_1);
        float y_e_11 = ((mode != 0) ? unpack_s_1[3] * gw8[3] : unpack_s_1[3]);
        ssq_e_1 = __fmaf_rn(y_e_11, y_e_11, ssq_e_1);
        float y_e_12 = ((mode != 0) ? unpack_s_1[4] * gw8[4] : unpack_s_1[4]);
        ssq_e_1 = __fmaf_rn(y_e_12, y_e_12, ssq_e_1);
        float y_e_13 = ((mode != 0) ? unpack_s_1[5] * gw8[5] : unpack_s_1[5]);
        ssq_e_1 = __fmaf_rn(y_e_13, y_e_13, ssq_e_1);
        float y_e_14 = ((mode != 0) ? unpack_s_1[6] * gw8[6] : unpack_s_1[6]);
        ssq_e_1 = __fmaf_rn(y_e_14, y_e_14, ssq_e_1);
        float y_e_15 = ((mode != 0) ? unpack_s_1[7] * gw8[7] : unpack_s_1[7]);
        ssq_e_1 = __fmaf_rn(y_e_15, y_e_15, ssq_e_1);
        ssq_k[1] = ssq_e_1;
        float unpack_lo_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_2[_pair * 2])[0]), "=f"((&unpack_lo_2[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[16 + _pair]));
        }
        float unpack_hi_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_2[_pair * 2])[0]), "=f"((&unpack_hi_2[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[20 + _pair]));
        }
        float s8_2[8];
        float t_lo_16 = (float)unpack_lo_2[0];
        s8_2[0] = t_lo_16;
        float t_hi_16 = (float)unpack_hi_2[0];
        s8_2[0] = s8_2[0] + t_hi_16;
        float t_lo_17 = (float)unpack_lo_2[1];
        s8_2[1] = t_lo_17;
        float t_hi_17 = (float)unpack_hi_2[1];
        s8_2[1] = s8_2[1] + t_hi_17;
        float t_lo_18 = (float)unpack_lo_2[2];
        s8_2[2] = t_lo_18;
        float t_hi_18 = (float)unpack_hi_2[2];
        s8_2[2] = s8_2[2] + t_hi_18;
        float t_lo_19 = (float)unpack_lo_2[3];
        s8_2[3] = t_lo_19;
        float t_hi_19 = (float)unpack_hi_2[3];
        s8_2[3] = s8_2[3] + t_hi_19;
        float t_lo_20 = (float)unpack_lo_2[4];
        s8_2[4] = t_lo_20;
        float t_hi_20 = (float)unpack_hi_2[4];
        s8_2[4] = s8_2[4] + t_hi_20;
        float t_lo_21 = (float)unpack_lo_2[5];
        s8_2[5] = t_lo_21;
        float t_hi_21 = (float)unpack_hi_2[5];
        s8_2[5] = s8_2[5] + t_hi_21;
        float t_lo_22 = (float)unpack_lo_2[6];
        s8_2[6] = t_lo_22;
        float t_hi_22 = (float)unpack_hi_2[6];
        s8_2[6] = s8_2[6] + t_hi_22;
        float t_lo_23 = (float)unpack_lo_2[7];
        s8_2[7] = t_lo_23;
        float t_hi_23 = (float)unpack_hi_2[7];
        s8_2[7] = s8_2[7] + t_hi_23;
        unsigned int s8_bf16_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_2[_lp*2 + 0], s8_2[_lp*2+1 + 0]));
            s8_bf16_2[_lp] = *(uint32_t*)&_bf2;
        }
        rw[8] = s8_bf16_2[0];
        rw[9] = s8_bf16_2[1];
        rw[10] = s8_bf16_2[2];
        rw[11] = s8_bf16_2[3];
        float unpack_s_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_2[_pair * 2])[0]), "=f"((&unpack_s_2[_pair * 2])[1])
                : "r"(s8_bf16_2[_pair]));
        }
        float ssq_e_2 = 0.0f;
        float y_e_16 = ((mode != 0) ? unpack_s_2[0] * gw8[0] : unpack_s_2[0]);
        ssq_e_2 = __fmaf_rn(y_e_16, y_e_16, ssq_e_2);
        float y_e_17 = ((mode != 0) ? unpack_s_2[1] * gw8[1] : unpack_s_2[1]);
        ssq_e_2 = __fmaf_rn(y_e_17, y_e_17, ssq_e_2);
        float y_e_18 = ((mode != 0) ? unpack_s_2[2] * gw8[2] : unpack_s_2[2]);
        ssq_e_2 = __fmaf_rn(y_e_18, y_e_18, ssq_e_2);
        float y_e_19 = ((mode != 0) ? unpack_s_2[3] * gw8[3] : unpack_s_2[3]);
        ssq_e_2 = __fmaf_rn(y_e_19, y_e_19, ssq_e_2);
        float y_e_20 = ((mode != 0) ? unpack_s_2[4] * gw8[4] : unpack_s_2[4]);
        ssq_e_2 = __fmaf_rn(y_e_20, y_e_20, ssq_e_2);
        float y_e_21 = ((mode != 0) ? unpack_s_2[5] * gw8[5] : unpack_s_2[5]);
        ssq_e_2 = __fmaf_rn(y_e_21, y_e_21, ssq_e_2);
        float y_e_22 = ((mode != 0) ? unpack_s_2[6] * gw8[6] : unpack_s_2[6]);
        ssq_e_2 = __fmaf_rn(y_e_22, y_e_22, ssq_e_2);
        float y_e_23 = ((mode != 0) ? unpack_s_2[7] * gw8[7] : unpack_s_2[7]);
        ssq_e_2 = __fmaf_rn(y_e_23, y_e_23, ssq_e_2);
        ssq_k[2] = ssq_e_2;
        float unpack_lo_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_3[_pair * 2])[0]), "=f"((&unpack_lo_3[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[24 + _pair]));
        }
        float unpack_hi_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_3[_pair * 2])[0]), "=f"((&unpack_hi_3[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[28 + _pair]));
        }
        float s8_3[8];
        float t_lo_24 = (float)unpack_lo_3[0];
        s8_3[0] = t_lo_24;
        float t_hi_24 = (float)unpack_hi_3[0];
        s8_3[0] = s8_3[0] + t_hi_24;
        float t_lo_25 = (float)unpack_lo_3[1];
        s8_3[1] = t_lo_25;
        float t_hi_25 = (float)unpack_hi_3[1];
        s8_3[1] = s8_3[1] + t_hi_25;
        float t_lo_26 = (float)unpack_lo_3[2];
        s8_3[2] = t_lo_26;
        float t_hi_26 = (float)unpack_hi_3[2];
        s8_3[2] = s8_3[2] + t_hi_26;
        float t_lo_27 = (float)unpack_lo_3[3];
        s8_3[3] = t_lo_27;
        float t_hi_27 = (float)unpack_hi_3[3];
        s8_3[3] = s8_3[3] + t_hi_27;
        float t_lo_28 = (float)unpack_lo_3[4];
        s8_3[4] = t_lo_28;
        float t_hi_28 = (float)unpack_hi_3[4];
        s8_3[4] = s8_3[4] + t_hi_28;
        float t_lo_29 = (float)unpack_lo_3[5];
        s8_3[5] = t_lo_29;
        float t_hi_29 = (float)unpack_hi_3[5];
        s8_3[5] = s8_3[5] + t_hi_29;
        float t_lo_30 = (float)unpack_lo_3[6];
        s8_3[6] = t_lo_30;
        float t_hi_30 = (float)unpack_hi_3[6];
        s8_3[6] = s8_3[6] + t_hi_30;
        float t_lo_31 = (float)unpack_lo_3[7];
        s8_3[7] = t_lo_31;
        float t_hi_31 = (float)unpack_hi_3[7];
        s8_3[7] = s8_3[7] + t_hi_31;
        unsigned int s8_bf16_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_3[_lp*2 + 0], s8_3[_lp*2+1 + 0]));
            s8_bf16_3[_lp] = *(uint32_t*)&_bf2;
        }
        rw[12] = s8_bf16_3[0];
        rw[13] = s8_bf16_3[1];
        rw[14] = s8_bf16_3[2];
        rw[15] = s8_bf16_3[3];
        float unpack_s_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_3[_pair * 2])[0]), "=f"((&unpack_s_3[_pair * 2])[1])
                : "r"(s8_bf16_3[_pair]));
        }
        float ssq_e_3 = 0.0f;
        float y_e_24 = ((mode != 0) ? unpack_s_3[0] * gw8[0] : unpack_s_3[0]);
        ssq_e_3 = __fmaf_rn(y_e_24, y_e_24, ssq_e_3);
        float y_e_25 = ((mode != 0) ? unpack_s_3[1] * gw8[1] : unpack_s_3[1]);
        ssq_e_3 = __fmaf_rn(y_e_25, y_e_25, ssq_e_3);
        float y_e_26 = ((mode != 0) ? unpack_s_3[2] * gw8[2] : unpack_s_3[2]);
        ssq_e_3 = __fmaf_rn(y_e_26, y_e_26, ssq_e_3);
        float y_e_27 = ((mode != 0) ? unpack_s_3[3] * gw8[3] : unpack_s_3[3]);
        ssq_e_3 = __fmaf_rn(y_e_27, y_e_27, ssq_e_3);
        float y_e_28 = ((mode != 0) ? unpack_s_3[4] * gw8[4] : unpack_s_3[4]);
        ssq_e_3 = __fmaf_rn(y_e_28, y_e_28, ssq_e_3);
        float y_e_29 = ((mode != 0) ? unpack_s_3[5] * gw8[5] : unpack_s_3[5]);
        ssq_e_3 = __fmaf_rn(y_e_29, y_e_29, ssq_e_3);
        float y_e_30 = ((mode != 0) ? unpack_s_3[6] * gw8[6] : unpack_s_3[6]);
        ssq_e_3 = __fmaf_rn(y_e_30, y_e_30, ssq_e_3);
        float y_e_31 = ((mode != 0) ? unpack_s_3[7] * gw8[7] : unpack_s_3[7]);
        ssq_e_3 = __fmaf_rn(y_e_31, y_e_31, ssq_e_3);
        ssq_k[3] = ssq_e_3;
        float unpack_lo_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_4[_pair * 2])[0]), "=f"((&unpack_lo_4[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[32 + _pair]));
        }
        float unpack_hi_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_4[_pair * 2])[0]), "=f"((&unpack_hi_4[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[36 + _pair]));
        }
        float s8_4[8];
        float t_lo_32 = (float)unpack_lo_4[0];
        s8_4[0] = t_lo_32;
        float t_hi_32 = (float)unpack_hi_4[0];
        s8_4[0] = s8_4[0] + t_hi_32;
        float t_lo_33 = (float)unpack_lo_4[1];
        s8_4[1] = t_lo_33;
        float t_hi_33 = (float)unpack_hi_4[1];
        s8_4[1] = s8_4[1] + t_hi_33;
        float t_lo_34 = (float)unpack_lo_4[2];
        s8_4[2] = t_lo_34;
        float t_hi_34 = (float)unpack_hi_4[2];
        s8_4[2] = s8_4[2] + t_hi_34;
        float t_lo_35 = (float)unpack_lo_4[3];
        s8_4[3] = t_lo_35;
        float t_hi_35 = (float)unpack_hi_4[3];
        s8_4[3] = s8_4[3] + t_hi_35;
        float t_lo_36 = (float)unpack_lo_4[4];
        s8_4[4] = t_lo_36;
        float t_hi_36 = (float)unpack_hi_4[4];
        s8_4[4] = s8_4[4] + t_hi_36;
        float t_lo_37 = (float)unpack_lo_4[5];
        s8_4[5] = t_lo_37;
        float t_hi_37 = (float)unpack_hi_4[5];
        s8_4[5] = s8_4[5] + t_hi_37;
        float t_lo_38 = (float)unpack_lo_4[6];
        s8_4[6] = t_lo_38;
        float t_hi_38 = (float)unpack_hi_4[6];
        s8_4[6] = s8_4[6] + t_hi_38;
        float t_lo_39 = (float)unpack_lo_4[7];
        s8_4[7] = t_lo_39;
        float t_hi_39 = (float)unpack_hi_4[7];
        s8_4[7] = s8_4[7] + t_hi_39;
        unsigned int s8_bf16_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_4[_lp*2 + 0], s8_4[_lp*2+1 + 0]));
            s8_bf16_4[_lp] = *(uint32_t*)&_bf2;
        }
        rw[16] = s8_bf16_4[0];
        rw[17] = s8_bf16_4[1];
        rw[18] = s8_bf16_4[2];
        rw[19] = s8_bf16_4[3];
        float unpack_s_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_4[_pair * 2])[0]), "=f"((&unpack_s_4[_pair * 2])[1])
                : "r"(s8_bf16_4[_pair]));
        }
        float ssq_e_4 = 0.0f;
        float y_e_32 = ((mode != 0) ? unpack_s_4[0] * gw8[0] : unpack_s_4[0]);
        ssq_e_4 = __fmaf_rn(y_e_32, y_e_32, ssq_e_4);
        float y_e_33 = ((mode != 0) ? unpack_s_4[1] * gw8[1] : unpack_s_4[1]);
        ssq_e_4 = __fmaf_rn(y_e_33, y_e_33, ssq_e_4);
        float y_e_34 = ((mode != 0) ? unpack_s_4[2] * gw8[2] : unpack_s_4[2]);
        ssq_e_4 = __fmaf_rn(y_e_34, y_e_34, ssq_e_4);
        float y_e_35 = ((mode != 0) ? unpack_s_4[3] * gw8[3] : unpack_s_4[3]);
        ssq_e_4 = __fmaf_rn(y_e_35, y_e_35, ssq_e_4);
        float y_e_36 = ((mode != 0) ? unpack_s_4[4] * gw8[4] : unpack_s_4[4]);
        ssq_e_4 = __fmaf_rn(y_e_36, y_e_36, ssq_e_4);
        float y_e_37 = ((mode != 0) ? unpack_s_4[5] * gw8[5] : unpack_s_4[5]);
        ssq_e_4 = __fmaf_rn(y_e_37, y_e_37, ssq_e_4);
        float y_e_38 = ((mode != 0) ? unpack_s_4[6] * gw8[6] : unpack_s_4[6]);
        ssq_e_4 = __fmaf_rn(y_e_38, y_e_38, ssq_e_4);
        float y_e_39 = ((mode != 0) ? unpack_s_4[7] * gw8[7] : unpack_s_4[7]);
        ssq_e_4 = __fmaf_rn(y_e_39, y_e_39, ssq_e_4);
        ssq_k[4] = ssq_e_4;
        float unpack_lo_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_5[_pair * 2])[0]), "=f"((&unpack_lo_5[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[40 + _pair]));
        }
        float unpack_hi_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_5[_pair * 2])[0]), "=f"((&unpack_hi_5[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[44 + _pair]));
        }
        float s8_5[8];
        float t_lo_40 = (float)unpack_lo_5[0];
        s8_5[0] = t_lo_40;
        float t_hi_40 = (float)unpack_hi_5[0];
        s8_5[0] = s8_5[0] + t_hi_40;
        float t_lo_41 = (float)unpack_lo_5[1];
        s8_5[1] = t_lo_41;
        float t_hi_41 = (float)unpack_hi_5[1];
        s8_5[1] = s8_5[1] + t_hi_41;
        float t_lo_42 = (float)unpack_lo_5[2];
        s8_5[2] = t_lo_42;
        float t_hi_42 = (float)unpack_hi_5[2];
        s8_5[2] = s8_5[2] + t_hi_42;
        float t_lo_43 = (float)unpack_lo_5[3];
        s8_5[3] = t_lo_43;
        float t_hi_43 = (float)unpack_hi_5[3];
        s8_5[3] = s8_5[3] + t_hi_43;
        float t_lo_44 = (float)unpack_lo_5[4];
        s8_5[4] = t_lo_44;
        float t_hi_44 = (float)unpack_hi_5[4];
        s8_5[4] = s8_5[4] + t_hi_44;
        float t_lo_45 = (float)unpack_lo_5[5];
        s8_5[5] = t_lo_45;
        float t_hi_45 = (float)unpack_hi_5[5];
        s8_5[5] = s8_5[5] + t_hi_45;
        float t_lo_46 = (float)unpack_lo_5[6];
        s8_5[6] = t_lo_46;
        float t_hi_46 = (float)unpack_hi_5[6];
        s8_5[6] = s8_5[6] + t_hi_46;
        float t_lo_47 = (float)unpack_lo_5[7];
        s8_5[7] = t_lo_47;
        float t_hi_47 = (float)unpack_hi_5[7];
        s8_5[7] = s8_5[7] + t_hi_47;
        unsigned int s8_bf16_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_5[_lp*2 + 0], s8_5[_lp*2+1 + 0]));
            s8_bf16_5[_lp] = *(uint32_t*)&_bf2;
        }
        rw[20] = s8_bf16_5[0];
        rw[21] = s8_bf16_5[1];
        rw[22] = s8_bf16_5[2];
        rw[23] = s8_bf16_5[3];
        float unpack_s_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_5[_pair * 2])[0]), "=f"((&unpack_s_5[_pair * 2])[1])
                : "r"(s8_bf16_5[_pair]));
        }
        float ssq_e_5 = 0.0f;
        float y_e_40 = ((mode != 0) ? unpack_s_5[0] * gw8[0] : unpack_s_5[0]);
        ssq_e_5 = __fmaf_rn(y_e_40, y_e_40, ssq_e_5);
        float y_e_41 = ((mode != 0) ? unpack_s_5[1] * gw8[1] : unpack_s_5[1]);
        ssq_e_5 = __fmaf_rn(y_e_41, y_e_41, ssq_e_5);
        float y_e_42 = ((mode != 0) ? unpack_s_5[2] * gw8[2] : unpack_s_5[2]);
        ssq_e_5 = __fmaf_rn(y_e_42, y_e_42, ssq_e_5);
        float y_e_43 = ((mode != 0) ? unpack_s_5[3] * gw8[3] : unpack_s_5[3]);
        ssq_e_5 = __fmaf_rn(y_e_43, y_e_43, ssq_e_5);
        float y_e_44 = ((mode != 0) ? unpack_s_5[4] * gw8[4] : unpack_s_5[4]);
        ssq_e_5 = __fmaf_rn(y_e_44, y_e_44, ssq_e_5);
        float y_e_45 = ((mode != 0) ? unpack_s_5[5] * gw8[5] : unpack_s_5[5]);
        ssq_e_5 = __fmaf_rn(y_e_45, y_e_45, ssq_e_5);
        float y_e_46 = ((mode != 0) ? unpack_s_5[6] * gw8[6] : unpack_s_5[6]);
        ssq_e_5 = __fmaf_rn(y_e_46, y_e_46, ssq_e_5);
        float y_e_47 = ((mode != 0) ? unpack_s_5[7] * gw8[7] : unpack_s_5[7]);
        ssq_e_5 = __fmaf_rn(y_e_47, y_e_47, ssq_e_5);
        ssq_k[5] = ssq_e_5;
        float unpack_lo_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_6[_pair * 2])[0]), "=f"((&unpack_lo_6[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[48 + _pair]));
        }
        float unpack_hi_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_6[_pair * 2])[0]), "=f"((&unpack_hi_6[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[52 + _pair]));
        }
        float s8_6[8];
        float t_lo_48 = (float)unpack_lo_6[0];
        s8_6[0] = t_lo_48;
        float t_hi_48 = (float)unpack_hi_6[0];
        s8_6[0] = s8_6[0] + t_hi_48;
        float t_lo_49 = (float)unpack_lo_6[1];
        s8_6[1] = t_lo_49;
        float t_hi_49 = (float)unpack_hi_6[1];
        s8_6[1] = s8_6[1] + t_hi_49;
        float t_lo_50 = (float)unpack_lo_6[2];
        s8_6[2] = t_lo_50;
        float t_hi_50 = (float)unpack_hi_6[2];
        s8_6[2] = s8_6[2] + t_hi_50;
        float t_lo_51 = (float)unpack_lo_6[3];
        s8_6[3] = t_lo_51;
        float t_hi_51 = (float)unpack_hi_6[3];
        s8_6[3] = s8_6[3] + t_hi_51;
        float t_lo_52 = (float)unpack_lo_6[4];
        s8_6[4] = t_lo_52;
        float t_hi_52 = (float)unpack_hi_6[4];
        s8_6[4] = s8_6[4] + t_hi_52;
        float t_lo_53 = (float)unpack_lo_6[5];
        s8_6[5] = t_lo_53;
        float t_hi_53 = (float)unpack_hi_6[5];
        s8_6[5] = s8_6[5] + t_hi_53;
        float t_lo_54 = (float)unpack_lo_6[6];
        s8_6[6] = t_lo_54;
        float t_hi_54 = (float)unpack_hi_6[6];
        s8_6[6] = s8_6[6] + t_hi_54;
        float t_lo_55 = (float)unpack_lo_6[7];
        s8_6[7] = t_lo_55;
        float t_hi_55 = (float)unpack_hi_6[7];
        s8_6[7] = s8_6[7] + t_hi_55;
        unsigned int s8_bf16_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_6[_lp*2 + 0], s8_6[_lp*2+1 + 0]));
            s8_bf16_6[_lp] = *(uint32_t*)&_bf2;
        }
        rw[24] = s8_bf16_6[0];
        rw[25] = s8_bf16_6[1];
        rw[26] = s8_bf16_6[2];
        rw[27] = s8_bf16_6[3];
        float unpack_s_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_6[_pair * 2])[0]), "=f"((&unpack_s_6[_pair * 2])[1])
                : "r"(s8_bf16_6[_pair]));
        }
        float ssq_e_6 = 0.0f;
        float y_e_48 = ((mode != 0) ? unpack_s_6[0] * gw8[0] : unpack_s_6[0]);
        ssq_e_6 = __fmaf_rn(y_e_48, y_e_48, ssq_e_6);
        float y_e_49 = ((mode != 0) ? unpack_s_6[1] * gw8[1] : unpack_s_6[1]);
        ssq_e_6 = __fmaf_rn(y_e_49, y_e_49, ssq_e_6);
        float y_e_50 = ((mode != 0) ? unpack_s_6[2] * gw8[2] : unpack_s_6[2]);
        ssq_e_6 = __fmaf_rn(y_e_50, y_e_50, ssq_e_6);
        float y_e_51 = ((mode != 0) ? unpack_s_6[3] * gw8[3] : unpack_s_6[3]);
        ssq_e_6 = __fmaf_rn(y_e_51, y_e_51, ssq_e_6);
        float y_e_52 = ((mode != 0) ? unpack_s_6[4] * gw8[4] : unpack_s_6[4]);
        ssq_e_6 = __fmaf_rn(y_e_52, y_e_52, ssq_e_6);
        float y_e_53 = ((mode != 0) ? unpack_s_6[5] * gw8[5] : unpack_s_6[5]);
        ssq_e_6 = __fmaf_rn(y_e_53, y_e_53, ssq_e_6);
        float y_e_54 = ((mode != 0) ? unpack_s_6[6] * gw8[6] : unpack_s_6[6]);
        ssq_e_6 = __fmaf_rn(y_e_54, y_e_54, ssq_e_6);
        float y_e_55 = ((mode != 0) ? unpack_s_6[7] * gw8[7] : unpack_s_6[7]);
        ssq_e_6 = __fmaf_rn(y_e_55, y_e_55, ssq_e_6);
        ssq_k[6] = ssq_e_6;
        float unpack_lo_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_lo_7[_pair * 2])[0]), "=f"((&unpack_lo_7[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[56 + _pair]));
        }
        float unpack_hi_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_hi_7[_pair * 2])[0]), "=f"((&unpack_hi_7[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[60 + _pair]));
        }
        float s8_7[8];
        float t_lo_56 = (float)unpack_lo_7[0];
        s8_7[0] = t_lo_56;
        float t_hi_56 = (float)unpack_hi_7[0];
        s8_7[0] = s8_7[0] + t_hi_56;
        float t_lo_57 = (float)unpack_lo_7[1];
        s8_7[1] = t_lo_57;
        float t_hi_57 = (float)unpack_hi_7[1];
        s8_7[1] = s8_7[1] + t_hi_57;
        float t_lo_58 = (float)unpack_lo_7[2];
        s8_7[2] = t_lo_58;
        float t_hi_58 = (float)unpack_hi_7[2];
        s8_7[2] = s8_7[2] + t_hi_58;
        float t_lo_59 = (float)unpack_lo_7[3];
        s8_7[3] = t_lo_59;
        float t_hi_59 = (float)unpack_hi_7[3];
        s8_7[3] = s8_7[3] + t_hi_59;
        float t_lo_60 = (float)unpack_lo_7[4];
        s8_7[4] = t_lo_60;
        float t_hi_60 = (float)unpack_hi_7[4];
        s8_7[4] = s8_7[4] + t_hi_60;
        float t_lo_61 = (float)unpack_lo_7[5];
        s8_7[5] = t_lo_61;
        float t_hi_61 = (float)unpack_hi_7[5];
        s8_7[5] = s8_7[5] + t_hi_61;
        float t_lo_62 = (float)unpack_lo_7[6];
        s8_7[6] = t_lo_62;
        float t_hi_62 = (float)unpack_hi_7[6];
        s8_7[6] = s8_7[6] + t_hi_62;
        float t_lo_63 = (float)unpack_lo_7[7];
        s8_7[7] = t_lo_63;
        float t_hi_63 = (float)unpack_hi_7[7];
        s8_7[7] = s8_7[7] + t_hi_63;
        unsigned int s8_bf16_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_7[_lp*2 + 0], s8_7[_lp*2+1 + 0]));
            s8_bf16_7[_lp] = *(uint32_t*)&_bf2;
        }
        rw[28] = s8_bf16_7[0];
        rw[29] = s8_bf16_7[1];
        rw[30] = s8_bf16_7[2];
        rw[31] = s8_bf16_7[3];
        float unpack_s_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_s_7[_pair * 2])[0]), "=f"((&unpack_s_7[_pair * 2])[1])
                : "r"(s8_bf16_7[_pair]));
        }
        float ssq_e_7 = 0.0f;
        float y_e_56 = ((mode != 0) ? unpack_s_7[0] * gw8[0] : unpack_s_7[0]);
        ssq_e_7 = __fmaf_rn(y_e_56, y_e_56, ssq_e_7);
        float y_e_57 = ((mode != 0) ? unpack_s_7[1] * gw8[1] : unpack_s_7[1]);
        ssq_e_7 = __fmaf_rn(y_e_57, y_e_57, ssq_e_7);
        float y_e_58 = ((mode != 0) ? unpack_s_7[2] * gw8[2] : unpack_s_7[2]);
        ssq_e_7 = __fmaf_rn(y_e_58, y_e_58, ssq_e_7);
        float y_e_59 = ((mode != 0) ? unpack_s_7[3] * gw8[3] : unpack_s_7[3]);
        ssq_e_7 = __fmaf_rn(y_e_59, y_e_59, ssq_e_7);
        float y_e_60 = ((mode != 0) ? unpack_s_7[4] * gw8[4] : unpack_s_7[4]);
        ssq_e_7 = __fmaf_rn(y_e_60, y_e_60, ssq_e_7);
        float y_e_61 = ((mode != 0) ? unpack_s_7[5] * gw8[5] : unpack_s_7[5]);
        ssq_e_7 = __fmaf_rn(y_e_61, y_e_61, ssq_e_7);
        float y_e_62 = ((mode != 0) ? unpack_s_7[6] * gw8[6] : unpack_s_7[6]);
        ssq_e_7 = __fmaf_rn(y_e_62, y_e_62, ssq_e_7);
        float y_e_63 = ((mode != 0) ? unpack_s_7[7] * gw8[7] : unpack_s_7[7]);
        ssq_e_7 = __fmaf_rn(y_e_63, y_e_63, ssq_e_7);
        ssq_k[7] = ssq_e_7;
        float warp_reduce_r_0 = ssq_k[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_0 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_0, offset);
        if (lane == 0) {
            warp_tab[warp * 8] = warp_reduce_r_0;
        }
        float warp_reduce_r_1 = ssq_k[1];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_1 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_1, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 1] = warp_reduce_r_1;
        }
        float warp_reduce_r_2 = ssq_k[2];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_2 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_2, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 2] = warp_reduce_r_2;
        }
        float warp_reduce_r_3 = ssq_k[3];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_3 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_3, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 3] = warp_reduce_r_3;
        }
        float warp_reduce_r_4 = ssq_k[4];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_4 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_4, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 4] = warp_reduce_r_4;
        }
        float warp_reduce_r_5 = ssq_k[5];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_5 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_5, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 5] = warp_reduce_r_5;
        }
        float warp_reduce_r_6 = ssq_k[6];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_6 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_6, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 6] = warp_reduce_r_6;
        }
        float warp_reduce_r_7 = ssq_k[7];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_7 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_7, offset);
        if (lane == 0) {
            warp_tab[warp * 8 + 7] = warp_reduce_r_7;
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
#if __CUDA_ARCH__ == 1030
            uint32_t _cluster_mapa_39;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_39) : "r"(slot_addr), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_39), "f"(part_t) : "memory");
#else
#endif
            uint32_t _cluster_mapa_40;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cluster_mapa_40) : "r"(slot_addr), "r"(1));
#else
                : "=r"(_cluster_mapa_40) : "r"(slot_addr), "r"(0));
#endif
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_40), "f"(part_t) : "memory");
            uint32_t _cluster_mapa_41;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cluster_mapa_41) : "r"(slot_addr), "r"(2));
#else
                : "=r"(_cluster_mapa_41) : "r"(slot_addr), "r"(1));
#endif
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_41), "f"(part_t) : "memory");
            uint32_t _cluster_mapa_42;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cluster_mapa_42) : "r"(slot_addr), "r"(3));
#else
                : "=r"(_cluster_mapa_42) : "r"(slot_addr), "r"(2));
#endif
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_42), "f"(part_t) : "memory");
#if __CUDA_ARCH__ == 1030
#else
            uint32_t _cluster_mapa_43;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_43) : "r"(slot_addr), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_43), "f"(part_t) : "memory");
#endif
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
#if __CUDA_ARCH__ == 1030
        float _sqrt_43;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_43) : "f"(__fmaf_rn(tot_k[0], 0.00024414062f, eps)));
        float rs_e_0 = __frcp_rn(_sqrt_43);
#else
        float _sqrt_44;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_44) : "f"(__fmaf_rn(tot_k[0], 0.00024414062f, eps)));
        float rs_e_0 = __frcp_rn(_sqrt_44);
#endif
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float unpack_o_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_0[_pair * 2])[0]), "=f"((&unpack_o_0[_pair * 2])[1])
                : "r"(rw[_pair]));
        }
        float yn_e_0 = ((mode != 0) ? unpack_o_0[0] * gw8[0] : unpack_o_0[0]);
        float o_e_0 = yn_e_0 * rs_e_0;
        float o2_e_0 = ((mode != 0) ? o_e_0 : o_e_0 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_0, wk[0], pp[0]);
        float yn_e_1 = ((mode != 0) ? unpack_o_0[1] * gw8[1] : unpack_o_0[1]);
        float o_e_1 = yn_e_1 * rs_e_0;
        float o2_e_1 = ((mode != 0) ? o_e_1 : o_e_1 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_1, wk[0], pp[1]);
        float yn_e_2 = ((mode != 0) ? unpack_o_0[2] * gw8[2] : unpack_o_0[2]);
        float o_e_2 = yn_e_2 * rs_e_0;
        float o2_e_2 = ((mode != 0) ? o_e_2 : o_e_2 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_2, wk[0], pp[2]);
        float yn_e_3 = ((mode != 0) ? unpack_o_0[3] * gw8[3] : unpack_o_0[3]);
        float o_e_3 = yn_e_3 * rs_e_0;
        float o2_e_3 = ((mode != 0) ? o_e_3 : o_e_3 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_3, wk[0], pp[3]);
        float yn_e_4 = ((mode != 0) ? unpack_o_0[4] * gw8[4] : unpack_o_0[4]);
        float o_e_4 = yn_e_4 * rs_e_0;
        float o2_e_4 = ((mode != 0) ? o_e_4 : o_e_4 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_4, wk[0], pp[4]);
        float yn_e_5 = ((mode != 0) ? unpack_o_0[5] * gw8[5] : unpack_o_0[5]);
        float o_e_5 = yn_e_5 * rs_e_0;
        float o2_e_5 = ((mode != 0) ? o_e_5 : o_e_5 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_5, wk[0], pp[5]);
        float yn_e_6 = ((mode != 0) ? unpack_o_0[6] * gw8[6] : unpack_o_0[6]);
        float o_e_6 = yn_e_6 * rs_e_0;
        float o2_e_6 = ((mode != 0) ? o_e_6 : o_e_6 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_6, wk[0], pp[6]);
        float yn_e_7 = ((mode != 0) ? unpack_o_0[7] * gw8[7] : unpack_o_0[7]);
        float o_e_7 = yn_e_7 * rs_e_0;
        float o2_e_7 = ((mode != 0) ? o_e_7 : o_e_7 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_7, wk[0], pp[7]);
#if __CUDA_ARCH__ == 1030
        float _sqrt_44;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_44) : "f"(__fmaf_rn(tot_k[1], 0.00024414062f, eps)));
        float rs_e_1 = __frcp_rn(_sqrt_44);
#else
        float _sqrt_45;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_45) : "f"(__fmaf_rn(tot_k[1], 0.00024414062f, eps)));
        float rs_e_1 = __frcp_rn(_sqrt_45);
#endif
        float unpack_o_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_1[_pair * 2])[0]), "=f"((&unpack_o_1[_pair * 2])[1])
                : "r"(rw[4 + _pair]));
        }
        float yn_e_8 = ((mode != 0) ? unpack_o_1[0] * gw8[0] : unpack_o_1[0]);
        float o_e_8 = yn_e_8 * rs_e_1;
        float o2_e_8 = ((mode != 0) ? o_e_8 : o_e_8 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_8, wk[1], pp[0]);
        float yn_e_9 = ((mode != 0) ? unpack_o_1[1] * gw8[1] : unpack_o_1[1]);
        float o_e_9 = yn_e_9 * rs_e_1;
        float o2_e_9 = ((mode != 0) ? o_e_9 : o_e_9 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_9, wk[1], pp[1]);
        float yn_e_10 = ((mode != 0) ? unpack_o_1[2] * gw8[2] : unpack_o_1[2]);
        float o_e_10 = yn_e_10 * rs_e_1;
        float o2_e_10 = ((mode != 0) ? o_e_10 : o_e_10 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_10, wk[1], pp[2]);
        float yn_e_11 = ((mode != 0) ? unpack_o_1[3] * gw8[3] : unpack_o_1[3]);
        float o_e_11 = yn_e_11 * rs_e_1;
        float o2_e_11 = ((mode != 0) ? o_e_11 : o_e_11 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_11, wk[1], pp[3]);
        float yn_e_12 = ((mode != 0) ? unpack_o_1[4] * gw8[4] : unpack_o_1[4]);
        float o_e_12 = yn_e_12 * rs_e_1;
        float o2_e_12 = ((mode != 0) ? o_e_12 : o_e_12 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_12, wk[1], pp[4]);
        float yn_e_13 = ((mode != 0) ? unpack_o_1[5] * gw8[5] : unpack_o_1[5]);
        float o_e_13 = yn_e_13 * rs_e_1;
        float o2_e_13 = ((mode != 0) ? o_e_13 : o_e_13 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_13, wk[1], pp[5]);
        float yn_e_14 = ((mode != 0) ? unpack_o_1[6] * gw8[6] : unpack_o_1[6]);
        float o_e_14 = yn_e_14 * rs_e_1;
        float o2_e_14 = ((mode != 0) ? o_e_14 : o_e_14 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_14, wk[1], pp[6]);
        float yn_e_15 = ((mode != 0) ? unpack_o_1[7] * gw8[7] : unpack_o_1[7]);
        float o_e_15 = yn_e_15 * rs_e_1;
        float o2_e_15 = ((mode != 0) ? o_e_15 : o_e_15 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_15, wk[1], pp[7]);
#if __CUDA_ARCH__ == 1030
        float _sqrt_45;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_45) : "f"(__fmaf_rn(tot_k[2], 0.00024414062f, eps)));
        float rs_e_2 = __frcp_rn(_sqrt_45);
#else
        float _sqrt_46;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_46) : "f"(__fmaf_rn(tot_k[2], 0.00024414062f, eps)));
        float rs_e_2 = __frcp_rn(_sqrt_46);
#endif
        float unpack_o_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_2[_pair * 2])[0]), "=f"((&unpack_o_2[_pair * 2])[1])
                : "r"(rw[8 + _pair]));
        }
        float yn_e_16 = ((mode != 0) ? unpack_o_2[0] * gw8[0] : unpack_o_2[0]);
        float o_e_16 = yn_e_16 * rs_e_2;
        float o2_e_16 = ((mode != 0) ? o_e_16 : o_e_16 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_16, wk[2], pp[0]);
        float yn_e_17 = ((mode != 0) ? unpack_o_2[1] * gw8[1] : unpack_o_2[1]);
        float o_e_17 = yn_e_17 * rs_e_2;
        float o2_e_17 = ((mode != 0) ? o_e_17 : o_e_17 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_17, wk[2], pp[1]);
        float yn_e_18 = ((mode != 0) ? unpack_o_2[2] * gw8[2] : unpack_o_2[2]);
        float o_e_18 = yn_e_18 * rs_e_2;
        float o2_e_18 = ((mode != 0) ? o_e_18 : o_e_18 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_18, wk[2], pp[2]);
        float yn_e_19 = ((mode != 0) ? unpack_o_2[3] * gw8[3] : unpack_o_2[3]);
        float o_e_19 = yn_e_19 * rs_e_2;
        float o2_e_19 = ((mode != 0) ? o_e_19 : o_e_19 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_19, wk[2], pp[3]);
        float yn_e_20 = ((mode != 0) ? unpack_o_2[4] * gw8[4] : unpack_o_2[4]);
        float o_e_20 = yn_e_20 * rs_e_2;
        float o2_e_20 = ((mode != 0) ? o_e_20 : o_e_20 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_20, wk[2], pp[4]);
        float yn_e_21 = ((mode != 0) ? unpack_o_2[5] * gw8[5] : unpack_o_2[5]);
        float o_e_21 = yn_e_21 * rs_e_2;
        float o2_e_21 = ((mode != 0) ? o_e_21 : o_e_21 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_21, wk[2], pp[5]);
        float yn_e_22 = ((mode != 0) ? unpack_o_2[6] * gw8[6] : unpack_o_2[6]);
        float o_e_22 = yn_e_22 * rs_e_2;
        float o2_e_22 = ((mode != 0) ? o_e_22 : o_e_22 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_22, wk[2], pp[6]);
        float yn_e_23 = ((mode != 0) ? unpack_o_2[7] * gw8[7] : unpack_o_2[7]);
        float o_e_23 = yn_e_23 * rs_e_2;
        float o2_e_23 = ((mode != 0) ? o_e_23 : o_e_23 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_23, wk[2], pp[7]);
#if __CUDA_ARCH__ == 1030
        float _sqrt_46;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_46) : "f"(__fmaf_rn(tot_k[3], 0.00024414062f, eps)));
        float rs_e_3 = __frcp_rn(_sqrt_46);
#else
        float _sqrt_47;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_47) : "f"(__fmaf_rn(tot_k[3], 0.00024414062f, eps)));
        float rs_e_3 = __frcp_rn(_sqrt_47);
#endif
        float unpack_o_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_3[_pair * 2])[0]), "=f"((&unpack_o_3[_pair * 2])[1])
                : "r"(rw[12 + _pair]));
        }
        float yn_e_24 = ((mode != 0) ? unpack_o_3[0] * gw8[0] : unpack_o_3[0]);
        float o_e_24 = yn_e_24 * rs_e_3;
        float o2_e_24 = ((mode != 0) ? o_e_24 : o_e_24 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_24, wk[3], pp[0]);
        float yn_e_25 = ((mode != 0) ? unpack_o_3[1] * gw8[1] : unpack_o_3[1]);
        float o_e_25 = yn_e_25 * rs_e_3;
        float o2_e_25 = ((mode != 0) ? o_e_25 : o_e_25 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_25, wk[3], pp[1]);
        float yn_e_26 = ((mode != 0) ? unpack_o_3[2] * gw8[2] : unpack_o_3[2]);
        float o_e_26 = yn_e_26 * rs_e_3;
        float o2_e_26 = ((mode != 0) ? o_e_26 : o_e_26 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_26, wk[3], pp[2]);
        float yn_e_27 = ((mode != 0) ? unpack_o_3[3] * gw8[3] : unpack_o_3[3]);
        float o_e_27 = yn_e_27 * rs_e_3;
        float o2_e_27 = ((mode != 0) ? o_e_27 : o_e_27 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_27, wk[3], pp[3]);
        float yn_e_28 = ((mode != 0) ? unpack_o_3[4] * gw8[4] : unpack_o_3[4]);
        float o_e_28 = yn_e_28 * rs_e_3;
        float o2_e_28 = ((mode != 0) ? o_e_28 : o_e_28 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_28, wk[3], pp[4]);
        float yn_e_29 = ((mode != 0) ? unpack_o_3[5] * gw8[5] : unpack_o_3[5]);
        float o_e_29 = yn_e_29 * rs_e_3;
        float o2_e_29 = ((mode != 0) ? o_e_29 : o_e_29 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_29, wk[3], pp[5]);
        float yn_e_30 = ((mode != 0) ? unpack_o_3[6] * gw8[6] : unpack_o_3[6]);
        float o_e_30 = yn_e_30 * rs_e_3;
        float o2_e_30 = ((mode != 0) ? o_e_30 : o_e_30 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_30, wk[3], pp[6]);
        float yn_e_31 = ((mode != 0) ? unpack_o_3[7] * gw8[7] : unpack_o_3[7]);
        float o_e_31 = yn_e_31 * rs_e_3;
        float o2_e_31 = ((mode != 0) ? o_e_31 : o_e_31 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_31, wk[3], pp[7]);
        acc[0] = pp[0];
        acc[1] = pp[1];
        acc[2] = pp[2];
        acc[3] = pp[3];
        acc[4] = pp[4];
        acc[5] = pp[5];
        acc[6] = pp[6];
        acc[7] = pp[7];
#if __CUDA_ARCH__ == 1030
        float _sqrt_47;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_47) : "f"(__fmaf_rn(tot_k[4], 0.00024414062f, eps)));
        float rs_e_4 = __frcp_rn(_sqrt_47);
#else
        float _sqrt_48;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_48) : "f"(__fmaf_rn(tot_k[4], 0.00024414062f, eps)));
        float rs_e_4 = __frcp_rn(_sqrt_48);
#endif
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float unpack_o_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_4[_pair * 2])[0]), "=f"((&unpack_o_4[_pair * 2])[1])
                : "r"(rw[16 + _pair]));
        }
        float yn_e_32 = ((mode != 0) ? unpack_o_4[0] * gw8[0] : unpack_o_4[0]);
        float o_e_32 = yn_e_32 * rs_e_4;
        float o2_e_32 = ((mode != 0) ? o_e_32 : o_e_32 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_32, wk[4], pp[0]);
        float yn_e_33 = ((mode != 0) ? unpack_o_4[1] * gw8[1] : unpack_o_4[1]);
        float o_e_33 = yn_e_33 * rs_e_4;
        float o2_e_33 = ((mode != 0) ? o_e_33 : o_e_33 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_33, wk[4], pp[1]);
        float yn_e_34 = ((mode != 0) ? unpack_o_4[2] * gw8[2] : unpack_o_4[2]);
        float o_e_34 = yn_e_34 * rs_e_4;
        float o2_e_34 = ((mode != 0) ? o_e_34 : o_e_34 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_34, wk[4], pp[2]);
        float yn_e_35 = ((mode != 0) ? unpack_o_4[3] * gw8[3] : unpack_o_4[3]);
        float o_e_35 = yn_e_35 * rs_e_4;
        float o2_e_35 = ((mode != 0) ? o_e_35 : o_e_35 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_35, wk[4], pp[3]);
        float yn_e_36 = ((mode != 0) ? unpack_o_4[4] * gw8[4] : unpack_o_4[4]);
        float o_e_36 = yn_e_36 * rs_e_4;
        float o2_e_36 = ((mode != 0) ? o_e_36 : o_e_36 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_36, wk[4], pp[4]);
        float yn_e_37 = ((mode != 0) ? unpack_o_4[5] * gw8[5] : unpack_o_4[5]);
        float o_e_37 = yn_e_37 * rs_e_4;
        float o2_e_37 = ((mode != 0) ? o_e_37 : o_e_37 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_37, wk[4], pp[5]);
        float yn_e_38 = ((mode != 0) ? unpack_o_4[6] * gw8[6] : unpack_o_4[6]);
        float o_e_38 = yn_e_38 * rs_e_4;
        float o2_e_38 = ((mode != 0) ? o_e_38 : o_e_38 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_38, wk[4], pp[6]);
        float yn_e_39 = ((mode != 0) ? unpack_o_4[7] * gw8[7] : unpack_o_4[7]);
        float o_e_39 = yn_e_39 * rs_e_4;
        float o2_e_39 = ((mode != 0) ? o_e_39 : o_e_39 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_39, wk[4], pp[7]);
#if __CUDA_ARCH__ == 1030
        float _sqrt_48;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_48) : "f"(__fmaf_rn(tot_k[5], 0.00024414062f, eps)));
        float rs_e_5 = __frcp_rn(_sqrt_48);
#else
        float _sqrt_49;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_49) : "f"(__fmaf_rn(tot_k[5], 0.00024414062f, eps)));
        float rs_e_5 = __frcp_rn(_sqrt_49);
#endif
        float unpack_o_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_5[_pair * 2])[0]), "=f"((&unpack_o_5[_pair * 2])[1])
                : "r"(rw[20 + _pair]));
        }
        float yn_e_40 = ((mode != 0) ? unpack_o_5[0] * gw8[0] : unpack_o_5[0]);
        float o_e_40 = yn_e_40 * rs_e_5;
        float o2_e_40 = ((mode != 0) ? o_e_40 : o_e_40 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_40, wk[5], pp[0]);
        float yn_e_41 = ((mode != 0) ? unpack_o_5[1] * gw8[1] : unpack_o_5[1]);
        float o_e_41 = yn_e_41 * rs_e_5;
        float o2_e_41 = ((mode != 0) ? o_e_41 : o_e_41 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_41, wk[5], pp[1]);
        float yn_e_42 = ((mode != 0) ? unpack_o_5[2] * gw8[2] : unpack_o_5[2]);
        float o_e_42 = yn_e_42 * rs_e_5;
        float o2_e_42 = ((mode != 0) ? o_e_42 : o_e_42 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_42, wk[5], pp[2]);
        float yn_e_43 = ((mode != 0) ? unpack_o_5[3] * gw8[3] : unpack_o_5[3]);
        float o_e_43 = yn_e_43 * rs_e_5;
        float o2_e_43 = ((mode != 0) ? o_e_43 : o_e_43 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_43, wk[5], pp[3]);
        float yn_e_44 = ((mode != 0) ? unpack_o_5[4] * gw8[4] : unpack_o_5[4]);
        float o_e_44 = yn_e_44 * rs_e_5;
        float o2_e_44 = ((mode != 0) ? o_e_44 : o_e_44 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_44, wk[5], pp[4]);
        float yn_e_45 = ((mode != 0) ? unpack_o_5[5] * gw8[5] : unpack_o_5[5]);
        float o_e_45 = yn_e_45 * rs_e_5;
        float o2_e_45 = ((mode != 0) ? o_e_45 : o_e_45 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_45, wk[5], pp[5]);
        float yn_e_46 = ((mode != 0) ? unpack_o_5[6] * gw8[6] : unpack_o_5[6]);
        float o_e_46 = yn_e_46 * rs_e_5;
        float o2_e_46 = ((mode != 0) ? o_e_46 : o_e_46 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_46, wk[5], pp[6]);
        float yn_e_47 = ((mode != 0) ? unpack_o_5[7] * gw8[7] : unpack_o_5[7]);
        float o_e_47 = yn_e_47 * rs_e_5;
        float o2_e_47 = ((mode != 0) ? o_e_47 : o_e_47 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_47, wk[5], pp[7]);
#if __CUDA_ARCH__ == 1030
        float _sqrt_49;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_49) : "f"(__fmaf_rn(tot_k[6], 0.00024414062f, eps)));
        float rs_e_6 = __frcp_rn(_sqrt_49);
#else
        float _sqrt_50;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_50) : "f"(__fmaf_rn(tot_k[6], 0.00024414062f, eps)));
        float rs_e_6 = __frcp_rn(_sqrt_50);
#endif
        float unpack_o_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_6[_pair * 2])[0]), "=f"((&unpack_o_6[_pair * 2])[1])
                : "r"(rw[24 + _pair]));
        }
        float yn_e_48 = ((mode != 0) ? unpack_o_6[0] * gw8[0] : unpack_o_6[0]);
        float o_e_48 = yn_e_48 * rs_e_6;
        float o2_e_48 = ((mode != 0) ? o_e_48 : o_e_48 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_48, wk[6], pp[0]);
        float yn_e_49 = ((mode != 0) ? unpack_o_6[1] * gw8[1] : unpack_o_6[1]);
        float o_e_49 = yn_e_49 * rs_e_6;
        float o2_e_49 = ((mode != 0) ? o_e_49 : o_e_49 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_49, wk[6], pp[1]);
        float yn_e_50 = ((mode != 0) ? unpack_o_6[2] * gw8[2] : unpack_o_6[2]);
        float o_e_50 = yn_e_50 * rs_e_6;
        float o2_e_50 = ((mode != 0) ? o_e_50 : o_e_50 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_50, wk[6], pp[2]);
        float yn_e_51 = ((mode != 0) ? unpack_o_6[3] * gw8[3] : unpack_o_6[3]);
        float o_e_51 = yn_e_51 * rs_e_6;
        float o2_e_51 = ((mode != 0) ? o_e_51 : o_e_51 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_51, wk[6], pp[3]);
        float yn_e_52 = ((mode != 0) ? unpack_o_6[4] * gw8[4] : unpack_o_6[4]);
        float o_e_52 = yn_e_52 * rs_e_6;
        float o2_e_52 = ((mode != 0) ? o_e_52 : o_e_52 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_52, wk[6], pp[4]);
        float yn_e_53 = ((mode != 0) ? unpack_o_6[5] * gw8[5] : unpack_o_6[5]);
        float o_e_53 = yn_e_53 * rs_e_6;
        float o2_e_53 = ((mode != 0) ? o_e_53 : o_e_53 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_53, wk[6], pp[5]);
        float yn_e_54 = ((mode != 0) ? unpack_o_6[6] * gw8[6] : unpack_o_6[6]);
        float o_e_54 = yn_e_54 * rs_e_6;
        float o2_e_54 = ((mode != 0) ? o_e_54 : o_e_54 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_54, wk[6], pp[6]);
        float yn_e_55 = ((mode != 0) ? unpack_o_6[7] * gw8[7] : unpack_o_6[7]);
        float o_e_55 = yn_e_55 * rs_e_6;
        float o2_e_55 = ((mode != 0) ? o_e_55 : o_e_55 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_55, wk[6], pp[7]);
#if __CUDA_ARCH__ == 1030
        float _sqrt_50;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_50) : "f"(__fmaf_rn(tot_k[7], 0.00024414062f, eps)));
        float rs_e_7 = __frcp_rn(_sqrt_50);
#else
        float _sqrt_51;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_51) : "f"(__fmaf_rn(tot_k[7], 0.00024414062f, eps)));
        float rs_e_7 = __frcp_rn(_sqrt_51);
#endif
        float unpack_o_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&unpack_o_7[_pair * 2])[0]), "=f"((&unpack_o_7[_pair * 2])[1])
                : "r"(rw[28 + _pair]));
        }
        float yn_e_56 = ((mode != 0) ? unpack_o_7[0] * gw8[0] : unpack_o_7[0]);
        float o_e_56 = yn_e_56 * rs_e_7;
        float o2_e_56 = ((mode != 0) ? o_e_56 : o_e_56 * gw8[0]);
        pp[0] = __fmaf_rn(o2_e_56, wk[7], pp[0]);
        float yn_e_57 = ((mode != 0) ? unpack_o_7[1] * gw8[1] : unpack_o_7[1]);
        float o_e_57 = yn_e_57 * rs_e_7;
        float o2_e_57 = ((mode != 0) ? o_e_57 : o_e_57 * gw8[1]);
        pp[1] = __fmaf_rn(o2_e_57, wk[7], pp[1]);
        float yn_e_58 = ((mode != 0) ? unpack_o_7[2] * gw8[2] : unpack_o_7[2]);
        float o_e_58 = yn_e_58 * rs_e_7;
        float o2_e_58 = ((mode != 0) ? o_e_58 : o_e_58 * gw8[2]);
        pp[2] = __fmaf_rn(o2_e_58, wk[7], pp[2]);
        float yn_e_59 = ((mode != 0) ? unpack_o_7[3] * gw8[3] : unpack_o_7[3]);
        float o_e_59 = yn_e_59 * rs_e_7;
        float o2_e_59 = ((mode != 0) ? o_e_59 : o_e_59 * gw8[3]);
        pp[3] = __fmaf_rn(o2_e_59, wk[7], pp[3]);
        float yn_e_60 = ((mode != 0) ? unpack_o_7[4] * gw8[4] : unpack_o_7[4]);
        float o_e_60 = yn_e_60 * rs_e_7;
        float o2_e_60 = ((mode != 0) ? o_e_60 : o_e_60 * gw8[4]);
        pp[4] = __fmaf_rn(o2_e_60, wk[7], pp[4]);
        float yn_e_61 = ((mode != 0) ? unpack_o_7[5] * gw8[5] : unpack_o_7[5]);
        float o_e_61 = yn_e_61 * rs_e_7;
        float o2_e_61 = ((mode != 0) ? o_e_61 : o_e_61 * gw8[5]);
        pp[5] = __fmaf_rn(o2_e_61, wk[7], pp[5]);
        float yn_e_62 = ((mode != 0) ? unpack_o_7[6] * gw8[6] : unpack_o_7[6]);
        float o_e_62 = yn_e_62 * rs_e_7;
        float o2_e_62 = ((mode != 0) ? o_e_62 : o_e_62 * gw8[6]);
        pp[6] = __fmaf_rn(o2_e_62, wk[7], pp[6]);
        float yn_e_63 = ((mode != 0) ? unpack_o_7[7] * gw8[7] : unpack_o_7[7]);
        float o_e_63 = yn_e_63 * rs_e_7;
        float o2_e_63 = ((mode != 0) ? o_e_63 : o_e_63 * gw8[7]);
        pp[7] = __fmaf_rn(o2_e_63, wk[7], pp[7]);
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
#if __CUDA_ARCH__ == 1030
            uint32_t _p16s_carrier_51 = static_cast<uint32_t>(acc_bf16[_p16s_word]);
            uint32_t _p16s_lo_51 = _p16s_carrier_51 & 0x0000ffffu;
            uint32_t _p16s_hi_51 = (_p16s_carrier_51 >> 16) & 0x0000ffffu;
            if (_p16s_lo_51 == 0x00008000u) _p16s_lo_51 = 0u;
            if (_p16s_hi_51 == 0x00008000u) _p16s_hi_51 = 0u;
            _packed16_sanitize_8[_p16s_word] = _p16s_lo_51 | (_p16s_hi_51 << 16);
#else
            uint32_t _p16s_carrier_52 = static_cast<uint32_t>(acc_bf16[_p16s_word]);
            uint32_t _p16s_lo_52 = _p16s_carrier_52 & 0x0000ffffu;
            uint32_t _p16s_hi_52 = (_p16s_carrier_52 >> 16) & 0x0000ffffu;
            if (_p16s_lo_52 == 0x00008000u) _p16s_lo_52 = 0u;
            if (_p16s_hi_52 == 0x00008000u) _p16s_hi_52 = 0u;
            _packed16_sanitize_8[_p16s_word] = _p16s_lo_52 | (_p16s_hi_52 << 16);
#endif
        }
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(r_peer0 + (r_cur + rank64 * 4096)), "r"((_packed16_sanitize_8)[0]), "r"((_packed16_sanitize_8)[1]), "r"((_packed16_sanitize_8)[2]), "r"((_packed16_sanitize_8)[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(r_peer1 + (r_cur + rank64 * 4096)), "r"((_packed16_sanitize_8)[0]), "r"((_packed16_sanitize_8)[1]), "r"((_packed16_sanitize_8)[2]), "r"((_packed16_sanitize_8)[3]) : "memory");
    } else {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    if (tok_push < n_tok && d != rank_i) {
        uint32_t _sysv_poll_group_1[4];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(r_local + (r_cur + dest64 * 4096))) : "memory");
        } while (
            (((_sysv_poll_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[3] >> 16) & 0xffffu) == 0x8000u)
        );
        unsigned int row4[4];
        row4[0] = _sysv_poll_group_1[0];
        row4[1] = _sysv_poll_group_1[1];
        row4[2] = _sysv_poll_group_1[2];
        row4[3] = _sysv_poll_group_1[3];
        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output + ((long long)tok_push * 4096 + col64)))[0] = reinterpret_cast<int4*>(row4)[0];
    }
    int gn_next = ((gn == 2) ? 0 : gn + 1);
    if (tid == 0 && cta_rank_0 == 0) {
        *(reinterpret_cast<int*>(gen + gidx) + (0)) = gn_next;
    }
}

} // extern "C"
