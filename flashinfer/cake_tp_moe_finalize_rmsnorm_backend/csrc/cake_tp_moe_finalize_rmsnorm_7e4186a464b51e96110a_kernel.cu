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
#define SMEM_WARP_TAB_STAGE_BYTES 64
#define SMEM_WARP_TAB_STRIDE 64
#define SMEM_CTA_TAB_OFF 64
#define SMEM_CTA_TAB_STAGE_BYTES 64
#define SMEM_CTA_TAB_STRIDE 64
#define SMEM_FLAG_OFF 128
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 256
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_tp_moe_finalize_rmsnorm_7e4186a464b51e96110a(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen, float* __restrict__ part, unsigned int* __restrict__ ctr, unsigned long long r_mcast)
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
    float* cta_tab = reinterpret_cast<float*>(smem_raw + 64);
    const int cta_tab_addr = smem + 64;
    unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 128);
    const int flag_addr = smem + 128;

    // === Task calls (declaration order) ===
    int band = blockIdx.x;
    int g = blockIdx.y;
    int d = blockIdx.z;
    int cta_rank_0 = cta_rank;
    int rank_i = rank;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
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
    long long xf_old = (long long)d * 4;
    {
        auto _gst_28 = a_local + (a_old + xf_old * 4096);
        __builtin_assume(__isGlobal(_gst_28));
        reinterpret_cast<int4*>(_gst_28)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_29 = a_local + (a_old + (8 + xf_old) * 4096);
        __builtin_assume(__isGlobal(_gst_29));
        reinterpret_cast<int4*>(_gst_29)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_30 = a_local + (a_old + (xf_old + 1) * 4096);
        __builtin_assume(__isGlobal(_gst_30));
        reinterpret_cast<int4*>(_gst_30)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_31 = a_local + (a_old + (8 + (xf_old + 1)) * 4096);
        __builtin_assume(__isGlobal(_gst_31));
        reinterpret_cast<int4*>(_gst_31)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_32 = a_local + (a_old + (xf_old + 2) * 4096);
        __builtin_assume(__isGlobal(_gst_32));
        reinterpret_cast<int4*>(_gst_32)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_33 = a_local + (a_old + (8 + (xf_old + 2)) * 4096);
        __builtin_assume(__isGlobal(_gst_33));
        reinterpret_cast<int4*>(_gst_33)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_34 = a_local + (a_old + (xf_old + 3) * 4096);
        __builtin_assume(__isGlobal(_gst_34));
        reinterpret_cast<int4*>(_gst_34)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_35 = a_local + (a_old + (8 + (xf_old + 3)) * 4096);
        __builtin_assume(__isGlobal(_gst_35));
        reinterpret_cast<int4*>(_gst_35)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    if (tok_own < n_tok) {
        long long xf = (long long)d * 4;
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
        float wk[4];
        wk[0] = ((d == 1) ? w8o[4] : w8o[0]);
        wk[1] = ((d == 1) ? w8o[5] : w8o[1]);
        wk[2] = ((d == 1) ? w8o[6] : w8o[2]);
        wk[3] = ((d == 1) ? w8o[7] : w8o[3]);
        unsigned int rw[16];
        float ssq_k[4];
        uint32_t _sysv_poll_group_0[32];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + xf * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + xf) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 1) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 1)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 2) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 2)) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (xf + 3) * 4096))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + (8 + (xf + 3)) * 4096))) : "memory");
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
            (((_sysv_poll_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 16) & 0xffffu) == 0x8000u)
        );
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
        float warp_reduce_r_0 = ssq_k[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_0 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_0, offset);
        if (lane == 0) {
            warp_tab[warp * 4] = warp_reduce_r_0;
        }
        float warp_reduce_r_1 = ssq_k[1];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_1 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_1, offset);
        if (lane == 0) {
            warp_tab[warp * 4 + 1] = warp_reduce_r_1;
        }
        float warp_reduce_r_2 = ssq_k[2];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_2 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_2, offset);
        if (lane == 0) {
            warp_tab[warp * 4 + 2] = warp_reduce_r_2;
        }
        float warp_reduce_r_3 = ssq_k[3];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_r_3 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_r_3, offset);
        if (lane == 0) {
            warp_tab[warp * 4 + 3] = warp_reduce_r_3;
        }
        __syncthreads();
        float tot_k[4];
        int kk_t = tid & 3;
        float part_t = warp_tab[kk_t];
        part_t = part_t + warp_tab[4 + kk_t];
        part_t = part_t + warp_tab[8 + kk_t];
        part_t = part_t + warp_tab[12 + kk_t];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (tid < 4) {
            unsigned int slot_addr = cta_tab_addr + (unsigned int)((cta_rank_0 * 4 + tid) * 4);
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
        tot_k[0] = tot_k[0] + cta_tab[4];
        tot_k[0] = tot_k[0] + cta_tab[8];
        tot_k[0] = tot_k[0] + cta_tab[12];
        tot_k[1] = cta_tab[1];
        tot_k[1] = tot_k[1] + cta_tab[5];
        tot_k[1] = tot_k[1] + cta_tab[9];
        tot_k[1] = tot_k[1] + cta_tab[13];
        tot_k[2] = cta_tab[2];
        tot_k[2] = tot_k[2] + cta_tab[6];
        tot_k[2] = tot_k[2] + cta_tab[10];
        tot_k[2] = tot_k[2] + cta_tab[14];
        tot_k[3] = cta_tab[3];
        tot_k[3] = tot_k[3] + cta_tab[7];
        tot_k[3] = tot_k[3] + cta_tab[11];
        tot_k[3] = tot_k[3] + cta_tab[15];
        float pp[8];
        float acc[8];
        float _sqrt_43;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_43) : "f"(__fmaf_rn(tot_k[0], 0.00024414062f, eps)));
        float rs_e_0 = __frcp_rn(_sqrt_43);
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
        float _sqrt_44;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_44) : "f"(__fmaf_rn(tot_k[1], 0.00024414062f, eps)));
        float rs_e_1 = __frcp_rn(_sqrt_44);
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
        float _sqrt_45;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_45) : "f"(__fmaf_rn(tot_k[2], 0.00024414062f, eps)));
        float rs_e_2 = __frcp_rn(_sqrt_45);
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
        float _sqrt_46;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_46) : "f"(__fmaf_rn(tot_k[3], 0.00024414062f, eps)));
        float rs_e_3 = __frcp_rn(_sqrt_46);
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
        long long c_base = slot64 * 8192 + col64;
        float lo4[4];
        float hi4[4];
        lo4[0] = acc[0];
        hi4[0] = acc[4];
        lo4[1] = acc[1];
        hi4[1] = acc[5];
        lo4[2] = acc[2];
        hi4[2] = acc[6];
        lo4[3] = acc[3];
        hi4[3] = acc[7];
        {
            float4 _v4 = make_float4(lo4[0 + 0], lo4[0 + 1], lo4[0 + 2], lo4[0 + 3]);
            *reinterpret_cast<float4*>((part + (c_base + dest64 * 4096)) + 0) = _v4;
        }
        {
            float4 _v4 = make_float4(hi4[0 + 0], hi4[0 + 1], hi4[0 + 2], hi4[0 + 3]);
            *reinterpret_cast<float4*>((part + (c_base + dest64 * 4096 + 4)) + 0) = _v4;
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int _atomic_fetch_add_47 = atomicAdd(&ctr[g * 8 + band], 1);
            flag[0] = _atomic_fetch_add_47;
        }
        __syncthreads();
        unsigned int arrived = flag[0];
        if ((arrived & 1) == 1) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            float out8[8];
            float _vec_load_4[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(part + c_base + 0);
                _vec_load_4[0 + 0] = _v4.x;
                _vec_load_4[0 + 1] = _v4.y;
                _vec_load_4[0 + 2] = _v4.z;
                _vec_load_4[0 + 3] = _v4.w;
            }
            float _vec_load_5[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(part + (c_base + 4) + 0);
                _vec_load_5[0 + 0] = _v4.x;
                _vec_load_5[0 + 1] = _v4.y;
                _vec_load_5[0 + 2] = _v4.z;
                _vec_load_5[0 + 3] = _v4.w;
            }
            out8[0] = _vec_load_4[0];
            out8[4] = _vec_load_5[0];
            out8[1] = _vec_load_4[1];
            out8[5] = _vec_load_5[1];
            out8[2] = _vec_load_4[2];
            out8[6] = _vec_load_5[2];
            out8[3] = _vec_load_4[3];
            out8[7] = _vec_load_5[3];
            float _vec_load_6[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(part + (c_base + 4096) + 0);
                _vec_load_6[0 + 0] = _v4.x;
                _vec_load_6[0 + 1] = _v4.y;
                _vec_load_6[0 + 2] = _v4.z;
                _vec_load_6[0 + 3] = _v4.w;
            }
            float _vec_load_7[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(part + (c_base + 4100) + 0);
                _vec_load_7[0 + 0] = _v4.x;
                _vec_load_7[0 + 1] = _v4.y;
                _vec_load_7[0 + 2] = _v4.z;
                _vec_load_7[0 + 3] = _v4.w;
            }
            out8[0] = out8[0] + _vec_load_6[0];
            out8[4] = out8[4] + _vec_load_7[0];
            out8[1] = out8[1] + _vec_load_6[1];
            out8[5] = out8[5] + _vec_load_7[1];
            out8[2] = out8[2] + _vec_load_6[2];
            out8[6] = out8[6] + _vec_load_7[2];
            out8[3] = out8[3] + _vec_load_6[3];
            out8[7] = out8[7] + _vec_load_7[3];
            unsigned int out8_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out8[_lp*2 + 0], out8[_lp*2+1 + 0]));
                out8_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output + ((long long)tok_own * 4096 + col64)))[0] = reinterpret_cast<int4*>(out8_bf16)[0];
            uint32_t _packed16_sanitize_8[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_52 = static_cast<uint32_t>(out8_bf16[_p16s_word]);
                uint32_t _p16s_lo_52 = _p16s_carrier_52 & 0x0000ffffu;
                uint32_t _p16s_hi_52 = (_p16s_carrier_52 >> 16) & 0x0000ffffu;
                if (_p16s_lo_52 == 0x00008000u) _p16s_lo_52 = 0u;
                if (_p16s_hi_52 == 0x00008000u) _p16s_hi_52 = 0u;
                _packed16_sanitize_8[_p16s_word] = _p16s_lo_52 | (_p16s_hi_52 << 16);
            }
            asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(r_peer0 + (r_cur + rank64 * 4096)), "r"((_packed16_sanitize_8)[0]), "r"((_packed16_sanitize_8)[1]), "r"((_packed16_sanitize_8)[2]), "r"((_packed16_sanitize_8)[3]) : "memory");
            asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(r_peer1 + (r_cur + rank64 * 4096)), "r"((_packed16_sanitize_8)[0]), "r"((_packed16_sanitize_8)[1]), "r"((_packed16_sanitize_8)[2]), "r"((_packed16_sanitize_8)[3]) : "memory");
        }
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
