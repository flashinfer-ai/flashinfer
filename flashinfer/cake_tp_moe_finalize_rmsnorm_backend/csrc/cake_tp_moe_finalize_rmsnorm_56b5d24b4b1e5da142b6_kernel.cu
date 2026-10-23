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
#define SMEM_WARP_TAB_STAGE_BYTES 256
#define SMEM_WARP_TAB_STRIDE 256
#define SMEM_CTA_TAB_OFF 256
#define SMEM_CTA_TAB_STAGE_BYTES 16
#define SMEM_CTA_TAB_STRIDE 16
#define SMEM_FLAG_OFF 272
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 384
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_tp_moe_finalize_rmsnorm_56b5d24b4b1e5da142b6(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen, float* __restrict__ part, unsigned int* __restrict__ ctr, unsigned long long r_mcast)
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
    float* warp_tab = reinterpret_cast<float*>(smem_raw + 0);
    const int warp_tab_addr = smem + 0;
    float* cta_tab = reinterpret_cast<float*>(smem_raw + 256);
    const int cta_tab_addr = smem + 256;
    unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 272);
    const int flag_addr = smem + 272;

    // === Task calls (declaration order) ===
    int band = blockIdx.x;
    int g = blockIdx.y;
    int d = blockIdx.z;
    int cta_rank_1 = 0;
    int rank_i = rank;
    int col = (band * 512 + tid) * 8;
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
    int ridx1 = idx8[1];
    if (ridx1 >= 0 && ridx1 < rows) {
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1 * (long long)row_stride + col64) + 0);
            packets[4 + 0] = _uv4_5.x;
            packets[4 + 1] = _uv4_5.y;
            packets[4 + 2] = _uv4_5.z;
            packets[4 + 3] = _uv4_5.w;
        }
    }
    int ridx2 = idx8[2];
    if (ridx2 >= 0 && ridx2 < rows) {
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx2 * (long long)row_stride + col64) + 0);
            packets[8 + 0] = _uv4_6.x;
            packets[8 + 1] = _uv4_6.y;
            packets[8 + 2] = _uv4_6.z;
            packets[8 + 3] = _uv4_6.w;
        }
    }
    int ridx3 = idx8[3];
    if (ridx3 >= 0 && ridx3 < rows) {
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx3 * (long long)row_stride + col64) + 0);
            packets[12 + 0] = _uv4_7.x;
            packets[12 + 1] = _uv4_7.y;
            packets[12 + 2] = _uv4_7.z;
            packets[12 + 3] = _uv4_7.w;
        }
    }
    int ridx4 = idx8[4];
    if (ridx4 >= 0 && ridx4 < rows) {
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx4 * (long long)row_stride + col64) + 0);
            packets[16 + 0] = _uv4_8.x;
            packets[16 + 1] = _uv4_8.y;
            packets[16 + 2] = _uv4_8.z;
            packets[16 + 3] = _uv4_8.w;
        }
    }
    int ridx5 = idx8[5];
    if (ridx5 >= 0 && ridx5 < rows) {
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx5 * (long long)row_stride + col64) + 0);
            packets[20 + 0] = _uv4_9.x;
            packets[20 + 1] = _uv4_9.y;
            packets[20 + 2] = _uv4_9.z;
            packets[20 + 3] = _uv4_9.w;
        }
    }
    int ridx6 = idx8[6];
    if (ridx6 >= 0 && ridx6 < rows) {
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx6 * (long long)row_stride + col64) + 0);
            packets[24 + 0] = _uv4_10.x;
            packets[24 + 1] = _uv4_10.y;
            packets[24 + 2] = _uv4_10.z;
            packets[24 + 3] = _uv4_10.w;
        }
    }
    int ridx7 = idx8[7];
    if (ridx7 >= 0 && ridx7 < rows) {
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx7 * (long long)row_stride + col64) + 0);
            packets[28 + 0] = _uv4_11.x;
            packets[28 + 1] = _uv4_11.y;
            packets[28 + 2] = _uv4_11.z;
            packets[28 + 3] = _uv4_11.w;
        }
    }
    if (tok_push < n_tok) {
        unsigned int raw0[4];
        raw0[0] = packets[0];
        raw0[1] = packets[1];
        raw0[2] = packets[2];
        raw0[3] = packets[3];
        uint32_t _packed16_sanitize_0[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_12 = static_cast<uint32_t>(raw0[_p16s_word]);
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
        unsigned int raw1[4];
        raw1[0] = packets[4];
        raw1[1] = packets[5];
        raw1[2] = packets[6];
        raw1[3] = packets[7];
        uint32_t _packed16_sanitize_1[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_14 = static_cast<uint32_t>(raw1[_p16s_word]);
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
        unsigned int raw2[4];
        raw2[0] = packets[8];
        raw2[1] = packets[9];
        raw2[2] = packets[10];
        raw2[3] = packets[11];
        uint32_t _packed16_sanitize_2[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_16 = static_cast<uint32_t>(raw2[_p16s_word]);
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
        unsigned int raw3[4];
        raw3[0] = packets[12];
        raw3[1] = packets[13];
        raw3[2] = packets[14];
        raw3[3] = packets[15];
        uint32_t _packed16_sanitize_3[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_18 = static_cast<uint32_t>(raw3[_p16s_word]);
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
        unsigned int raw4[4];
        raw4[0] = packets[16];
        raw4[1] = packets[17];
        raw4[2] = packets[18];
        raw4[3] = packets[19];
        uint32_t _packed16_sanitize_4[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_20 = static_cast<uint32_t>(raw4[_p16s_word]);
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
        unsigned int raw5[4];
        raw5[0] = packets[20];
        raw5[1] = packets[21];
        raw5[2] = packets[22];
        raw5[3] = packets[23];
        uint32_t _packed16_sanitize_5[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_22 = static_cast<uint32_t>(raw5[_p16s_word]);
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
        unsigned int raw6[4];
        raw6[0] = packets[24];
        raw6[1] = packets[25];
        raw6[2] = packets[26];
        raw6[3] = packets[27];
        uint32_t _packed16_sanitize_6[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_24 = static_cast<uint32_t>(raw6[_p16s_word]);
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
        unsigned int raw7[4];
        raw7[0] = packets[28];
        raw7[1] = packets[29];
        raw7[2] = packets[30];
        raw7[3] = packets[31];
        uint32_t _packed16_sanitize_7[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_26 = static_cast<uint32_t>(raw7[_p16s_word]);
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
        float _unpack_bf16_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[_pair]));
        }
        float _unpack_bf16_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[4 + _pair]));
        }
        float s8_0[8];
        float t0_0_0 = (float)_unpack_bf16_2[0];
        s8_0[0] = t0_0_0;
        float t0_0_1 = (float)_unpack_bf16_3[0];
        s8_0[0] = s8_0[0] + t0_0_1;
        float t0_1_0 = (float)_unpack_bf16_2[1];
        s8_0[1] = t0_1_0;
        float t0_1_1 = (float)_unpack_bf16_3[1];
        s8_0[1] = s8_0[1] + t0_1_1;
        float t0_2_0 = (float)_unpack_bf16_2[2];
        s8_0[2] = t0_2_0;
        float t0_2_1 = (float)_unpack_bf16_3[2];
        s8_0[2] = s8_0[2] + t0_2_1;
        float t0_3_0 = (float)_unpack_bf16_2[3];
        s8_0[3] = t0_3_0;
        float t0_3_1 = (float)_unpack_bf16_3[3];
        s8_0[3] = s8_0[3] + t0_3_1;
        float t0_4_0 = (float)_unpack_bf16_2[4];
        s8_0[4] = t0_4_0;
        float t0_4_1 = (float)_unpack_bf16_3[4];
        s8_0[4] = s8_0[4] + t0_4_1;
        float t0_5_0 = (float)_unpack_bf16_2[5];
        s8_0[5] = t0_5_0;
        float t0_5_1 = (float)_unpack_bf16_3[5];
        s8_0[5] = s8_0[5] + t0_5_1;
        float t0_6_0 = (float)_unpack_bf16_2[6];
        s8_0[6] = t0_6_0;
        float t0_6_1 = (float)_unpack_bf16_3[6];
        s8_0[6] = s8_0[6] + t0_6_1;
        float t0_7_0 = (float)_unpack_bf16_2[7];
        s8_0[7] = t0_7_0;
        float t0_7_1 = (float)_unpack_bf16_3[7];
        s8_0[7] = s8_0[7] + t0_7_1;
        unsigned int s8_0_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_0[_lp*2 + 0], s8_0[_lp*2+1 + 0]));
            s8_0_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        rw[0] = s8_0_bf16[0];
        rw[1] = s8_0_bf16[1];
        rw[2] = s8_0_bf16[2];
        rw[3] = s8_0_bf16[3];
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                : "r"(s8_0_bf16[_pair]));
        }
        float ssq0 = 0.0f;
        float y0_0 = ((mode != 0) ? _unpack_bf16_4[0] * gw8[0] : _unpack_bf16_4[0]);
        ssq0 = __fmaf_rn(y0_0, y0_0, ssq0);
        float y0_1 = ((mode != 0) ? _unpack_bf16_4[1] * gw8[1] : _unpack_bf16_4[1]);
        ssq0 = __fmaf_rn(y0_1, y0_1, ssq0);
        float y0_2 = ((mode != 0) ? _unpack_bf16_4[2] * gw8[2] : _unpack_bf16_4[2]);
        ssq0 = __fmaf_rn(y0_2, y0_2, ssq0);
        float y0_3 = ((mode != 0) ? _unpack_bf16_4[3] * gw8[3] : _unpack_bf16_4[3]);
        ssq0 = __fmaf_rn(y0_3, y0_3, ssq0);
        float y0_4 = ((mode != 0) ? _unpack_bf16_4[4] * gw8[4] : _unpack_bf16_4[4]);
        ssq0 = __fmaf_rn(y0_4, y0_4, ssq0);
        float y0_5 = ((mode != 0) ? _unpack_bf16_4[5] * gw8[5] : _unpack_bf16_4[5]);
        ssq0 = __fmaf_rn(y0_5, y0_5, ssq0);
        float y0_6 = ((mode != 0) ? _unpack_bf16_4[6] * gw8[6] : _unpack_bf16_4[6]);
        ssq0 = __fmaf_rn(y0_6, y0_6, ssq0);
        float y0_7 = ((mode != 0) ? _unpack_bf16_4[7] * gw8[7] : _unpack_bf16_4[7]);
        ssq0 = __fmaf_rn(y0_7, y0_7, ssq0);
        ssq_k[0] = ssq0;
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[8 + _pair]));
        }
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[12 + _pair]));
        }
        float s8_1[8];
        float t1_0_0 = (float)_unpack_bf16_5[0];
        s8_1[0] = t1_0_0;
        float t1_0_1 = (float)_unpack_bf16_6[0];
        s8_1[0] = s8_1[0] + t1_0_1;
        float t1_1_0 = (float)_unpack_bf16_5[1];
        s8_1[1] = t1_1_0;
        float t1_1_1 = (float)_unpack_bf16_6[1];
        s8_1[1] = s8_1[1] + t1_1_1;
        float t1_2_0 = (float)_unpack_bf16_5[2];
        s8_1[2] = t1_2_0;
        float t1_2_1 = (float)_unpack_bf16_6[2];
        s8_1[2] = s8_1[2] + t1_2_1;
        float t1_3_0 = (float)_unpack_bf16_5[3];
        s8_1[3] = t1_3_0;
        float t1_3_1 = (float)_unpack_bf16_6[3];
        s8_1[3] = s8_1[3] + t1_3_1;
        float t1_4_0 = (float)_unpack_bf16_5[4];
        s8_1[4] = t1_4_0;
        float t1_4_1 = (float)_unpack_bf16_6[4];
        s8_1[4] = s8_1[4] + t1_4_1;
        float t1_5_0 = (float)_unpack_bf16_5[5];
        s8_1[5] = t1_5_0;
        float t1_5_1 = (float)_unpack_bf16_6[5];
        s8_1[5] = s8_1[5] + t1_5_1;
        float t1_6_0 = (float)_unpack_bf16_5[6];
        s8_1[6] = t1_6_0;
        float t1_6_1 = (float)_unpack_bf16_6[6];
        s8_1[6] = s8_1[6] + t1_6_1;
        float t1_7_0 = (float)_unpack_bf16_5[7];
        s8_1[7] = t1_7_0;
        float t1_7_1 = (float)_unpack_bf16_6[7];
        s8_1[7] = s8_1[7] + t1_7_1;
        unsigned int s8_1_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_1[_lp*2 + 0], s8_1[_lp*2+1 + 0]));
            s8_1_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        rw[4] = s8_1_bf16[0];
        rw[5] = s8_1_bf16[1];
        rw[6] = s8_1_bf16[2];
        rw[7] = s8_1_bf16[3];
        float _unpack_bf16_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                : "r"(s8_1_bf16[_pair]));
        }
        float ssq1 = 0.0f;
        float y1_0 = ((mode != 0) ? _unpack_bf16_7[0] * gw8[0] : _unpack_bf16_7[0]);
        ssq1 = __fmaf_rn(y1_0, y1_0, ssq1);
        float y1_1 = ((mode != 0) ? _unpack_bf16_7[1] * gw8[1] : _unpack_bf16_7[1]);
        ssq1 = __fmaf_rn(y1_1, y1_1, ssq1);
        float y1_2 = ((mode != 0) ? _unpack_bf16_7[2] * gw8[2] : _unpack_bf16_7[2]);
        ssq1 = __fmaf_rn(y1_2, y1_2, ssq1);
        float y1_3 = ((mode != 0) ? _unpack_bf16_7[3] * gw8[3] : _unpack_bf16_7[3]);
        ssq1 = __fmaf_rn(y1_3, y1_3, ssq1);
        float y1_4 = ((mode != 0) ? _unpack_bf16_7[4] * gw8[4] : _unpack_bf16_7[4]);
        ssq1 = __fmaf_rn(y1_4, y1_4, ssq1);
        float y1_5 = ((mode != 0) ? _unpack_bf16_7[5] * gw8[5] : _unpack_bf16_7[5]);
        ssq1 = __fmaf_rn(y1_5, y1_5, ssq1);
        float y1_6 = ((mode != 0) ? _unpack_bf16_7[6] * gw8[6] : _unpack_bf16_7[6]);
        ssq1 = __fmaf_rn(y1_6, y1_6, ssq1);
        float y1_7 = ((mode != 0) ? _unpack_bf16_7[7] * gw8[7] : _unpack_bf16_7[7]);
        ssq1 = __fmaf_rn(y1_7, y1_7, ssq1);
        ssq_k[1] = ssq1;
        float _unpack_bf16_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[16 + _pair]));
        }
        float _unpack_bf16_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[20 + _pair]));
        }
        float s8_2[8];
        float t2_0_0 = (float)_unpack_bf16_8[0];
        s8_2[0] = t2_0_0;
        float t2_0_1 = (float)_unpack_bf16_9[0];
        s8_2[0] = s8_2[0] + t2_0_1;
        float t2_1_0 = (float)_unpack_bf16_8[1];
        s8_2[1] = t2_1_0;
        float t2_1_1 = (float)_unpack_bf16_9[1];
        s8_2[1] = s8_2[1] + t2_1_1;
        float t2_2_0 = (float)_unpack_bf16_8[2];
        s8_2[2] = t2_2_0;
        float t2_2_1 = (float)_unpack_bf16_9[2];
        s8_2[2] = s8_2[2] + t2_2_1;
        float t2_3_0 = (float)_unpack_bf16_8[3];
        s8_2[3] = t2_3_0;
        float t2_3_1 = (float)_unpack_bf16_9[3];
        s8_2[3] = s8_2[3] + t2_3_1;
        float t2_4_0 = (float)_unpack_bf16_8[4];
        s8_2[4] = t2_4_0;
        float t2_4_1 = (float)_unpack_bf16_9[4];
        s8_2[4] = s8_2[4] + t2_4_1;
        float t2_5_0 = (float)_unpack_bf16_8[5];
        s8_2[5] = t2_5_0;
        float t2_5_1 = (float)_unpack_bf16_9[5];
        s8_2[5] = s8_2[5] + t2_5_1;
        float t2_6_0 = (float)_unpack_bf16_8[6];
        s8_2[6] = t2_6_0;
        float t2_6_1 = (float)_unpack_bf16_9[6];
        s8_2[6] = s8_2[6] + t2_6_1;
        float t2_7_0 = (float)_unpack_bf16_8[7];
        s8_2[7] = t2_7_0;
        float t2_7_1 = (float)_unpack_bf16_9[7];
        s8_2[7] = s8_2[7] + t2_7_1;
        unsigned int s8_2_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_2[_lp*2 + 0], s8_2[_lp*2+1 + 0]));
            s8_2_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        rw[8] = s8_2_bf16[0];
        rw[9] = s8_2_bf16[1];
        rw[10] = s8_2_bf16[2];
        rw[11] = s8_2_bf16[3];
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                : "r"(s8_2_bf16[_pair]));
        }
        float ssq2 = 0.0f;
        float y2_0 = ((mode != 0) ? _unpack_bf16_10[0] * gw8[0] : _unpack_bf16_10[0]);
        ssq2 = __fmaf_rn(y2_0, y2_0, ssq2);
        float y2_1 = ((mode != 0) ? _unpack_bf16_10[1] * gw8[1] : _unpack_bf16_10[1]);
        ssq2 = __fmaf_rn(y2_1, y2_1, ssq2);
        float y2_2 = ((mode != 0) ? _unpack_bf16_10[2] * gw8[2] : _unpack_bf16_10[2]);
        ssq2 = __fmaf_rn(y2_2, y2_2, ssq2);
        float y2_3 = ((mode != 0) ? _unpack_bf16_10[3] * gw8[3] : _unpack_bf16_10[3]);
        ssq2 = __fmaf_rn(y2_3, y2_3, ssq2);
        float y2_4 = ((mode != 0) ? _unpack_bf16_10[4] * gw8[4] : _unpack_bf16_10[4]);
        ssq2 = __fmaf_rn(y2_4, y2_4, ssq2);
        float y2_5 = ((mode != 0) ? _unpack_bf16_10[5] * gw8[5] : _unpack_bf16_10[5]);
        ssq2 = __fmaf_rn(y2_5, y2_5, ssq2);
        float y2_6 = ((mode != 0) ? _unpack_bf16_10[6] * gw8[6] : _unpack_bf16_10[6]);
        ssq2 = __fmaf_rn(y2_6, y2_6, ssq2);
        float y2_7 = ((mode != 0) ? _unpack_bf16_10[7] * gw8[7] : _unpack_bf16_10[7]);
        ssq2 = __fmaf_rn(y2_7, y2_7, ssq2);
        ssq_k[2] = ssq2;
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[24 + _pair]));
        }
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[28 + _pair]));
        }
        float s8_3[8];
        float t3_0_0 = (float)_unpack_bf16_11[0];
        s8_3[0] = t3_0_0;
        float t3_0_1 = (float)_unpack_bf16_12[0];
        s8_3[0] = s8_3[0] + t3_0_1;
        float t3_1_0 = (float)_unpack_bf16_11[1];
        s8_3[1] = t3_1_0;
        float t3_1_1 = (float)_unpack_bf16_12[1];
        s8_3[1] = s8_3[1] + t3_1_1;
        float t3_2_0 = (float)_unpack_bf16_11[2];
        s8_3[2] = t3_2_0;
        float t3_2_1 = (float)_unpack_bf16_12[2];
        s8_3[2] = s8_3[2] + t3_2_1;
        float t3_3_0 = (float)_unpack_bf16_11[3];
        s8_3[3] = t3_3_0;
        float t3_3_1 = (float)_unpack_bf16_12[3];
        s8_3[3] = s8_3[3] + t3_3_1;
        float t3_4_0 = (float)_unpack_bf16_11[4];
        s8_3[4] = t3_4_0;
        float t3_4_1 = (float)_unpack_bf16_12[4];
        s8_3[4] = s8_3[4] + t3_4_1;
        float t3_5_0 = (float)_unpack_bf16_11[5];
        s8_3[5] = t3_5_0;
        float t3_5_1 = (float)_unpack_bf16_12[5];
        s8_3[5] = s8_3[5] + t3_5_1;
        float t3_6_0 = (float)_unpack_bf16_11[6];
        s8_3[6] = t3_6_0;
        float t3_6_1 = (float)_unpack_bf16_12[6];
        s8_3[6] = s8_3[6] + t3_6_1;
        float t3_7_0 = (float)_unpack_bf16_11[7];
        s8_3[7] = t3_7_0;
        float t3_7_1 = (float)_unpack_bf16_12[7];
        s8_3[7] = s8_3[7] + t3_7_1;
        unsigned int s8_3_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_3[_lp*2 + 0], s8_3[_lp*2+1 + 0]));
            s8_3_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        rw[12] = s8_3_bf16[0];
        rw[13] = s8_3_bf16[1];
        rw[14] = s8_3_bf16[2];
        rw[15] = s8_3_bf16[3];
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                : "r"(s8_3_bf16[_pair]));
        }
        float ssq3 = 0.0f;
        float y3_0 = ((mode != 0) ? _unpack_bf16_13[0] * gw8[0] : _unpack_bf16_13[0]);
        ssq3 = __fmaf_rn(y3_0, y3_0, ssq3);
        float y3_1 = ((mode != 0) ? _unpack_bf16_13[1] * gw8[1] : _unpack_bf16_13[1]);
        ssq3 = __fmaf_rn(y3_1, y3_1, ssq3);
        float y3_2 = ((mode != 0) ? _unpack_bf16_13[2] * gw8[2] : _unpack_bf16_13[2]);
        ssq3 = __fmaf_rn(y3_2, y3_2, ssq3);
        float y3_3 = ((mode != 0) ? _unpack_bf16_13[3] * gw8[3] : _unpack_bf16_13[3]);
        ssq3 = __fmaf_rn(y3_3, y3_3, ssq3);
        float y3_4 = ((mode != 0) ? _unpack_bf16_13[4] * gw8[4] : _unpack_bf16_13[4]);
        ssq3 = __fmaf_rn(y3_4, y3_4, ssq3);
        float y3_5 = ((mode != 0) ? _unpack_bf16_13[5] * gw8[5] : _unpack_bf16_13[5]);
        ssq3 = __fmaf_rn(y3_5, y3_5, ssq3);
        float y3_6 = ((mode != 0) ? _unpack_bf16_13[6] * gw8[6] : _unpack_bf16_13[6]);
        ssq3 = __fmaf_rn(y3_6, y3_6, ssq3);
        float y3_7 = ((mode != 0) ? _unpack_bf16_13[7] * gw8[7] : _unpack_bf16_13[7]);
        ssq3 = __fmaf_rn(y3_7, y3_7, ssq3);
        ssq_k[3] = ssq3;
        float _warp_reduce_0 = ssq_k[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        if (lane == 0) {
            warp_tab[warp * 4] = _warp_reduce_0;
        }
        float _warp_reduce_1 = ssq_k[1];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        if (lane == 0) {
            warp_tab[warp * 4 + 1] = _warp_reduce_1;
        }
        float _warp_reduce_2 = ssq_k[2];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        if (lane == 0) {
            warp_tab[warp * 4 + 2] = _warp_reduce_2;
        }
        float _warp_reduce_3 = ssq_k[3];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        if (lane == 0) {
            warp_tab[warp * 4 + 3] = _warp_reduce_3;
        }
        __syncthreads();
        float tot_k[4];
        tot_k[0] = warp_tab[0];
        tot_k[0] = tot_k[0] + warp_tab[4];
        tot_k[0] = tot_k[0] + warp_tab[8];
        tot_k[0] = tot_k[0] + warp_tab[12];
        tot_k[0] = tot_k[0] + warp_tab[16];
        tot_k[0] = tot_k[0] + warp_tab[20];
        tot_k[0] = tot_k[0] + warp_tab[24];
        tot_k[0] = tot_k[0] + warp_tab[28];
        tot_k[0] = tot_k[0] + warp_tab[32];
        tot_k[0] = tot_k[0] + warp_tab[36];
        tot_k[0] = tot_k[0] + warp_tab[40];
        tot_k[0] = tot_k[0] + warp_tab[44];
        tot_k[0] = tot_k[0] + warp_tab[48];
        tot_k[0] = tot_k[0] + warp_tab[52];
        tot_k[0] = tot_k[0] + warp_tab[56];
        tot_k[0] = tot_k[0] + warp_tab[60];
        tot_k[1] = warp_tab[1];
        tot_k[1] = tot_k[1] + warp_tab[5];
        tot_k[1] = tot_k[1] + warp_tab[9];
        tot_k[1] = tot_k[1] + warp_tab[13];
        tot_k[1] = tot_k[1] + warp_tab[17];
        tot_k[1] = tot_k[1] + warp_tab[21];
        tot_k[1] = tot_k[1] + warp_tab[25];
        tot_k[1] = tot_k[1] + warp_tab[29];
        tot_k[1] = tot_k[1] + warp_tab[33];
        tot_k[1] = tot_k[1] + warp_tab[37];
        tot_k[1] = tot_k[1] + warp_tab[41];
        tot_k[1] = tot_k[1] + warp_tab[45];
        tot_k[1] = tot_k[1] + warp_tab[49];
        tot_k[1] = tot_k[1] + warp_tab[53];
        tot_k[1] = tot_k[1] + warp_tab[57];
        tot_k[1] = tot_k[1] + warp_tab[61];
        tot_k[2] = warp_tab[2];
        tot_k[2] = tot_k[2] + warp_tab[6];
        tot_k[2] = tot_k[2] + warp_tab[10];
        tot_k[2] = tot_k[2] + warp_tab[14];
        tot_k[2] = tot_k[2] + warp_tab[18];
        tot_k[2] = tot_k[2] + warp_tab[22];
        tot_k[2] = tot_k[2] + warp_tab[26];
        tot_k[2] = tot_k[2] + warp_tab[30];
        tot_k[2] = tot_k[2] + warp_tab[34];
        tot_k[2] = tot_k[2] + warp_tab[38];
        tot_k[2] = tot_k[2] + warp_tab[42];
        tot_k[2] = tot_k[2] + warp_tab[46];
        tot_k[2] = tot_k[2] + warp_tab[50];
        tot_k[2] = tot_k[2] + warp_tab[54];
        tot_k[2] = tot_k[2] + warp_tab[58];
        tot_k[2] = tot_k[2] + warp_tab[62];
        tot_k[3] = warp_tab[3];
        tot_k[3] = tot_k[3] + warp_tab[7];
        tot_k[3] = tot_k[3] + warp_tab[11];
        tot_k[3] = tot_k[3] + warp_tab[15];
        tot_k[3] = tot_k[3] + warp_tab[19];
        tot_k[3] = tot_k[3] + warp_tab[23];
        tot_k[3] = tot_k[3] + warp_tab[27];
        tot_k[3] = tot_k[3] + warp_tab[31];
        tot_k[3] = tot_k[3] + warp_tab[35];
        tot_k[3] = tot_k[3] + warp_tab[39];
        tot_k[3] = tot_k[3] + warp_tab[43];
        tot_k[3] = tot_k[3] + warp_tab[47];
        tot_k[3] = tot_k[3] + warp_tab[51];
        tot_k[3] = tot_k[3] + warp_tab[55];
        tot_k[3] = tot_k[3] + warp_tab[59];
        tot_k[3] = tot_k[3] + warp_tab[63];
        float pp[8];
        float acc[8];
        float _sqrt_39;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_39) : "f"(__fmaf_rn(tot_k[0], 0.00024414062f, eps)));
        float rs0 = __frcp_rn(_sqrt_39);
        pp[0] = 0.0f;
        pp[1] = 0.0f;
        pp[2] = 0.0f;
        pp[3] = 0.0f;
        pp[4] = 0.0f;
        pp[5] = 0.0f;
        pp[6] = 0.0f;
        pp[7] = 0.0f;
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                : "r"(rw[_pair]));
        }
        float yn0_0 = ((mode != 0) ? _unpack_bf16_14[0] * gw8[0] : _unpack_bf16_14[0]);
        float o0_0 = yn0_0 * rs0;
        float o2_0_0 = ((mode != 0) ? o0_0 : o0_0 * gw8[0]);
        pp[0] = __fmaf_rn(o2_0_0, wk[0], pp[0]);
        float yn0_1 = ((mode != 0) ? _unpack_bf16_14[1] * gw8[1] : _unpack_bf16_14[1]);
        float o0_1 = yn0_1 * rs0;
        float o2_0_1 = ((mode != 0) ? o0_1 : o0_1 * gw8[1]);
        pp[1] = __fmaf_rn(o2_0_1, wk[0], pp[1]);
        float yn0_2 = ((mode != 0) ? _unpack_bf16_14[2] * gw8[2] : _unpack_bf16_14[2]);
        float o0_2 = yn0_2 * rs0;
        float o2_0_2 = ((mode != 0) ? o0_2 : o0_2 * gw8[2]);
        pp[2] = __fmaf_rn(o2_0_2, wk[0], pp[2]);
        float yn0_3 = ((mode != 0) ? _unpack_bf16_14[3] * gw8[3] : _unpack_bf16_14[3]);
        float o0_3 = yn0_3 * rs0;
        float o2_0_3 = ((mode != 0) ? o0_3 : o0_3 * gw8[3]);
        pp[3] = __fmaf_rn(o2_0_3, wk[0], pp[3]);
        float yn0_4 = ((mode != 0) ? _unpack_bf16_14[4] * gw8[4] : _unpack_bf16_14[4]);
        float o0_4 = yn0_4 * rs0;
        float o2_0_4 = ((mode != 0) ? o0_4 : o0_4 * gw8[4]);
        pp[4] = __fmaf_rn(o2_0_4, wk[0], pp[4]);
        float yn0_5 = ((mode != 0) ? _unpack_bf16_14[5] * gw8[5] : _unpack_bf16_14[5]);
        float o0_5 = yn0_5 * rs0;
        float o2_0_5 = ((mode != 0) ? o0_5 : o0_5 * gw8[5]);
        pp[5] = __fmaf_rn(o2_0_5, wk[0], pp[5]);
        float yn0_6 = ((mode != 0) ? _unpack_bf16_14[6] * gw8[6] : _unpack_bf16_14[6]);
        float o0_6 = yn0_6 * rs0;
        float o2_0_6 = ((mode != 0) ? o0_6 : o0_6 * gw8[6]);
        pp[6] = __fmaf_rn(o2_0_6, wk[0], pp[6]);
        float yn0_7 = ((mode != 0) ? _unpack_bf16_14[7] * gw8[7] : _unpack_bf16_14[7]);
        float o0_7 = yn0_7 * rs0;
        float o2_0_7 = ((mode != 0) ? o0_7 : o0_7 * gw8[7]);
        pp[7] = __fmaf_rn(o2_0_7, wk[0], pp[7]);
        float _sqrt_40;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_40) : "f"(__fmaf_rn(tot_k[1], 0.00024414062f, eps)));
        float rs1 = __frcp_rn(_sqrt_40);
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                : "r"(rw[4 + _pair]));
        }
        float yn1_0 = ((mode != 0) ? _unpack_bf16_15[0] * gw8[0] : _unpack_bf16_15[0]);
        float o1_0 = yn1_0 * rs1;
        float o2_1_0 = ((mode != 0) ? o1_0 : o1_0 * gw8[0]);
        pp[0] = __fmaf_rn(o2_1_0, wk[1], pp[0]);
        float yn1_1 = ((mode != 0) ? _unpack_bf16_15[1] * gw8[1] : _unpack_bf16_15[1]);
        float o1_1 = yn1_1 * rs1;
        float o2_1_1 = ((mode != 0) ? o1_1 : o1_1 * gw8[1]);
        pp[1] = __fmaf_rn(o2_1_1, wk[1], pp[1]);
        float yn1_2 = ((mode != 0) ? _unpack_bf16_15[2] * gw8[2] : _unpack_bf16_15[2]);
        float o1_2 = yn1_2 * rs1;
        float o2_1_2 = ((mode != 0) ? o1_2 : o1_2 * gw8[2]);
        pp[2] = __fmaf_rn(o2_1_2, wk[1], pp[2]);
        float yn1_3 = ((mode != 0) ? _unpack_bf16_15[3] * gw8[3] : _unpack_bf16_15[3]);
        float o1_3 = yn1_3 * rs1;
        float o2_1_3 = ((mode != 0) ? o1_3 : o1_3 * gw8[3]);
        pp[3] = __fmaf_rn(o2_1_3, wk[1], pp[3]);
        float yn1_4 = ((mode != 0) ? _unpack_bf16_15[4] * gw8[4] : _unpack_bf16_15[4]);
        float o1_4 = yn1_4 * rs1;
        float o2_1_4 = ((mode != 0) ? o1_4 : o1_4 * gw8[4]);
        pp[4] = __fmaf_rn(o2_1_4, wk[1], pp[4]);
        float yn1_5 = ((mode != 0) ? _unpack_bf16_15[5] * gw8[5] : _unpack_bf16_15[5]);
        float o1_5 = yn1_5 * rs1;
        float o2_1_5 = ((mode != 0) ? o1_5 : o1_5 * gw8[5]);
        pp[5] = __fmaf_rn(o2_1_5, wk[1], pp[5]);
        float yn1_6 = ((mode != 0) ? _unpack_bf16_15[6] * gw8[6] : _unpack_bf16_15[6]);
        float o1_6 = yn1_6 * rs1;
        float o2_1_6 = ((mode != 0) ? o1_6 : o1_6 * gw8[6]);
        pp[6] = __fmaf_rn(o2_1_6, wk[1], pp[6]);
        float yn1_7 = ((mode != 0) ? _unpack_bf16_15[7] * gw8[7] : _unpack_bf16_15[7]);
        float o1_7 = yn1_7 * rs1;
        float o2_1_7 = ((mode != 0) ? o1_7 : o1_7 * gw8[7]);
        pp[7] = __fmaf_rn(o2_1_7, wk[1], pp[7]);
        float _sqrt_41;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_41) : "f"(__fmaf_rn(tot_k[2], 0.00024414062f, eps)));
        float rs2 = __frcp_rn(_sqrt_41);
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(rw[8 + _pair]));
        }
        float yn2_0 = ((mode != 0) ? _unpack_bf16_16[0] * gw8[0] : _unpack_bf16_16[0]);
        float o2_0 = yn2_0 * rs2;
        float o2_2_0 = ((mode != 0) ? o2_0 : o2_0 * gw8[0]);
        pp[0] = __fmaf_rn(o2_2_0, wk[2], pp[0]);
        float yn2_1 = ((mode != 0) ? _unpack_bf16_16[1] * gw8[1] : _unpack_bf16_16[1]);
        float o2_1 = yn2_1 * rs2;
        float o2_2_1 = ((mode != 0) ? o2_1 : o2_1 * gw8[1]);
        pp[1] = __fmaf_rn(o2_2_1, wk[2], pp[1]);
        float yn2_2 = ((mode != 0) ? _unpack_bf16_16[2] * gw8[2] : _unpack_bf16_16[2]);
        float o2_2 = yn2_2 * rs2;
        float o2_2_2 = ((mode != 0) ? o2_2 : o2_2 * gw8[2]);
        pp[2] = __fmaf_rn(o2_2_2, wk[2], pp[2]);
        float yn2_3 = ((mode != 0) ? _unpack_bf16_16[3] * gw8[3] : _unpack_bf16_16[3]);
        float o2_3 = yn2_3 * rs2;
        float o2_2_3 = ((mode != 0) ? o2_3 : o2_3 * gw8[3]);
        pp[3] = __fmaf_rn(o2_2_3, wk[2], pp[3]);
        float yn2_4 = ((mode != 0) ? _unpack_bf16_16[4] * gw8[4] : _unpack_bf16_16[4]);
        float o2_4 = yn2_4 * rs2;
        float o2_2_4 = ((mode != 0) ? o2_4 : o2_4 * gw8[4]);
        pp[4] = __fmaf_rn(o2_2_4, wk[2], pp[4]);
        float yn2_5 = ((mode != 0) ? _unpack_bf16_16[5] * gw8[5] : _unpack_bf16_16[5]);
        float o2_5 = yn2_5 * rs2;
        float o2_2_5 = ((mode != 0) ? o2_5 : o2_5 * gw8[5]);
        pp[5] = __fmaf_rn(o2_2_5, wk[2], pp[5]);
        float yn2_6 = ((mode != 0) ? _unpack_bf16_16[6] * gw8[6] : _unpack_bf16_16[6]);
        float o2_6 = yn2_6 * rs2;
        float o2_2_6 = ((mode != 0) ? o2_6 : o2_6 * gw8[6]);
        pp[6] = __fmaf_rn(o2_2_6, wk[2], pp[6]);
        float yn2_7 = ((mode != 0) ? _unpack_bf16_16[7] * gw8[7] : _unpack_bf16_16[7]);
        float o2_7 = yn2_7 * rs2;
        float o2_2_7 = ((mode != 0) ? o2_7 : o2_7 * gw8[7]);
        pp[7] = __fmaf_rn(o2_2_7, wk[2], pp[7]);
        float _sqrt_42;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_42) : "f"(__fmaf_rn(tot_k[3], 0.00024414062f, eps)));
        float rs3 = __frcp_rn(_sqrt_42);
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(rw[12 + _pair]));
        }
        float yn3_0 = ((mode != 0) ? _unpack_bf16_17[0] * gw8[0] : _unpack_bf16_17[0]);
        float o3_0 = yn3_0 * rs3;
        float o2_3_0 = ((mode != 0) ? o3_0 : o3_0 * gw8[0]);
        pp[0] = __fmaf_rn(o2_3_0, wk[3], pp[0]);
        float yn3_1 = ((mode != 0) ? _unpack_bf16_17[1] * gw8[1] : _unpack_bf16_17[1]);
        float o3_1 = yn3_1 * rs3;
        float o2_3_1 = ((mode != 0) ? o3_1 : o3_1 * gw8[1]);
        pp[1] = __fmaf_rn(o2_3_1, wk[3], pp[1]);
        float yn3_2 = ((mode != 0) ? _unpack_bf16_17[2] * gw8[2] : _unpack_bf16_17[2]);
        float o3_2 = yn3_2 * rs3;
        float o2_3_2 = ((mode != 0) ? o3_2 : o3_2 * gw8[2]);
        pp[2] = __fmaf_rn(o2_3_2, wk[3], pp[2]);
        float yn3_3 = ((mode != 0) ? _unpack_bf16_17[3] * gw8[3] : _unpack_bf16_17[3]);
        float o3_3 = yn3_3 * rs3;
        float o2_3_3 = ((mode != 0) ? o3_3 : o3_3 * gw8[3]);
        pp[3] = __fmaf_rn(o2_3_3, wk[3], pp[3]);
        float yn3_4 = ((mode != 0) ? _unpack_bf16_17[4] * gw8[4] : _unpack_bf16_17[4]);
        float o3_4 = yn3_4 * rs3;
        float o2_3_4 = ((mode != 0) ? o3_4 : o3_4 * gw8[4]);
        pp[4] = __fmaf_rn(o2_3_4, wk[3], pp[4]);
        float yn3_5 = ((mode != 0) ? _unpack_bf16_17[5] * gw8[5] : _unpack_bf16_17[5]);
        float o3_5 = yn3_5 * rs3;
        float o2_3_5 = ((mode != 0) ? o3_5 : o3_5 * gw8[5]);
        pp[5] = __fmaf_rn(o2_3_5, wk[3], pp[5]);
        float yn3_6 = ((mode != 0) ? _unpack_bf16_17[6] * gw8[6] : _unpack_bf16_17[6]);
        float o3_6 = yn3_6 * rs3;
        float o2_3_6 = ((mode != 0) ? o3_6 : o3_6 * gw8[6]);
        pp[6] = __fmaf_rn(o2_3_6, wk[3], pp[6]);
        float yn3_7 = ((mode != 0) ? _unpack_bf16_17[7] * gw8[7] : _unpack_bf16_17[7]);
        float o3_7 = yn3_7 * rs3;
        float o2_3_7 = ((mode != 0) ? o3_7 : o3_7 * gw8[7]);
        pp[7] = __fmaf_rn(o2_3_7, wk[3], pp[7]);
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
            unsigned int _atomic_fetch_add_43 = atomicAdd(&ctr[g * 8 + band], 1);
            flag[0] = _atomic_fetch_add_43;
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
                uint32_t _p16s_carrier_48 = static_cast<uint32_t>(out8_bf16[_p16s_word]);
                uint32_t _p16s_lo_48 = _p16s_carrier_48 & 0x0000ffffu;
                uint32_t _p16s_hi_48 = (_p16s_carrier_48 >> 16) & 0x0000ffffu;
                if (_p16s_lo_48 == 0x00008000u) _p16s_lo_48 = 0u;
                if (_p16s_hi_48 == 0x00008000u) _p16s_hi_48 = 0u;
                _packed16_sanitize_8[_p16s_word] = _p16s_lo_48 | (_p16s_hi_48 << 16);
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
    if (tid == 0 && cta_rank_1 == 0) {
        *(reinterpret_cast<int*>(gen + gidx) + (0)) = gn_next;
    }
}

} // extern "C"
