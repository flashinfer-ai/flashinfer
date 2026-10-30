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
#define SMEM_WARP_TAB_STAGE_BYTES 1024
#define SMEM_WARP_TAB_STRIDE 1024
#define SMEM_TOTAL 1024
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_tp_moe_finalize_rmsnorm_c7ca12dd0cda04ce776f(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int max_ctas, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, unsigned long long c_mcast, int* __restrict__ ctl)
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

    // === Task calls (declaration order) ===
    int b = bid;
    int grid = gridDim.x;
    int col = tid * 8;
    long long col64 = (long long)col;
    int s0 = ctl[b];
    int n_mine = ((b < n_tok) ? (n_tok - b + grid - 1) / grid : 0);
    int steps = n_mine + 6;
    long long push_gen_elems = (long long)max_ctas * 262144;
    long long push_cta = (long long)b * 262144 + col64;
    long long bcast_gen_elems = (long long)max_ctas * 4096;
    long long bcast_cta = (long long)b * 4096 + col64;
    long long rank64 = (long long)rank;
    long long pa_0 = peer_table[0];
    long long pa_1 = peer_table[1];
    long long pa_2 = peer_table[2];
    long long pa_3 = peer_table[3];
    long long pa_4 = peer_table[4];
    long long pa_5 = peer_table[5];
    long long pa_6 = peer_table[6];
    long long pa_7 = peer_table[7];
    __nv_bfloat16* pp_0 = reinterpret_cast<__nv_bfloat16*>(pa_0);
    __nv_bfloat16* pp_1 = reinterpret_cast<__nv_bfloat16*>(pa_1);
    __nv_bfloat16* pp_2 = reinterpret_cast<__nv_bfloat16*>(pa_2);
    __nv_bfloat16* pp_3 = reinterpret_cast<__nv_bfloat16*>(pa_3);
    __nv_bfloat16* pp_4 = reinterpret_cast<__nv_bfloat16*>(pa_4);
    __nv_bfloat16* pp_5 = reinterpret_cast<__nv_bfloat16*>(pa_5);
    __nv_bfloat16* pp_6 = reinterpret_cast<__nv_bfloat16*>(pa_6);
    __nv_bfloat16* pp_7 = reinterpret_cast<__nv_bfloat16*>(pa_7);
    long long ba_0 = peer_table[8];
    long long ba_1 = peer_table[9];
    long long ba_2 = peer_table[10];
    long long ba_3 = peer_table[11];
    long long ba_4 = peer_table[12];
    long long ba_5 = peer_table[13];
    long long ba_6 = peer_table[14];
    long long ba_7 = peer_table[15];
    __nv_bfloat16* bp_0 = reinterpret_cast<__nv_bfloat16*>(ba_0);
    __nv_bfloat16* bp_1 = reinterpret_cast<__nv_bfloat16*>(ba_1);
    __nv_bfloat16* bp_2 = reinterpret_cast<__nv_bfloat16*>(ba_2);
    __nv_bfloat16* bp_3 = reinterpret_cast<__nv_bfloat16*>(ba_3);
    __nv_bfloat16* bp_4 = reinterpret_cast<__nv_bfloat16*>(ba_4);
    __nv_bfloat16* bp_5 = reinterpret_cast<__nv_bfloat16*>(ba_5);
    __nv_bfloat16* bp_6 = reinterpret_cast<__nv_bfloat16*>(ba_6);
    __nv_bfloat16* bp_7 = reinterpret_cast<__nv_bfloat16*>(ba_7);
    long long push_local_addr = peer_table[rank];
    __nv_bfloat16* push_local = reinterpret_cast<__nv_bfloat16*>(push_local_addr);
    long long bcast_local_addr = peer_table[8 + rank];
    __nv_bfloat16* bcast_local = reinterpret_cast<__nv_bfloat16*>(bcast_local_addr);
    unsigned int _vec_load_0[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(norm_weight + col + 0);
        _vec_load_0[0 + 0] = _uv4_0.x;
        _vec_load_0[0 + 1] = _uv4_0.y;
        _vec_load_0[0 + 2] = _uv4_0.z;
        _vec_load_0[0 + 3] = _uv4_0.w;
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
    #pragma unroll 1
    for (int step = 0; step < steps; step++) {
        int s = s0 + step;
        if (n_mine > step) {
            int tok = b + step * grid;
            int owner = b + step & 7;
            int gb = s >> 3 & 1;
            long long dst = push_gen_elems * (long long)gb + push_cta + rank64 * 4096;
            int idx8[8];
            {
                const int4* _ivptr_1 = reinterpret_cast<const int4*>(idx + (tok * 8) + 0);
                int4 _ivld_1;
                _ivld_1 = *_ivptr_1;
                idx8[0 + 0] = _ivld_1.x;
                idx8[0 + 1] = _ivld_1.y;
                idx8[0 + 2] = _ivld_1.z;
                idx8[0 + 3] = _ivld_1.w;
            }
            {
                const int4* _ivptr_2 = reinterpret_cast<const int4*>(idx + (tok * 8 + 4) + 0);
                int4 _ivld_2;
                _ivld_2 = *_ivptr_2;
                idx8[4 + 0] = _ivld_2.x;
                idx8[4 + 1] = _ivld_2.y;
                idx8[4 + 2] = _ivld_2.z;
                idx8[4 + 3] = _ivld_2.w;
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
                    uint4 _uv4_3 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx0 * (long long)row_stride + col64) + 0);
                    packets[0 + 0] = _uv4_3.x;
                    packets[0 + 1] = _uv4_3.y;
                    packets[0 + 2] = _uv4_3.z;
                    packets[0 + 3] = _uv4_3.w;
                }
            }
            int ridx1_0 = idx8[1];
            if (ridx1_0 >= 0 && ridx1_0 < rows) {
                {
                    uint4 _uv4_4 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_0 * (long long)row_stride + col64) + 0);
                    packets[4 + 0] = _uv4_4.x;
                    packets[4 + 1] = _uv4_4.y;
                    packets[4 + 2] = _uv4_4.z;
                    packets[4 + 3] = _uv4_4.w;
                }
            }
            int ridx1_1 = idx8[2];
            if (ridx1_1 >= 0 && ridx1_1 < rows) {
                {
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_1 * (long long)row_stride + col64) + 0);
                    packets[8 + 0] = _uv4_5.x;
                    packets[8 + 1] = _uv4_5.y;
                    packets[8 + 2] = _uv4_5.z;
                    packets[8 + 3] = _uv4_5.w;
                }
            }
            int ridx1_2 = idx8[3];
            if (ridx1_2 >= 0 && ridx1_2 < rows) {
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_2 * (long long)row_stride + col64) + 0);
                    packets[12 + 0] = _uv4_6.x;
                    packets[12 + 1] = _uv4_6.y;
                    packets[12 + 2] = _uv4_6.z;
                    packets[12 + 3] = _uv4_6.w;
                }
            }
            int ridx1_3 = idx8[4];
            if (ridx1_3 >= 0 && ridx1_3 < rows) {
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_3 * (long long)row_stride + col64) + 0);
                    packets[16 + 0] = _uv4_7.x;
                    packets[16 + 1] = _uv4_7.y;
                    packets[16 + 2] = _uv4_7.z;
                    packets[16 + 3] = _uv4_7.w;
                }
            }
            int ridx1_4 = idx8[5];
            if (ridx1_4 >= 0 && ridx1_4 < rows) {
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_4 * (long long)row_stride + col64) + 0);
                    packets[20 + 0] = _uv4_8.x;
                    packets[20 + 1] = _uv4_8.y;
                    packets[20 + 2] = _uv4_8.z;
                    packets[20 + 3] = _uv4_8.w;
                }
            }
            int ridx1_5 = idx8[6];
            if (ridx1_5 >= 0 && ridx1_5 < rows) {
                {
                    uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_5 * (long long)row_stride + col64) + 0);
                    packets[24 + 0] = _uv4_9.x;
                    packets[24 + 1] = _uv4_9.y;
                    packets[24 + 2] = _uv4_9.z;
                    packets[24 + 3] = _uv4_9.w;
                }
            }
            int ridx1_6 = idx8[7];
            if (ridx1_6 >= 0 && ridx1_6 < rows) {
                {
                    uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1_6 * (long long)row_stride + col64) + 0);
                    packets[28 + 0] = _uv4_10.x;
                    packets[28 + 1] = _uv4_10.y;
                    packets[28 + 2] = _uv4_10.z;
                    packets[28 + 3] = _uv4_10.w;
                }
            }
            unsigned int raw0_1_0[4];
            raw0_1_0[0] = packets[0];
            raw0_1_0[1] = packets[1];
            raw0_1_0[2] = packets[2];
            raw0_1_0[3] = packets[3];
            uint32_t _packed16_sanitize_0[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_11 = static_cast<uint32_t>(raw0_1_0[_p16s_word]);
                uint32_t _p16s_lo_11 = _p16s_carrier_11 & 0x0000ffffu;
                uint32_t _p16s_hi_11 = (_p16s_carrier_11 >> 16) & 0x0000ffffu;
                if (_p16s_lo_11 == 0x00008000u) _p16s_lo_11 = 0u;
                if (_p16s_hi_11 == 0x00008000u) _p16s_hi_11 = 0u;
                _packed16_sanitize_0[_p16s_word] = _p16s_lo_11 | (_p16s_hi_11 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            unsigned int raw0_1_1[4];
            raw0_1_1[0] = packets[4];
            raw0_1_1[1] = packets[5];
            raw0_1_1[2] = packets[6];
            raw0_1_1[3] = packets[7];
            uint32_t _packed16_sanitize_1[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_12 = static_cast<uint32_t>(raw0_1_1[_p16s_word]);
                uint32_t _p16s_lo_12 = _p16s_carrier_12 & 0x0000ffffu;
                uint32_t _p16s_hi_12 = (_p16s_carrier_12 >> 16) & 0x0000ffffu;
                if (_p16s_lo_12 == 0x00008000u) _p16s_lo_12 = 0u;
                if (_p16s_hi_12 == 0x00008000u) _p16s_hi_12 = 0u;
                _packed16_sanitize_1[_p16s_word] = _p16s_lo_12 | (_p16s_hi_12 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 32768)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            unsigned int raw0_1_2[4];
            raw0_1_2[0] = packets[8];
            raw0_1_2[1] = packets[9];
            raw0_1_2[2] = packets[10];
            raw0_1_2[3] = packets[11];
            uint32_t _packed16_sanitize_2[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_13 = static_cast<uint32_t>(raw0_1_2[_p16s_word]);
                uint32_t _p16s_lo_13 = _p16s_carrier_13 & 0x0000ffffu;
                uint32_t _p16s_hi_13 = (_p16s_carrier_13 >> 16) & 0x0000ffffu;
                if (_p16s_lo_13 == 0x00008000u) _p16s_lo_13 = 0u;
                if (_p16s_hi_13 == 0x00008000u) _p16s_hi_13 = 0u;
                _packed16_sanitize_2[_p16s_word] = _p16s_lo_13 | (_p16s_hi_13 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 65536)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            unsigned int raw0_1_3[4];
            raw0_1_3[0] = packets[12];
            raw0_1_3[1] = packets[13];
            raw0_1_3[2] = packets[14];
            raw0_1_3[3] = packets[15];
            uint32_t _packed16_sanitize_3[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_14 = static_cast<uint32_t>(raw0_1_3[_p16s_word]);
                uint32_t _p16s_lo_14 = _p16s_carrier_14 & 0x0000ffffu;
                uint32_t _p16s_hi_14 = (_p16s_carrier_14 >> 16) & 0x0000ffffu;
                if (_p16s_lo_14 == 0x00008000u) _p16s_lo_14 = 0u;
                if (_p16s_hi_14 == 0x00008000u) _p16s_hi_14 = 0u;
                _packed16_sanitize_3[_p16s_word] = _p16s_lo_14 | (_p16s_hi_14 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 98304)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            unsigned int raw0_1_4[4];
            raw0_1_4[0] = packets[16];
            raw0_1_4[1] = packets[17];
            raw0_1_4[2] = packets[18];
            raw0_1_4[3] = packets[19];
            uint32_t _packed16_sanitize_4[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_15 = static_cast<uint32_t>(raw0_1_4[_p16s_word]);
                uint32_t _p16s_lo_15 = _p16s_carrier_15 & 0x0000ffffu;
                uint32_t _p16s_hi_15 = (_p16s_carrier_15 >> 16) & 0x0000ffffu;
                if (_p16s_lo_15 == 0x00008000u) _p16s_lo_15 = 0u;
                if (_p16s_hi_15 == 0x00008000u) _p16s_hi_15 = 0u;
                _packed16_sanitize_4[_p16s_word] = _p16s_lo_15 | (_p16s_hi_15 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 131072)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            unsigned int raw0_1_5[4];
            raw0_1_5[0] = packets[20];
            raw0_1_5[1] = packets[21];
            raw0_1_5[2] = packets[22];
            raw0_1_5[3] = packets[23];
            uint32_t _packed16_sanitize_5[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_16 = static_cast<uint32_t>(raw0_1_5[_p16s_word]);
                uint32_t _p16s_lo_16 = _p16s_carrier_16 & 0x0000ffffu;
                uint32_t _p16s_hi_16 = (_p16s_carrier_16 >> 16) & 0x0000ffffu;
                if (_p16s_lo_16 == 0x00008000u) _p16s_lo_16 = 0u;
                if (_p16s_hi_16 == 0x00008000u) _p16s_hi_16 = 0u;
                _packed16_sanitize_5[_p16s_word] = _p16s_lo_16 | (_p16s_hi_16 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 163840)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            unsigned int raw0_1_6[4];
            raw0_1_6[0] = packets[24];
            raw0_1_6[1] = packets[25];
            raw0_1_6[2] = packets[26];
            raw0_1_6[3] = packets[27];
            uint32_t _packed16_sanitize_6[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_17 = static_cast<uint32_t>(raw0_1_6[_p16s_word]);
                uint32_t _p16s_lo_17 = _p16s_carrier_17 & 0x0000ffffu;
                uint32_t _p16s_hi_17 = (_p16s_carrier_17 >> 16) & 0x0000ffffu;
                if (_p16s_lo_17 == 0x00008000u) _p16s_lo_17 = 0u;
                if (_p16s_hi_17 == 0x00008000u) _p16s_hi_17 = 0u;
                _packed16_sanitize_6[_p16s_word] = _p16s_lo_17 | (_p16s_hi_17 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 196608)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            unsigned int raw0_1_7[4];
            raw0_1_7[0] = packets[28];
            raw0_1_7[1] = packets[29];
            raw0_1_7[2] = packets[30];
            raw0_1_7[3] = packets[31];
            uint32_t _packed16_sanitize_7[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_18 = static_cast<uint32_t>(raw0_1_7[_p16s_word]);
                uint32_t _p16s_lo_18 = _p16s_carrier_18 & 0x0000ffffu;
                uint32_t _p16s_hi_18 = (_p16s_carrier_18 >> 16) & 0x0000ffffu;
                if (_p16s_lo_18 == 0x00008000u) _p16s_lo_18 = 0u;
                if (_p16s_hi_18 == 0x00008000u) _p16s_hi_18 = 0u;
                _packed16_sanitize_7[_p16s_word] = _p16s_lo_18 | (_p16s_hi_18 << 16);
            }
            if (owner == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_0 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_1 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_2 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_3 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 4) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_4 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 5) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_5 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 6) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_6 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 7) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(pp_7 + (dst + 229376)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
        }
        int jr = step - 3;
        if (jr >= 0 && jr < n_mine && (b + jr & 7) == rank) {
            int tok_r = b + jr * grid;
            int sr = s0 + jr;
            int gbr = sr >> 3 & 1;
            long long src = push_gen_elems * (long long)gbr + push_cta;
            int parity = (jr >> 3 & 1) * 128;
            float w8[8];
            if (weights_fp32 != 0) {
                float _vec_load_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(expert_weights + (tok_r * 8)) + 0);
                    _vec_load_1[0 + 0] = _v4.x;
                    _vec_load_1[0 + 1] = _v4.y;
                    _vec_load_1[0 + 2] = _v4.z;
                    _vec_load_1[0 + 3] = _v4.w;
                }
                float _vec_load_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(expert_weights + (tok_r * 8 + 4)) + 0);
                    _vec_load_2[0 + 0] = _v4.x;
                    _vec_load_2[0 + 1] = _v4.y;
                    _vec_load_2[0 + 2] = _v4.z;
                    _vec_load_2[0 + 3] = _v4.w;
                }
                w8[0] = _vec_load_1[0];
                w8[4] = _vec_load_2[0];
                w8[1] = _vec_load_1[1];
                w8[5] = _vec_load_2[1];
                w8[2] = _vec_load_1[2];
                w8[6] = _vec_load_2[2];
                w8[3] = _vec_load_1[3];
                w8[7] = _vec_load_2[3];
            } else {
                unsigned int _vec_load_3[4];
                {
                    uint4 _uv4_21 = *reinterpret_cast<const uint4*>(expert_weights + (tok_r * 4) + 0);
                    _vec_load_3[0 + 0] = _uv4_21.x;
                    _vec_load_3[0 + 1] = _uv4_21.y;
                    _vec_load_3[0 + 2] = _uv4_21.z;
                    _vec_load_3[0 + 3] = _uv4_21.w;
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
                w8[0] = _unpack_bf16_1[0];
                w8[1] = _unpack_bf16_1[1];
                w8[2] = _unpack_bf16_1[2];
                w8[3] = _unpack_bf16_1[3];
                w8[4] = _unpack_bf16_1[4];
                w8[5] = _unpack_bf16_1[5];
                w8[6] = _unpack_bf16_1[6];
                w8[7] = _unpack_bf16_1[7];
            }
            unsigned int rw[32];
            float ssq8[8];
            uint32_t _sysv_poll_group_0[32];
            uint32_t _sysv_min_22;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + src)) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 4096))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 8192))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 12288))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 16384))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 20480))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 24576))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 28672))) : "memory");
                uint32_t _sysv_min_23_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_0) : "r"(_sysv_poll_group_0[0]), "r"(_sysv_poll_group_0[1]));
                uint32_t _sysv_min_23_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_1) : "r"(_sysv_poll_group_0[2]), "r"(_sysv_poll_group_0[3]));
                uint32_t _sysv_min_23_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_2) : "r"(_sysv_min_23_0), "r"(_sysv_min_23_1));
                uint32_t _sysv_min_23_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_3) : "r"(_sysv_poll_group_0[4]), "r"(_sysv_poll_group_0[5]));
                uint32_t _sysv_min_23_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_4) : "r"(_sysv_poll_group_0[6]), "r"(_sysv_poll_group_0[7]));
                uint32_t _sysv_min_23_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_5) : "r"(_sysv_min_23_3), "r"(_sysv_min_23_4));
                uint32_t _sysv_min_23_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_6) : "r"(_sysv_min_23_2), "r"(_sysv_min_23_5));
                uint32_t _sysv_min_23_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_7) : "r"(_sysv_poll_group_0[8]), "r"(_sysv_poll_group_0[9]));
                uint32_t _sysv_min_23_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_8) : "r"(_sysv_poll_group_0[10]), "r"(_sysv_poll_group_0[11]));
                uint32_t _sysv_min_23_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_9) : "r"(_sysv_min_23_7), "r"(_sysv_min_23_8));
                uint32_t _sysv_min_23_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_10) : "r"(_sysv_poll_group_0[12]), "r"(_sysv_poll_group_0[13]));
                uint32_t _sysv_min_23_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_11) : "r"(_sysv_poll_group_0[14]), "r"(_sysv_poll_group_0[15]));
                uint32_t _sysv_min_23_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_12) : "r"(_sysv_min_23_10), "r"(_sysv_min_23_11));
                uint32_t _sysv_min_23_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_13) : "r"(_sysv_min_23_9), "r"(_sysv_min_23_12));
                uint32_t _sysv_min_23_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_14) : "r"(_sysv_min_23_6), "r"(_sysv_min_23_13));
                uint32_t _sysv_min_23_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_15) : "r"(_sysv_poll_group_0[16]), "r"(_sysv_poll_group_0[17]));
                uint32_t _sysv_min_23_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_16) : "r"(_sysv_poll_group_0[18]), "r"(_sysv_poll_group_0[19]));
                uint32_t _sysv_min_23_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_17) : "r"(_sysv_min_23_15), "r"(_sysv_min_23_16));
                uint32_t _sysv_min_23_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_18) : "r"(_sysv_poll_group_0[20]), "r"(_sysv_poll_group_0[21]));
                uint32_t _sysv_min_23_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_19) : "r"(_sysv_poll_group_0[22]), "r"(_sysv_poll_group_0[23]));
                uint32_t _sysv_min_23_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_20) : "r"(_sysv_min_23_18), "r"(_sysv_min_23_19));
                uint32_t _sysv_min_23_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_21) : "r"(_sysv_min_23_17), "r"(_sysv_min_23_20));
                uint32_t _sysv_min_23_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_22) : "r"(_sysv_poll_group_0[24]), "r"(_sysv_poll_group_0[25]));
                uint32_t _sysv_min_23_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_23) : "r"(_sysv_poll_group_0[26]), "r"(_sysv_poll_group_0[27]));
                uint32_t _sysv_min_23_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_24) : "r"(_sysv_min_23_22), "r"(_sysv_min_23_23));
                uint32_t _sysv_min_23_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_25) : "r"(_sysv_poll_group_0[28]), "r"(_sysv_poll_group_0[29]));
                uint32_t _sysv_min_23_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_26) : "r"(_sysv_poll_group_0[30]), "r"(_sysv_poll_group_0[31]));
                uint32_t _sysv_min_23_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_27) : "r"(_sysv_min_23_25), "r"(_sysv_min_23_26));
                uint32_t _sysv_min_23_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_28) : "r"(_sysv_min_23_24), "r"(_sysv_min_23_27));
                uint32_t _sysv_min_23_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_29) : "r"(_sysv_min_23_21), "r"(_sysv_min_23_28));
                uint32_t _sysv_min_23_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_23_30) : "r"(_sysv_min_23_14), "r"(_sysv_min_23_29));
                _sysv_min_22 = _sysv_min_23_30;
            } while (
                ((_sysv_min_22 & 0xffffu) == 0x8000u) || ((_sysv_min_22 >> 16) == 0x8000u)
            );
            float s8_0[8];
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
            s8_0[0] = _unpack_bf16_2[0];
            s8_0[1] = _unpack_bf16_2[1];
            s8_0[2] = _unpack_bf16_2[2];
            s8_0[3] = _unpack_bf16_2[3];
            s8_0[4] = _unpack_bf16_2[4];
            s8_0[5] = _unpack_bf16_2[5];
            s8_0[6] = _unpack_bf16_2[6];
            s8_0[7] = _unpack_bf16_2[7];
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
            s8_0[0] = s8_0[0] + _unpack_bf16_3[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_3[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_3[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_3[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_3[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_3[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_3[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_3[7];
            float _unpack_bf16_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[8 + _pair]));
            }
            s8_0[0] = s8_0[0] + _unpack_bf16_4[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_4[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_4[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_4[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_4[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_4[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_4[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_4[7];
            float _unpack_bf16_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[12 + _pair]));
            }
            s8_0[0] = s8_0[0] + _unpack_bf16_5[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_5[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_5[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_5[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_5[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_5[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_5[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_5[7];
            float _unpack_bf16_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[16 + _pair]));
            }
            s8_0[0] = s8_0[0] + _unpack_bf16_6[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_6[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_6[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_6[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_6[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_6[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_6[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_6[7];
            float _unpack_bf16_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[20 + _pair]));
            }
            s8_0[0] = s8_0[0] + _unpack_bf16_7[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_7[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_7[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_7[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_7[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_7[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_7[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_7[7];
            float _unpack_bf16_8[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[24 + _pair]));
            }
            s8_0[0] = s8_0[0] + _unpack_bf16_8[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_8[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_8[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_8[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_8[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_8[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_8[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_8[7];
            float _unpack_bf16_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[28 + _pair]));
            }
            s8_0[0] = s8_0[0] + _unpack_bf16_9[0];
            s8_0[1] = s8_0[1] + _unpack_bf16_9[1];
            s8_0[2] = s8_0[2] + _unpack_bf16_9[2];
            s8_0[3] = s8_0[3] + _unpack_bf16_9[3];
            s8_0[4] = s8_0[4] + _unpack_bf16_9[4];
            s8_0[5] = s8_0[5] + _unpack_bf16_9[5];
            s8_0[6] = s8_0[6] + _unpack_bf16_9[6];
            s8_0[7] = s8_0[7] + _unpack_bf16_9[7];
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
            float _unpack_bf16_y_0[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_0[_pair * 2])[0]), "=f"((&_unpack_bf16_y_0[_pair * 2])[1])
                    : "r"(s8_bf16_0[_pair]));
            }
            float ssq_0 = 0.0f;
            float y_0 = ((mode != 0) ? _unpack_bf16_y_0[0] * gw8[0] : _unpack_bf16_y_0[0]);
            ssq_0 = __fmaf_rn(y_0, y_0, ssq_0);
            float y_1 = ((mode != 0) ? _unpack_bf16_y_0[1] * gw8[1] : _unpack_bf16_y_0[1]);
            ssq_0 = __fmaf_rn(y_1, y_1, ssq_0);
            float y_2 = ((mode != 0) ? _unpack_bf16_y_0[2] * gw8[2] : _unpack_bf16_y_0[2]);
            ssq_0 = __fmaf_rn(y_2, y_2, ssq_0);
            float y_3 = ((mode != 0) ? _unpack_bf16_y_0[3] * gw8[3] : _unpack_bf16_y_0[3]);
            ssq_0 = __fmaf_rn(y_3, y_3, ssq_0);
            float y_4 = ((mode != 0) ? _unpack_bf16_y_0[4] * gw8[4] : _unpack_bf16_y_0[4]);
            ssq_0 = __fmaf_rn(y_4, y_4, ssq_0);
            float y_5 = ((mode != 0) ? _unpack_bf16_y_0[5] * gw8[5] : _unpack_bf16_y_0[5]);
            ssq_0 = __fmaf_rn(y_5, y_5, ssq_0);
            float y_6 = ((mode != 0) ? _unpack_bf16_y_0[6] * gw8[6] : _unpack_bf16_y_0[6]);
            ssq_0 = __fmaf_rn(y_6, y_6, ssq_0);
            float y_7 = ((mode != 0) ? _unpack_bf16_y_0[7] * gw8[7] : _unpack_bf16_y_0[7]);
            ssq_0 = __fmaf_rn(y_7, y_7, ssq_0);
            ssq8[0] = ssq_0;
            {
                auto _gst_24 = push_local + src;
                __builtin_assume(__isGlobal(_gst_24));
                reinterpret_cast<int4*>(_gst_24)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_25 = push_local + (src + 4096);
                __builtin_assume(__isGlobal(_gst_25));
                reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_26 = push_local + (src + 8192);
                __builtin_assume(__isGlobal(_gst_26));
                reinterpret_cast<int4*>(_gst_26)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_27 = push_local + (src + 12288);
                __builtin_assume(__isGlobal(_gst_27));
                reinterpret_cast<int4*>(_gst_27)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_28 = push_local + (src + 16384);
                __builtin_assume(__isGlobal(_gst_28));
                reinterpret_cast<int4*>(_gst_28)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_29 = push_local + (src + 20480);
                __builtin_assume(__isGlobal(_gst_29));
                reinterpret_cast<int4*>(_gst_29)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_30 = push_local + (src + 24576);
                __builtin_assume(__isGlobal(_gst_30));
                reinterpret_cast<int4*>(_gst_30)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_31 = push_local + (src + 28672);
                __builtin_assume(__isGlobal(_gst_31));
                reinterpret_cast<int4*>(_gst_31)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_0 = ssq8[0];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_0 += __shfl_xor_sync(0xFFFFFFFF, _wrg_0, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8] = _wrg_0;
            }
            uint32_t _sysv_poll_group_1[32];
            uint32_t _sysv_min_32;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 32768))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 36864))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 40960))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 45056))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 49152))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 53248))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 57344))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 61440))) : "memory");
                uint32_t _sysv_min_33_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_0) : "r"(_sysv_poll_group_1[0]), "r"(_sysv_poll_group_1[1]));
                uint32_t _sysv_min_33_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_1) : "r"(_sysv_poll_group_1[2]), "r"(_sysv_poll_group_1[3]));
                uint32_t _sysv_min_33_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_2) : "r"(_sysv_min_33_0), "r"(_sysv_min_33_1));
                uint32_t _sysv_min_33_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_3) : "r"(_sysv_poll_group_1[4]), "r"(_sysv_poll_group_1[5]));
                uint32_t _sysv_min_33_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_4) : "r"(_sysv_poll_group_1[6]), "r"(_sysv_poll_group_1[7]));
                uint32_t _sysv_min_33_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_5) : "r"(_sysv_min_33_3), "r"(_sysv_min_33_4));
                uint32_t _sysv_min_33_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_6) : "r"(_sysv_min_33_2), "r"(_sysv_min_33_5));
                uint32_t _sysv_min_33_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_7) : "r"(_sysv_poll_group_1[8]), "r"(_sysv_poll_group_1[9]));
                uint32_t _sysv_min_33_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_8) : "r"(_sysv_poll_group_1[10]), "r"(_sysv_poll_group_1[11]));
                uint32_t _sysv_min_33_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_9) : "r"(_sysv_min_33_7), "r"(_sysv_min_33_8));
                uint32_t _sysv_min_33_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_10) : "r"(_sysv_poll_group_1[12]), "r"(_sysv_poll_group_1[13]));
                uint32_t _sysv_min_33_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_11) : "r"(_sysv_poll_group_1[14]), "r"(_sysv_poll_group_1[15]));
                uint32_t _sysv_min_33_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_12) : "r"(_sysv_min_33_10), "r"(_sysv_min_33_11));
                uint32_t _sysv_min_33_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_13) : "r"(_sysv_min_33_9), "r"(_sysv_min_33_12));
                uint32_t _sysv_min_33_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_14) : "r"(_sysv_min_33_6), "r"(_sysv_min_33_13));
                uint32_t _sysv_min_33_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_15) : "r"(_sysv_poll_group_1[16]), "r"(_sysv_poll_group_1[17]));
                uint32_t _sysv_min_33_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_16) : "r"(_sysv_poll_group_1[18]), "r"(_sysv_poll_group_1[19]));
                uint32_t _sysv_min_33_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_17) : "r"(_sysv_min_33_15), "r"(_sysv_min_33_16));
                uint32_t _sysv_min_33_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_18) : "r"(_sysv_poll_group_1[20]), "r"(_sysv_poll_group_1[21]));
                uint32_t _sysv_min_33_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_19) : "r"(_sysv_poll_group_1[22]), "r"(_sysv_poll_group_1[23]));
                uint32_t _sysv_min_33_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_20) : "r"(_sysv_min_33_18), "r"(_sysv_min_33_19));
                uint32_t _sysv_min_33_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_21) : "r"(_sysv_min_33_17), "r"(_sysv_min_33_20));
                uint32_t _sysv_min_33_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_22) : "r"(_sysv_poll_group_1[24]), "r"(_sysv_poll_group_1[25]));
                uint32_t _sysv_min_33_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_23) : "r"(_sysv_poll_group_1[26]), "r"(_sysv_poll_group_1[27]));
                uint32_t _sysv_min_33_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_24) : "r"(_sysv_min_33_22), "r"(_sysv_min_33_23));
                uint32_t _sysv_min_33_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_25) : "r"(_sysv_poll_group_1[28]), "r"(_sysv_poll_group_1[29]));
                uint32_t _sysv_min_33_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_26) : "r"(_sysv_poll_group_1[30]), "r"(_sysv_poll_group_1[31]));
                uint32_t _sysv_min_33_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_27) : "r"(_sysv_min_33_25), "r"(_sysv_min_33_26));
                uint32_t _sysv_min_33_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_28) : "r"(_sysv_min_33_24), "r"(_sysv_min_33_27));
                uint32_t _sysv_min_33_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_29) : "r"(_sysv_min_33_21), "r"(_sysv_min_33_28));
                uint32_t _sysv_min_33_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_33_30) : "r"(_sysv_min_33_14), "r"(_sysv_min_33_29));
                _sysv_min_32 = _sysv_min_33_30;
            } while (
                ((_sysv_min_32 & 0xffffu) == 0x8000u) || ((_sysv_min_32 >> 16) == 0x8000u)
            );
            float s8_1[8];
            float _unpack_bf16_10[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[_pair]));
            }
            s8_1[0] = _unpack_bf16_10[0];
            s8_1[1] = _unpack_bf16_10[1];
            s8_1[2] = _unpack_bf16_10[2];
            s8_1[3] = _unpack_bf16_10[3];
            s8_1[4] = _unpack_bf16_10[4];
            s8_1[5] = _unpack_bf16_10[5];
            s8_1[6] = _unpack_bf16_10[6];
            s8_1[7] = _unpack_bf16_10[7];
            float _unpack_bf16_11[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[4 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_11[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_11[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_11[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_11[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_11[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_11[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_11[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_11[7];
            float _unpack_bf16_12[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[8 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_12[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_12[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_12[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_12[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_12[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_12[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_12[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_12[7];
            float _unpack_bf16_13[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[12 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_13[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_13[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_13[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_13[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_13[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_13[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_13[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_13[7];
            float _unpack_bf16_14[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[16 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_14[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_14[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_14[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_14[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_14[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_14[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_14[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_14[7];
            float _unpack_bf16_15[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[20 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_15[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_15[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_15[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_15[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_15[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_15[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_15[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_15[7];
            float _unpack_bf16_16[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[24 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_16[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_16[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_16[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_16[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_16[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_16[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_16[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_16[7];
            float _unpack_bf16_17[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[28 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_17[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_17[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_17[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_17[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_17[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_17[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_17[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_17[7];
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
            float _unpack_bf16_y_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_1[_pair * 2])[0]), "=f"((&_unpack_bf16_y_1[_pair * 2])[1])
                    : "r"(s8_bf16_1[_pair]));
            }
            float ssq_1 = 0.0f;
            float y_8 = ((mode != 0) ? _unpack_bf16_y_1[0] * gw8[0] : _unpack_bf16_y_1[0]);
            ssq_1 = __fmaf_rn(y_8, y_8, ssq_1);
            float y_9 = ((mode != 0) ? _unpack_bf16_y_1[1] * gw8[1] : _unpack_bf16_y_1[1]);
            ssq_1 = __fmaf_rn(y_9, y_9, ssq_1);
            float y_10 = ((mode != 0) ? _unpack_bf16_y_1[2] * gw8[2] : _unpack_bf16_y_1[2]);
            ssq_1 = __fmaf_rn(y_10, y_10, ssq_1);
            float y_11 = ((mode != 0) ? _unpack_bf16_y_1[3] * gw8[3] : _unpack_bf16_y_1[3]);
            ssq_1 = __fmaf_rn(y_11, y_11, ssq_1);
            float y_12 = ((mode != 0) ? _unpack_bf16_y_1[4] * gw8[4] : _unpack_bf16_y_1[4]);
            ssq_1 = __fmaf_rn(y_12, y_12, ssq_1);
            float y_13 = ((mode != 0) ? _unpack_bf16_y_1[5] * gw8[5] : _unpack_bf16_y_1[5]);
            ssq_1 = __fmaf_rn(y_13, y_13, ssq_1);
            float y_14 = ((mode != 0) ? _unpack_bf16_y_1[6] * gw8[6] : _unpack_bf16_y_1[6]);
            ssq_1 = __fmaf_rn(y_14, y_14, ssq_1);
            float y_15 = ((mode != 0) ? _unpack_bf16_y_1[7] * gw8[7] : _unpack_bf16_y_1[7]);
            ssq_1 = __fmaf_rn(y_15, y_15, ssq_1);
            ssq8[1] = ssq_1;
            {
                auto _gst_34 = push_local + (src + 32768);
                __builtin_assume(__isGlobal(_gst_34));
                reinterpret_cast<int4*>(_gst_34)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_35 = push_local + (src + 36864);
                __builtin_assume(__isGlobal(_gst_35));
                reinterpret_cast<int4*>(_gst_35)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_36 = push_local + (src + 40960);
                __builtin_assume(__isGlobal(_gst_36));
                reinterpret_cast<int4*>(_gst_36)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_37 = push_local + (src + 45056);
                __builtin_assume(__isGlobal(_gst_37));
                reinterpret_cast<int4*>(_gst_37)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_38 = push_local + (src + 49152);
                __builtin_assume(__isGlobal(_gst_38));
                reinterpret_cast<int4*>(_gst_38)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_39 = push_local + (src + 53248);
                __builtin_assume(__isGlobal(_gst_39));
                reinterpret_cast<int4*>(_gst_39)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_40 = push_local + (src + 57344);
                __builtin_assume(__isGlobal(_gst_40));
                reinterpret_cast<int4*>(_gst_40)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_41 = push_local + (src + 61440);
                __builtin_assume(__isGlobal(_gst_41));
                reinterpret_cast<int4*>(_gst_41)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_1 = ssq8[1];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_1 += __shfl_xor_sync(0xFFFFFFFF, _wrg_1, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 1] = _wrg_1;
            }
            uint32_t _sysv_poll_group_2[32];
            uint32_t _sysv_min_42;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[0]), "=r"(_sysv_poll_group_2[1]), "=r"(_sysv_poll_group_2[2]), "=r"(_sysv_poll_group_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 65536))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[4]), "=r"(_sysv_poll_group_2[5]), "=r"(_sysv_poll_group_2[6]), "=r"(_sysv_poll_group_2[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 69632))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[8]), "=r"(_sysv_poll_group_2[9]), "=r"(_sysv_poll_group_2[10]), "=r"(_sysv_poll_group_2[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 73728))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[12]), "=r"(_sysv_poll_group_2[13]), "=r"(_sysv_poll_group_2[14]), "=r"(_sysv_poll_group_2[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 77824))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[16]), "=r"(_sysv_poll_group_2[17]), "=r"(_sysv_poll_group_2[18]), "=r"(_sysv_poll_group_2[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 81920))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[20]), "=r"(_sysv_poll_group_2[21]), "=r"(_sysv_poll_group_2[22]), "=r"(_sysv_poll_group_2[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 86016))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[24]), "=r"(_sysv_poll_group_2[25]), "=r"(_sysv_poll_group_2[26]), "=r"(_sysv_poll_group_2[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 90112))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[28]), "=r"(_sysv_poll_group_2[29]), "=r"(_sysv_poll_group_2[30]), "=r"(_sysv_poll_group_2[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 94208))) : "memory");
                uint32_t _sysv_min_43_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_0) : "r"(_sysv_poll_group_2[0]), "r"(_sysv_poll_group_2[1]));
                uint32_t _sysv_min_43_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_1) : "r"(_sysv_poll_group_2[2]), "r"(_sysv_poll_group_2[3]));
                uint32_t _sysv_min_43_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_2) : "r"(_sysv_min_43_0), "r"(_sysv_min_43_1));
                uint32_t _sysv_min_43_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_3) : "r"(_sysv_poll_group_2[4]), "r"(_sysv_poll_group_2[5]));
                uint32_t _sysv_min_43_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_4) : "r"(_sysv_poll_group_2[6]), "r"(_sysv_poll_group_2[7]));
                uint32_t _sysv_min_43_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_5) : "r"(_sysv_min_43_3), "r"(_sysv_min_43_4));
                uint32_t _sysv_min_43_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_6) : "r"(_sysv_min_43_2), "r"(_sysv_min_43_5));
                uint32_t _sysv_min_43_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_7) : "r"(_sysv_poll_group_2[8]), "r"(_sysv_poll_group_2[9]));
                uint32_t _sysv_min_43_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_8) : "r"(_sysv_poll_group_2[10]), "r"(_sysv_poll_group_2[11]));
                uint32_t _sysv_min_43_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_9) : "r"(_sysv_min_43_7), "r"(_sysv_min_43_8));
                uint32_t _sysv_min_43_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_10) : "r"(_sysv_poll_group_2[12]), "r"(_sysv_poll_group_2[13]));
                uint32_t _sysv_min_43_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_11) : "r"(_sysv_poll_group_2[14]), "r"(_sysv_poll_group_2[15]));
                uint32_t _sysv_min_43_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_12) : "r"(_sysv_min_43_10), "r"(_sysv_min_43_11));
                uint32_t _sysv_min_43_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_13) : "r"(_sysv_min_43_9), "r"(_sysv_min_43_12));
                uint32_t _sysv_min_43_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_14) : "r"(_sysv_min_43_6), "r"(_sysv_min_43_13));
                uint32_t _sysv_min_43_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_15) : "r"(_sysv_poll_group_2[16]), "r"(_sysv_poll_group_2[17]));
                uint32_t _sysv_min_43_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_16) : "r"(_sysv_poll_group_2[18]), "r"(_sysv_poll_group_2[19]));
                uint32_t _sysv_min_43_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_17) : "r"(_sysv_min_43_15), "r"(_sysv_min_43_16));
                uint32_t _sysv_min_43_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_18) : "r"(_sysv_poll_group_2[20]), "r"(_sysv_poll_group_2[21]));
                uint32_t _sysv_min_43_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_19) : "r"(_sysv_poll_group_2[22]), "r"(_sysv_poll_group_2[23]));
                uint32_t _sysv_min_43_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_20) : "r"(_sysv_min_43_18), "r"(_sysv_min_43_19));
                uint32_t _sysv_min_43_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_21) : "r"(_sysv_min_43_17), "r"(_sysv_min_43_20));
                uint32_t _sysv_min_43_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_22) : "r"(_sysv_poll_group_2[24]), "r"(_sysv_poll_group_2[25]));
                uint32_t _sysv_min_43_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_23) : "r"(_sysv_poll_group_2[26]), "r"(_sysv_poll_group_2[27]));
                uint32_t _sysv_min_43_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_24) : "r"(_sysv_min_43_22), "r"(_sysv_min_43_23));
                uint32_t _sysv_min_43_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_25) : "r"(_sysv_poll_group_2[28]), "r"(_sysv_poll_group_2[29]));
                uint32_t _sysv_min_43_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_26) : "r"(_sysv_poll_group_2[30]), "r"(_sysv_poll_group_2[31]));
                uint32_t _sysv_min_43_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_27) : "r"(_sysv_min_43_25), "r"(_sysv_min_43_26));
                uint32_t _sysv_min_43_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_28) : "r"(_sysv_min_43_24), "r"(_sysv_min_43_27));
                uint32_t _sysv_min_43_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_29) : "r"(_sysv_min_43_21), "r"(_sysv_min_43_28));
                uint32_t _sysv_min_43_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_43_30) : "r"(_sysv_min_43_14), "r"(_sysv_min_43_29));
                _sysv_min_42 = _sysv_min_43_30;
            } while (
                ((_sysv_min_42 & 0xffffu) == 0x8000u) || ((_sysv_min_42 >> 16) == 0x8000u)
            );
            float s8_2[8];
            float _unpack_bf16_18[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[_pair]));
            }
            s8_2[0] = _unpack_bf16_18[0];
            s8_2[1] = _unpack_bf16_18[1];
            s8_2[2] = _unpack_bf16_18[2];
            s8_2[3] = _unpack_bf16_18[3];
            s8_2[4] = _unpack_bf16_18[4];
            s8_2[5] = _unpack_bf16_18[5];
            s8_2[6] = _unpack_bf16_18[6];
            s8_2[7] = _unpack_bf16_18[7];
            float _unpack_bf16_19[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[4 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_19[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_19[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_19[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_19[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_19[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_19[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_19[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_19[7];
            float _unpack_bf16_20[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[8 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_20[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_20[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_20[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_20[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_20[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_20[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_20[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_20[7];
            float _unpack_bf16_21[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[12 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_21[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_21[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_21[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_21[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_21[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_21[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_21[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_21[7];
            float _unpack_bf16_22[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[16 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_22[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_22[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_22[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_22[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_22[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_22[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_22[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_22[7];
            float _unpack_bf16_23[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[20 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_23[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_23[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_23[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_23[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_23[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_23[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_23[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_23[7];
            float _unpack_bf16_24[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[24 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_24[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_24[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_24[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_24[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_24[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_24[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_24[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_24[7];
            float _unpack_bf16_25[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[28 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_25[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_25[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_25[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_25[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_25[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_25[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_25[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_25[7];
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
            float _unpack_bf16_y_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_2[_pair * 2])[0]), "=f"((&_unpack_bf16_y_2[_pair * 2])[1])
                    : "r"(s8_bf16_2[_pair]));
            }
            float ssq_2 = 0.0f;
            float y_16 = ((mode != 0) ? _unpack_bf16_y_2[0] * gw8[0] : _unpack_bf16_y_2[0]);
            ssq_2 = __fmaf_rn(y_16, y_16, ssq_2);
            float y_17 = ((mode != 0) ? _unpack_bf16_y_2[1] * gw8[1] : _unpack_bf16_y_2[1]);
            ssq_2 = __fmaf_rn(y_17, y_17, ssq_2);
            float y_18 = ((mode != 0) ? _unpack_bf16_y_2[2] * gw8[2] : _unpack_bf16_y_2[2]);
            ssq_2 = __fmaf_rn(y_18, y_18, ssq_2);
            float y_19 = ((mode != 0) ? _unpack_bf16_y_2[3] * gw8[3] : _unpack_bf16_y_2[3]);
            ssq_2 = __fmaf_rn(y_19, y_19, ssq_2);
            float y_20 = ((mode != 0) ? _unpack_bf16_y_2[4] * gw8[4] : _unpack_bf16_y_2[4]);
            ssq_2 = __fmaf_rn(y_20, y_20, ssq_2);
            float y_21 = ((mode != 0) ? _unpack_bf16_y_2[5] * gw8[5] : _unpack_bf16_y_2[5]);
            ssq_2 = __fmaf_rn(y_21, y_21, ssq_2);
            float y_22 = ((mode != 0) ? _unpack_bf16_y_2[6] * gw8[6] : _unpack_bf16_y_2[6]);
            ssq_2 = __fmaf_rn(y_22, y_22, ssq_2);
            float y_23 = ((mode != 0) ? _unpack_bf16_y_2[7] * gw8[7] : _unpack_bf16_y_2[7]);
            ssq_2 = __fmaf_rn(y_23, y_23, ssq_2);
            ssq8[2] = ssq_2;
            {
                auto _gst_44 = push_local + (src + 65536);
                __builtin_assume(__isGlobal(_gst_44));
                reinterpret_cast<int4*>(_gst_44)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_45 = push_local + (src + 69632);
                __builtin_assume(__isGlobal(_gst_45));
                reinterpret_cast<int4*>(_gst_45)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_46 = push_local + (src + 73728);
                __builtin_assume(__isGlobal(_gst_46));
                reinterpret_cast<int4*>(_gst_46)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_47 = push_local + (src + 77824);
                __builtin_assume(__isGlobal(_gst_47));
                reinterpret_cast<int4*>(_gst_47)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_48 = push_local + (src + 81920);
                __builtin_assume(__isGlobal(_gst_48));
                reinterpret_cast<int4*>(_gst_48)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_49 = push_local + (src + 86016);
                __builtin_assume(__isGlobal(_gst_49));
                reinterpret_cast<int4*>(_gst_49)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_50 = push_local + (src + 90112);
                __builtin_assume(__isGlobal(_gst_50));
                reinterpret_cast<int4*>(_gst_50)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_51 = push_local + (src + 94208);
                __builtin_assume(__isGlobal(_gst_51));
                reinterpret_cast<int4*>(_gst_51)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_2 = ssq8[2];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_2 += __shfl_xor_sync(0xFFFFFFFF, _wrg_2, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 2] = _wrg_2;
            }
            uint32_t _sysv_poll_group_3[32];
            uint32_t _sysv_min_52;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[0]), "=r"(_sysv_poll_group_3[1]), "=r"(_sysv_poll_group_3[2]), "=r"(_sysv_poll_group_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 98304))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[4]), "=r"(_sysv_poll_group_3[5]), "=r"(_sysv_poll_group_3[6]), "=r"(_sysv_poll_group_3[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 102400))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[8]), "=r"(_sysv_poll_group_3[9]), "=r"(_sysv_poll_group_3[10]), "=r"(_sysv_poll_group_3[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 106496))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[12]), "=r"(_sysv_poll_group_3[13]), "=r"(_sysv_poll_group_3[14]), "=r"(_sysv_poll_group_3[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 110592))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[16]), "=r"(_sysv_poll_group_3[17]), "=r"(_sysv_poll_group_3[18]), "=r"(_sysv_poll_group_3[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 114688))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[20]), "=r"(_sysv_poll_group_3[21]), "=r"(_sysv_poll_group_3[22]), "=r"(_sysv_poll_group_3[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 118784))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[24]), "=r"(_sysv_poll_group_3[25]), "=r"(_sysv_poll_group_3[26]), "=r"(_sysv_poll_group_3[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 122880))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[28]), "=r"(_sysv_poll_group_3[29]), "=r"(_sysv_poll_group_3[30]), "=r"(_sysv_poll_group_3[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 126976))) : "memory");
                uint32_t _sysv_min_53_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_0) : "r"(_sysv_poll_group_3[0]), "r"(_sysv_poll_group_3[1]));
                uint32_t _sysv_min_53_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_1) : "r"(_sysv_poll_group_3[2]), "r"(_sysv_poll_group_3[3]));
                uint32_t _sysv_min_53_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_2) : "r"(_sysv_min_53_0), "r"(_sysv_min_53_1));
                uint32_t _sysv_min_53_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_3) : "r"(_sysv_poll_group_3[4]), "r"(_sysv_poll_group_3[5]));
                uint32_t _sysv_min_53_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_4) : "r"(_sysv_poll_group_3[6]), "r"(_sysv_poll_group_3[7]));
                uint32_t _sysv_min_53_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_5) : "r"(_sysv_min_53_3), "r"(_sysv_min_53_4));
                uint32_t _sysv_min_53_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_6) : "r"(_sysv_min_53_2), "r"(_sysv_min_53_5));
                uint32_t _sysv_min_53_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_7) : "r"(_sysv_poll_group_3[8]), "r"(_sysv_poll_group_3[9]));
                uint32_t _sysv_min_53_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_8) : "r"(_sysv_poll_group_3[10]), "r"(_sysv_poll_group_3[11]));
                uint32_t _sysv_min_53_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_9) : "r"(_sysv_min_53_7), "r"(_sysv_min_53_8));
                uint32_t _sysv_min_53_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_10) : "r"(_sysv_poll_group_3[12]), "r"(_sysv_poll_group_3[13]));
                uint32_t _sysv_min_53_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_11) : "r"(_sysv_poll_group_3[14]), "r"(_sysv_poll_group_3[15]));
                uint32_t _sysv_min_53_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_12) : "r"(_sysv_min_53_10), "r"(_sysv_min_53_11));
                uint32_t _sysv_min_53_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_13) : "r"(_sysv_min_53_9), "r"(_sysv_min_53_12));
                uint32_t _sysv_min_53_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_14) : "r"(_sysv_min_53_6), "r"(_sysv_min_53_13));
                uint32_t _sysv_min_53_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_15) : "r"(_sysv_poll_group_3[16]), "r"(_sysv_poll_group_3[17]));
                uint32_t _sysv_min_53_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_16) : "r"(_sysv_poll_group_3[18]), "r"(_sysv_poll_group_3[19]));
                uint32_t _sysv_min_53_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_17) : "r"(_sysv_min_53_15), "r"(_sysv_min_53_16));
                uint32_t _sysv_min_53_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_18) : "r"(_sysv_poll_group_3[20]), "r"(_sysv_poll_group_3[21]));
                uint32_t _sysv_min_53_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_19) : "r"(_sysv_poll_group_3[22]), "r"(_sysv_poll_group_3[23]));
                uint32_t _sysv_min_53_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_20) : "r"(_sysv_min_53_18), "r"(_sysv_min_53_19));
                uint32_t _sysv_min_53_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_21) : "r"(_sysv_min_53_17), "r"(_sysv_min_53_20));
                uint32_t _sysv_min_53_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_22) : "r"(_sysv_poll_group_3[24]), "r"(_sysv_poll_group_3[25]));
                uint32_t _sysv_min_53_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_23) : "r"(_sysv_poll_group_3[26]), "r"(_sysv_poll_group_3[27]));
                uint32_t _sysv_min_53_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_24) : "r"(_sysv_min_53_22), "r"(_sysv_min_53_23));
                uint32_t _sysv_min_53_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_25) : "r"(_sysv_poll_group_3[28]), "r"(_sysv_poll_group_3[29]));
                uint32_t _sysv_min_53_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_26) : "r"(_sysv_poll_group_3[30]), "r"(_sysv_poll_group_3[31]));
                uint32_t _sysv_min_53_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_27) : "r"(_sysv_min_53_25), "r"(_sysv_min_53_26));
                uint32_t _sysv_min_53_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_28) : "r"(_sysv_min_53_24), "r"(_sysv_min_53_27));
                uint32_t _sysv_min_53_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_29) : "r"(_sysv_min_53_21), "r"(_sysv_min_53_28));
                uint32_t _sysv_min_53_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_53_30) : "r"(_sysv_min_53_14), "r"(_sysv_min_53_29));
                _sysv_min_52 = _sysv_min_53_30;
            } while (
                ((_sysv_min_52 & 0xffffu) == 0x8000u) || ((_sysv_min_52 >> 16) == 0x8000u)
            );
            float s8_3[8];
            float _unpack_bf16_26[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[_pair]));
            }
            s8_3[0] = _unpack_bf16_26[0];
            s8_3[1] = _unpack_bf16_26[1];
            s8_3[2] = _unpack_bf16_26[2];
            s8_3[3] = _unpack_bf16_26[3];
            s8_3[4] = _unpack_bf16_26[4];
            s8_3[5] = _unpack_bf16_26[5];
            s8_3[6] = _unpack_bf16_26[6];
            s8_3[7] = _unpack_bf16_26[7];
            float _unpack_bf16_27[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[4 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_27[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_27[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_27[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_27[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_27[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_27[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_27[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_27[7];
            float _unpack_bf16_28[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[8 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_28[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_28[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_28[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_28[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_28[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_28[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_28[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_28[7];
            float _unpack_bf16_29[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[12 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_29[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_29[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_29[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_29[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_29[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_29[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_29[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_29[7];
            float _unpack_bf16_30[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[16 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_30[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_30[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_30[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_30[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_30[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_30[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_30[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_30[7];
            float _unpack_bf16_31[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[20 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_31[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_31[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_31[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_31[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_31[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_31[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_31[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_31[7];
            float _unpack_bf16_32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[24 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_32[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_32[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_32[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_32[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_32[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_32[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_32[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_32[7];
            float _unpack_bf16_33[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[28 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_33[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_33[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_33[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_33[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_33[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_33[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_33[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_33[7];
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
            float _unpack_bf16_y_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_3[_pair * 2])[0]), "=f"((&_unpack_bf16_y_3[_pair * 2])[1])
                    : "r"(s8_bf16_3[_pair]));
            }
            float ssq_3 = 0.0f;
            float y_24 = ((mode != 0) ? _unpack_bf16_y_3[0] * gw8[0] : _unpack_bf16_y_3[0]);
            ssq_3 = __fmaf_rn(y_24, y_24, ssq_3);
            float y_25 = ((mode != 0) ? _unpack_bf16_y_3[1] * gw8[1] : _unpack_bf16_y_3[1]);
            ssq_3 = __fmaf_rn(y_25, y_25, ssq_3);
            float y_26 = ((mode != 0) ? _unpack_bf16_y_3[2] * gw8[2] : _unpack_bf16_y_3[2]);
            ssq_3 = __fmaf_rn(y_26, y_26, ssq_3);
            float y_27 = ((mode != 0) ? _unpack_bf16_y_3[3] * gw8[3] : _unpack_bf16_y_3[3]);
            ssq_3 = __fmaf_rn(y_27, y_27, ssq_3);
            float y_28 = ((mode != 0) ? _unpack_bf16_y_3[4] * gw8[4] : _unpack_bf16_y_3[4]);
            ssq_3 = __fmaf_rn(y_28, y_28, ssq_3);
            float y_29 = ((mode != 0) ? _unpack_bf16_y_3[5] * gw8[5] : _unpack_bf16_y_3[5]);
            ssq_3 = __fmaf_rn(y_29, y_29, ssq_3);
            float y_30 = ((mode != 0) ? _unpack_bf16_y_3[6] * gw8[6] : _unpack_bf16_y_3[6]);
            ssq_3 = __fmaf_rn(y_30, y_30, ssq_3);
            float y_31 = ((mode != 0) ? _unpack_bf16_y_3[7] * gw8[7] : _unpack_bf16_y_3[7]);
            ssq_3 = __fmaf_rn(y_31, y_31, ssq_3);
            ssq8[3] = ssq_3;
            {
                auto _gst_54 = push_local + (src + 98304);
                __builtin_assume(__isGlobal(_gst_54));
                reinterpret_cast<int4*>(_gst_54)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_55 = push_local + (src + 102400);
                __builtin_assume(__isGlobal(_gst_55));
                reinterpret_cast<int4*>(_gst_55)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_56 = push_local + (src + 106496);
                __builtin_assume(__isGlobal(_gst_56));
                reinterpret_cast<int4*>(_gst_56)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_57 = push_local + (src + 110592);
                __builtin_assume(__isGlobal(_gst_57));
                reinterpret_cast<int4*>(_gst_57)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_58 = push_local + (src + 114688);
                __builtin_assume(__isGlobal(_gst_58));
                reinterpret_cast<int4*>(_gst_58)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_59 = push_local + (src + 118784);
                __builtin_assume(__isGlobal(_gst_59));
                reinterpret_cast<int4*>(_gst_59)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_60 = push_local + (src + 122880);
                __builtin_assume(__isGlobal(_gst_60));
                reinterpret_cast<int4*>(_gst_60)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_61 = push_local + (src + 126976);
                __builtin_assume(__isGlobal(_gst_61));
                reinterpret_cast<int4*>(_gst_61)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_3 = ssq8[3];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_3 += __shfl_xor_sync(0xFFFFFFFF, _wrg_3, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 3] = _wrg_3;
            }
            uint32_t _sysv_poll_group_4[32];
            uint32_t _sysv_min_62;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[0]), "=r"(_sysv_poll_group_4[1]), "=r"(_sysv_poll_group_4[2]), "=r"(_sysv_poll_group_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 131072))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[4]), "=r"(_sysv_poll_group_4[5]), "=r"(_sysv_poll_group_4[6]), "=r"(_sysv_poll_group_4[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 135168))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[8]), "=r"(_sysv_poll_group_4[9]), "=r"(_sysv_poll_group_4[10]), "=r"(_sysv_poll_group_4[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 139264))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[12]), "=r"(_sysv_poll_group_4[13]), "=r"(_sysv_poll_group_4[14]), "=r"(_sysv_poll_group_4[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 143360))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[16]), "=r"(_sysv_poll_group_4[17]), "=r"(_sysv_poll_group_4[18]), "=r"(_sysv_poll_group_4[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 147456))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[20]), "=r"(_sysv_poll_group_4[21]), "=r"(_sysv_poll_group_4[22]), "=r"(_sysv_poll_group_4[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 151552))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[24]), "=r"(_sysv_poll_group_4[25]), "=r"(_sysv_poll_group_4[26]), "=r"(_sysv_poll_group_4[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 155648))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[28]), "=r"(_sysv_poll_group_4[29]), "=r"(_sysv_poll_group_4[30]), "=r"(_sysv_poll_group_4[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 159744))) : "memory");
                uint32_t _sysv_min_63_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_0) : "r"(_sysv_poll_group_4[0]), "r"(_sysv_poll_group_4[1]));
                uint32_t _sysv_min_63_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_1) : "r"(_sysv_poll_group_4[2]), "r"(_sysv_poll_group_4[3]));
                uint32_t _sysv_min_63_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_2) : "r"(_sysv_min_63_0), "r"(_sysv_min_63_1));
                uint32_t _sysv_min_63_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_3) : "r"(_sysv_poll_group_4[4]), "r"(_sysv_poll_group_4[5]));
                uint32_t _sysv_min_63_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_4) : "r"(_sysv_poll_group_4[6]), "r"(_sysv_poll_group_4[7]));
                uint32_t _sysv_min_63_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_5) : "r"(_sysv_min_63_3), "r"(_sysv_min_63_4));
                uint32_t _sysv_min_63_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_6) : "r"(_sysv_min_63_2), "r"(_sysv_min_63_5));
                uint32_t _sysv_min_63_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_7) : "r"(_sysv_poll_group_4[8]), "r"(_sysv_poll_group_4[9]));
                uint32_t _sysv_min_63_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_8) : "r"(_sysv_poll_group_4[10]), "r"(_sysv_poll_group_4[11]));
                uint32_t _sysv_min_63_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_9) : "r"(_sysv_min_63_7), "r"(_sysv_min_63_8));
                uint32_t _sysv_min_63_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_10) : "r"(_sysv_poll_group_4[12]), "r"(_sysv_poll_group_4[13]));
                uint32_t _sysv_min_63_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_11) : "r"(_sysv_poll_group_4[14]), "r"(_sysv_poll_group_4[15]));
                uint32_t _sysv_min_63_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_12) : "r"(_sysv_min_63_10), "r"(_sysv_min_63_11));
                uint32_t _sysv_min_63_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_13) : "r"(_sysv_min_63_9), "r"(_sysv_min_63_12));
                uint32_t _sysv_min_63_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_14) : "r"(_sysv_min_63_6), "r"(_sysv_min_63_13));
                uint32_t _sysv_min_63_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_15) : "r"(_sysv_poll_group_4[16]), "r"(_sysv_poll_group_4[17]));
                uint32_t _sysv_min_63_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_16) : "r"(_sysv_poll_group_4[18]), "r"(_sysv_poll_group_4[19]));
                uint32_t _sysv_min_63_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_17) : "r"(_sysv_min_63_15), "r"(_sysv_min_63_16));
                uint32_t _sysv_min_63_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_18) : "r"(_sysv_poll_group_4[20]), "r"(_sysv_poll_group_4[21]));
                uint32_t _sysv_min_63_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_19) : "r"(_sysv_poll_group_4[22]), "r"(_sysv_poll_group_4[23]));
                uint32_t _sysv_min_63_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_20) : "r"(_sysv_min_63_18), "r"(_sysv_min_63_19));
                uint32_t _sysv_min_63_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_21) : "r"(_sysv_min_63_17), "r"(_sysv_min_63_20));
                uint32_t _sysv_min_63_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_22) : "r"(_sysv_poll_group_4[24]), "r"(_sysv_poll_group_4[25]));
                uint32_t _sysv_min_63_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_23) : "r"(_sysv_poll_group_4[26]), "r"(_sysv_poll_group_4[27]));
                uint32_t _sysv_min_63_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_24) : "r"(_sysv_min_63_22), "r"(_sysv_min_63_23));
                uint32_t _sysv_min_63_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_25) : "r"(_sysv_poll_group_4[28]), "r"(_sysv_poll_group_4[29]));
                uint32_t _sysv_min_63_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_26) : "r"(_sysv_poll_group_4[30]), "r"(_sysv_poll_group_4[31]));
                uint32_t _sysv_min_63_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_27) : "r"(_sysv_min_63_25), "r"(_sysv_min_63_26));
                uint32_t _sysv_min_63_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_28) : "r"(_sysv_min_63_24), "r"(_sysv_min_63_27));
                uint32_t _sysv_min_63_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_29) : "r"(_sysv_min_63_21), "r"(_sysv_min_63_28));
                uint32_t _sysv_min_63_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_63_30) : "r"(_sysv_min_63_14), "r"(_sysv_min_63_29));
                _sysv_min_62 = _sysv_min_63_30;
            } while (
                ((_sysv_min_62 & 0xffffu) == 0x8000u) || ((_sysv_min_62 >> 16) == 0x8000u)
            );
            float s8_4[8];
            float _unpack_bf16_34[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[_pair]));
            }
            s8_4[0] = _unpack_bf16_34[0];
            s8_4[1] = _unpack_bf16_34[1];
            s8_4[2] = _unpack_bf16_34[2];
            s8_4[3] = _unpack_bf16_34[3];
            s8_4[4] = _unpack_bf16_34[4];
            s8_4[5] = _unpack_bf16_34[5];
            s8_4[6] = _unpack_bf16_34[6];
            s8_4[7] = _unpack_bf16_34[7];
            float _unpack_bf16_35[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[4 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_35[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_35[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_35[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_35[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_35[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_35[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_35[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_35[7];
            float _unpack_bf16_36[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[8 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_36[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_36[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_36[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_36[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_36[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_36[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_36[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_36[7];
            float _unpack_bf16_37[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[12 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_37[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_37[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_37[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_37[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_37[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_37[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_37[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_37[7];
            float _unpack_bf16_38[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[16 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_38[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_38[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_38[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_38[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_38[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_38[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_38[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_38[7];
            float _unpack_bf16_39[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[20 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_39[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_39[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_39[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_39[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_39[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_39[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_39[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_39[7];
            float _unpack_bf16_40[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[24 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_40[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_40[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_40[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_40[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_40[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_40[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_40[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_40[7];
            float _unpack_bf16_41[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                    : "r"(_sysv_poll_group_4[28 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_41[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_41[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_41[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_41[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_41[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_41[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_41[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_41[7];
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
            float _unpack_bf16_y_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_4[_pair * 2])[0]), "=f"((&_unpack_bf16_y_4[_pair * 2])[1])
                    : "r"(s8_bf16_4[_pair]));
            }
            float ssq_4 = 0.0f;
            float y_32 = ((mode != 0) ? _unpack_bf16_y_4[0] * gw8[0] : _unpack_bf16_y_4[0]);
            ssq_4 = __fmaf_rn(y_32, y_32, ssq_4);
            float y_33 = ((mode != 0) ? _unpack_bf16_y_4[1] * gw8[1] : _unpack_bf16_y_4[1]);
            ssq_4 = __fmaf_rn(y_33, y_33, ssq_4);
            float y_34 = ((mode != 0) ? _unpack_bf16_y_4[2] * gw8[2] : _unpack_bf16_y_4[2]);
            ssq_4 = __fmaf_rn(y_34, y_34, ssq_4);
            float y_35 = ((mode != 0) ? _unpack_bf16_y_4[3] * gw8[3] : _unpack_bf16_y_4[3]);
            ssq_4 = __fmaf_rn(y_35, y_35, ssq_4);
            float y_36 = ((mode != 0) ? _unpack_bf16_y_4[4] * gw8[4] : _unpack_bf16_y_4[4]);
            ssq_4 = __fmaf_rn(y_36, y_36, ssq_4);
            float y_37 = ((mode != 0) ? _unpack_bf16_y_4[5] * gw8[5] : _unpack_bf16_y_4[5]);
            ssq_4 = __fmaf_rn(y_37, y_37, ssq_4);
            float y_38 = ((mode != 0) ? _unpack_bf16_y_4[6] * gw8[6] : _unpack_bf16_y_4[6]);
            ssq_4 = __fmaf_rn(y_38, y_38, ssq_4);
            float y_39 = ((mode != 0) ? _unpack_bf16_y_4[7] * gw8[7] : _unpack_bf16_y_4[7]);
            ssq_4 = __fmaf_rn(y_39, y_39, ssq_4);
            ssq8[4] = ssq_4;
            {
                auto _gst_64 = push_local + (src + 131072);
                __builtin_assume(__isGlobal(_gst_64));
                reinterpret_cast<int4*>(_gst_64)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_65 = push_local + (src + 135168);
                __builtin_assume(__isGlobal(_gst_65));
                reinterpret_cast<int4*>(_gst_65)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_66 = push_local + (src + 139264);
                __builtin_assume(__isGlobal(_gst_66));
                reinterpret_cast<int4*>(_gst_66)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_67 = push_local + (src + 143360);
                __builtin_assume(__isGlobal(_gst_67));
                reinterpret_cast<int4*>(_gst_67)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_68 = push_local + (src + 147456);
                __builtin_assume(__isGlobal(_gst_68));
                reinterpret_cast<int4*>(_gst_68)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_69 = push_local + (src + 151552);
                __builtin_assume(__isGlobal(_gst_69));
                reinterpret_cast<int4*>(_gst_69)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_70 = push_local + (src + 155648);
                __builtin_assume(__isGlobal(_gst_70));
                reinterpret_cast<int4*>(_gst_70)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_71 = push_local + (src + 159744);
                __builtin_assume(__isGlobal(_gst_71));
                reinterpret_cast<int4*>(_gst_71)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_4 = ssq8[4];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_4 += __shfl_xor_sync(0xFFFFFFFF, _wrg_4, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 4] = _wrg_4;
            }
            uint32_t _sysv_poll_group_5[32];
            uint32_t _sysv_min_72;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[0]), "=r"(_sysv_poll_group_5[1]), "=r"(_sysv_poll_group_5[2]), "=r"(_sysv_poll_group_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 163840))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[4]), "=r"(_sysv_poll_group_5[5]), "=r"(_sysv_poll_group_5[6]), "=r"(_sysv_poll_group_5[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 167936))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[8]), "=r"(_sysv_poll_group_5[9]), "=r"(_sysv_poll_group_5[10]), "=r"(_sysv_poll_group_5[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 172032))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[12]), "=r"(_sysv_poll_group_5[13]), "=r"(_sysv_poll_group_5[14]), "=r"(_sysv_poll_group_5[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 176128))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[16]), "=r"(_sysv_poll_group_5[17]), "=r"(_sysv_poll_group_5[18]), "=r"(_sysv_poll_group_5[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 180224))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[20]), "=r"(_sysv_poll_group_5[21]), "=r"(_sysv_poll_group_5[22]), "=r"(_sysv_poll_group_5[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 184320))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[24]), "=r"(_sysv_poll_group_5[25]), "=r"(_sysv_poll_group_5[26]), "=r"(_sysv_poll_group_5[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 188416))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_5[28]), "=r"(_sysv_poll_group_5[29]), "=r"(_sysv_poll_group_5[30]), "=r"(_sysv_poll_group_5[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 192512))) : "memory");
                uint32_t _sysv_min_73_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_0) : "r"(_sysv_poll_group_5[0]), "r"(_sysv_poll_group_5[1]));
                uint32_t _sysv_min_73_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_1) : "r"(_sysv_poll_group_5[2]), "r"(_sysv_poll_group_5[3]));
                uint32_t _sysv_min_73_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_2) : "r"(_sysv_min_73_0), "r"(_sysv_min_73_1));
                uint32_t _sysv_min_73_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_3) : "r"(_sysv_poll_group_5[4]), "r"(_sysv_poll_group_5[5]));
                uint32_t _sysv_min_73_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_4) : "r"(_sysv_poll_group_5[6]), "r"(_sysv_poll_group_5[7]));
                uint32_t _sysv_min_73_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_5) : "r"(_sysv_min_73_3), "r"(_sysv_min_73_4));
                uint32_t _sysv_min_73_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_6) : "r"(_sysv_min_73_2), "r"(_sysv_min_73_5));
                uint32_t _sysv_min_73_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_7) : "r"(_sysv_poll_group_5[8]), "r"(_sysv_poll_group_5[9]));
                uint32_t _sysv_min_73_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_8) : "r"(_sysv_poll_group_5[10]), "r"(_sysv_poll_group_5[11]));
                uint32_t _sysv_min_73_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_9) : "r"(_sysv_min_73_7), "r"(_sysv_min_73_8));
                uint32_t _sysv_min_73_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_10) : "r"(_sysv_poll_group_5[12]), "r"(_sysv_poll_group_5[13]));
                uint32_t _sysv_min_73_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_11) : "r"(_sysv_poll_group_5[14]), "r"(_sysv_poll_group_5[15]));
                uint32_t _sysv_min_73_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_12) : "r"(_sysv_min_73_10), "r"(_sysv_min_73_11));
                uint32_t _sysv_min_73_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_13) : "r"(_sysv_min_73_9), "r"(_sysv_min_73_12));
                uint32_t _sysv_min_73_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_14) : "r"(_sysv_min_73_6), "r"(_sysv_min_73_13));
                uint32_t _sysv_min_73_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_15) : "r"(_sysv_poll_group_5[16]), "r"(_sysv_poll_group_5[17]));
                uint32_t _sysv_min_73_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_16) : "r"(_sysv_poll_group_5[18]), "r"(_sysv_poll_group_5[19]));
                uint32_t _sysv_min_73_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_17) : "r"(_sysv_min_73_15), "r"(_sysv_min_73_16));
                uint32_t _sysv_min_73_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_18) : "r"(_sysv_poll_group_5[20]), "r"(_sysv_poll_group_5[21]));
                uint32_t _sysv_min_73_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_19) : "r"(_sysv_poll_group_5[22]), "r"(_sysv_poll_group_5[23]));
                uint32_t _sysv_min_73_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_20) : "r"(_sysv_min_73_18), "r"(_sysv_min_73_19));
                uint32_t _sysv_min_73_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_21) : "r"(_sysv_min_73_17), "r"(_sysv_min_73_20));
                uint32_t _sysv_min_73_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_22) : "r"(_sysv_poll_group_5[24]), "r"(_sysv_poll_group_5[25]));
                uint32_t _sysv_min_73_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_23) : "r"(_sysv_poll_group_5[26]), "r"(_sysv_poll_group_5[27]));
                uint32_t _sysv_min_73_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_24) : "r"(_sysv_min_73_22), "r"(_sysv_min_73_23));
                uint32_t _sysv_min_73_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_25) : "r"(_sysv_poll_group_5[28]), "r"(_sysv_poll_group_5[29]));
                uint32_t _sysv_min_73_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_26) : "r"(_sysv_poll_group_5[30]), "r"(_sysv_poll_group_5[31]));
                uint32_t _sysv_min_73_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_27) : "r"(_sysv_min_73_25), "r"(_sysv_min_73_26));
                uint32_t _sysv_min_73_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_28) : "r"(_sysv_min_73_24), "r"(_sysv_min_73_27));
                uint32_t _sysv_min_73_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_29) : "r"(_sysv_min_73_21), "r"(_sysv_min_73_28));
                uint32_t _sysv_min_73_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_73_30) : "r"(_sysv_min_73_14), "r"(_sysv_min_73_29));
                _sysv_min_72 = _sysv_min_73_30;
            } while (
                ((_sysv_min_72 & 0xffffu) == 0x8000u) || ((_sysv_min_72 >> 16) == 0x8000u)
            );
            float s8_5[8];
            float _unpack_bf16_42[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[_pair]));
            }
            s8_5[0] = _unpack_bf16_42[0];
            s8_5[1] = _unpack_bf16_42[1];
            s8_5[2] = _unpack_bf16_42[2];
            s8_5[3] = _unpack_bf16_42[3];
            s8_5[4] = _unpack_bf16_42[4];
            s8_5[5] = _unpack_bf16_42[5];
            s8_5[6] = _unpack_bf16_42[6];
            s8_5[7] = _unpack_bf16_42[7];
            float _unpack_bf16_43[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[4 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_43[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_43[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_43[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_43[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_43[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_43[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_43[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_43[7];
            float _unpack_bf16_44[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_44[_pair * 2])[0]), "=f"((&_unpack_bf16_44[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[8 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_44[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_44[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_44[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_44[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_44[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_44[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_44[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_44[7];
            float _unpack_bf16_45[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_45[_pair * 2])[0]), "=f"((&_unpack_bf16_45[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[12 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_45[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_45[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_45[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_45[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_45[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_45[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_45[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_45[7];
            float _unpack_bf16_46[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_46[_pair * 2])[0]), "=f"((&_unpack_bf16_46[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[16 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_46[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_46[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_46[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_46[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_46[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_46[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_46[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_46[7];
            float _unpack_bf16_47[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_47[_pair * 2])[0]), "=f"((&_unpack_bf16_47[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[20 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_47[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_47[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_47[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_47[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_47[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_47[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_47[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_47[7];
            float _unpack_bf16_48[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_48[_pair * 2])[0]), "=f"((&_unpack_bf16_48[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[24 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_48[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_48[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_48[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_48[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_48[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_48[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_48[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_48[7];
            float _unpack_bf16_49[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_49[_pair * 2])[0]), "=f"((&_unpack_bf16_49[_pair * 2])[1])
                    : "r"(_sysv_poll_group_5[28 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_49[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_49[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_49[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_49[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_49[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_49[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_49[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_49[7];
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
            float _unpack_bf16_y_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_5[_pair * 2])[0]), "=f"((&_unpack_bf16_y_5[_pair * 2])[1])
                    : "r"(s8_bf16_5[_pair]));
            }
            float ssq_5 = 0.0f;
            float y_40 = ((mode != 0) ? _unpack_bf16_y_5[0] * gw8[0] : _unpack_bf16_y_5[0]);
            ssq_5 = __fmaf_rn(y_40, y_40, ssq_5);
            float y_41 = ((mode != 0) ? _unpack_bf16_y_5[1] * gw8[1] : _unpack_bf16_y_5[1]);
            ssq_5 = __fmaf_rn(y_41, y_41, ssq_5);
            float y_42 = ((mode != 0) ? _unpack_bf16_y_5[2] * gw8[2] : _unpack_bf16_y_5[2]);
            ssq_5 = __fmaf_rn(y_42, y_42, ssq_5);
            float y_43 = ((mode != 0) ? _unpack_bf16_y_5[3] * gw8[3] : _unpack_bf16_y_5[3]);
            ssq_5 = __fmaf_rn(y_43, y_43, ssq_5);
            float y_44 = ((mode != 0) ? _unpack_bf16_y_5[4] * gw8[4] : _unpack_bf16_y_5[4]);
            ssq_5 = __fmaf_rn(y_44, y_44, ssq_5);
            float y_45 = ((mode != 0) ? _unpack_bf16_y_5[5] * gw8[5] : _unpack_bf16_y_5[5]);
            ssq_5 = __fmaf_rn(y_45, y_45, ssq_5);
            float y_46 = ((mode != 0) ? _unpack_bf16_y_5[6] * gw8[6] : _unpack_bf16_y_5[6]);
            ssq_5 = __fmaf_rn(y_46, y_46, ssq_5);
            float y_47 = ((mode != 0) ? _unpack_bf16_y_5[7] * gw8[7] : _unpack_bf16_y_5[7]);
            ssq_5 = __fmaf_rn(y_47, y_47, ssq_5);
            ssq8[5] = ssq_5;
            {
                auto _gst_74 = push_local + (src + 163840);
                __builtin_assume(__isGlobal(_gst_74));
                reinterpret_cast<int4*>(_gst_74)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_75 = push_local + (src + 167936);
                __builtin_assume(__isGlobal(_gst_75));
                reinterpret_cast<int4*>(_gst_75)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_76 = push_local + (src + 172032);
                __builtin_assume(__isGlobal(_gst_76));
                reinterpret_cast<int4*>(_gst_76)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_77 = push_local + (src + 176128);
                __builtin_assume(__isGlobal(_gst_77));
                reinterpret_cast<int4*>(_gst_77)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_78 = push_local + (src + 180224);
                __builtin_assume(__isGlobal(_gst_78));
                reinterpret_cast<int4*>(_gst_78)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_79 = push_local + (src + 184320);
                __builtin_assume(__isGlobal(_gst_79));
                reinterpret_cast<int4*>(_gst_79)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_80 = push_local + (src + 188416);
                __builtin_assume(__isGlobal(_gst_80));
                reinterpret_cast<int4*>(_gst_80)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_81 = push_local + (src + 192512);
                __builtin_assume(__isGlobal(_gst_81));
                reinterpret_cast<int4*>(_gst_81)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_5 = ssq8[5];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_5 += __shfl_xor_sync(0xFFFFFFFF, _wrg_5, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 5] = _wrg_5;
            }
            uint32_t _sysv_poll_group_6[32];
            uint32_t _sysv_min_82;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[0]), "=r"(_sysv_poll_group_6[1]), "=r"(_sysv_poll_group_6[2]), "=r"(_sysv_poll_group_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 196608))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[4]), "=r"(_sysv_poll_group_6[5]), "=r"(_sysv_poll_group_6[6]), "=r"(_sysv_poll_group_6[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 200704))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[8]), "=r"(_sysv_poll_group_6[9]), "=r"(_sysv_poll_group_6[10]), "=r"(_sysv_poll_group_6[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 204800))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[12]), "=r"(_sysv_poll_group_6[13]), "=r"(_sysv_poll_group_6[14]), "=r"(_sysv_poll_group_6[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 208896))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[16]), "=r"(_sysv_poll_group_6[17]), "=r"(_sysv_poll_group_6[18]), "=r"(_sysv_poll_group_6[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 212992))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[20]), "=r"(_sysv_poll_group_6[21]), "=r"(_sysv_poll_group_6[22]), "=r"(_sysv_poll_group_6[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 217088))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[24]), "=r"(_sysv_poll_group_6[25]), "=r"(_sysv_poll_group_6[26]), "=r"(_sysv_poll_group_6[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 221184))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_6[28]), "=r"(_sysv_poll_group_6[29]), "=r"(_sysv_poll_group_6[30]), "=r"(_sysv_poll_group_6[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 225280))) : "memory");
                uint32_t _sysv_min_83_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_0) : "r"(_sysv_poll_group_6[0]), "r"(_sysv_poll_group_6[1]));
                uint32_t _sysv_min_83_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_1) : "r"(_sysv_poll_group_6[2]), "r"(_sysv_poll_group_6[3]));
                uint32_t _sysv_min_83_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_2) : "r"(_sysv_min_83_0), "r"(_sysv_min_83_1));
                uint32_t _sysv_min_83_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_3) : "r"(_sysv_poll_group_6[4]), "r"(_sysv_poll_group_6[5]));
                uint32_t _sysv_min_83_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_4) : "r"(_sysv_poll_group_6[6]), "r"(_sysv_poll_group_6[7]));
                uint32_t _sysv_min_83_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_5) : "r"(_sysv_min_83_3), "r"(_sysv_min_83_4));
                uint32_t _sysv_min_83_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_6) : "r"(_sysv_min_83_2), "r"(_sysv_min_83_5));
                uint32_t _sysv_min_83_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_7) : "r"(_sysv_poll_group_6[8]), "r"(_sysv_poll_group_6[9]));
                uint32_t _sysv_min_83_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_8) : "r"(_sysv_poll_group_6[10]), "r"(_sysv_poll_group_6[11]));
                uint32_t _sysv_min_83_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_9) : "r"(_sysv_min_83_7), "r"(_sysv_min_83_8));
                uint32_t _sysv_min_83_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_10) : "r"(_sysv_poll_group_6[12]), "r"(_sysv_poll_group_6[13]));
                uint32_t _sysv_min_83_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_11) : "r"(_sysv_poll_group_6[14]), "r"(_sysv_poll_group_6[15]));
                uint32_t _sysv_min_83_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_12) : "r"(_sysv_min_83_10), "r"(_sysv_min_83_11));
                uint32_t _sysv_min_83_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_13) : "r"(_sysv_min_83_9), "r"(_sysv_min_83_12));
                uint32_t _sysv_min_83_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_14) : "r"(_sysv_min_83_6), "r"(_sysv_min_83_13));
                uint32_t _sysv_min_83_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_15) : "r"(_sysv_poll_group_6[16]), "r"(_sysv_poll_group_6[17]));
                uint32_t _sysv_min_83_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_16) : "r"(_sysv_poll_group_6[18]), "r"(_sysv_poll_group_6[19]));
                uint32_t _sysv_min_83_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_17) : "r"(_sysv_min_83_15), "r"(_sysv_min_83_16));
                uint32_t _sysv_min_83_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_18) : "r"(_sysv_poll_group_6[20]), "r"(_sysv_poll_group_6[21]));
                uint32_t _sysv_min_83_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_19) : "r"(_sysv_poll_group_6[22]), "r"(_sysv_poll_group_6[23]));
                uint32_t _sysv_min_83_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_20) : "r"(_sysv_min_83_18), "r"(_sysv_min_83_19));
                uint32_t _sysv_min_83_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_21) : "r"(_sysv_min_83_17), "r"(_sysv_min_83_20));
                uint32_t _sysv_min_83_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_22) : "r"(_sysv_poll_group_6[24]), "r"(_sysv_poll_group_6[25]));
                uint32_t _sysv_min_83_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_23) : "r"(_sysv_poll_group_6[26]), "r"(_sysv_poll_group_6[27]));
                uint32_t _sysv_min_83_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_24) : "r"(_sysv_min_83_22), "r"(_sysv_min_83_23));
                uint32_t _sysv_min_83_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_25) : "r"(_sysv_poll_group_6[28]), "r"(_sysv_poll_group_6[29]));
                uint32_t _sysv_min_83_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_26) : "r"(_sysv_poll_group_6[30]), "r"(_sysv_poll_group_6[31]));
                uint32_t _sysv_min_83_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_27) : "r"(_sysv_min_83_25), "r"(_sysv_min_83_26));
                uint32_t _sysv_min_83_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_28) : "r"(_sysv_min_83_24), "r"(_sysv_min_83_27));
                uint32_t _sysv_min_83_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_29) : "r"(_sysv_min_83_21), "r"(_sysv_min_83_28));
                uint32_t _sysv_min_83_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_83_30) : "r"(_sysv_min_83_14), "r"(_sysv_min_83_29));
                _sysv_min_82 = _sysv_min_83_30;
            } while (
                ((_sysv_min_82 & 0xffffu) == 0x8000u) || ((_sysv_min_82 >> 16) == 0x8000u)
            );
            float s8_6[8];
            float _unpack_bf16_50[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_50[_pair * 2])[0]), "=f"((&_unpack_bf16_50[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[_pair]));
            }
            s8_6[0] = _unpack_bf16_50[0];
            s8_6[1] = _unpack_bf16_50[1];
            s8_6[2] = _unpack_bf16_50[2];
            s8_6[3] = _unpack_bf16_50[3];
            s8_6[4] = _unpack_bf16_50[4];
            s8_6[5] = _unpack_bf16_50[5];
            s8_6[6] = _unpack_bf16_50[6];
            s8_6[7] = _unpack_bf16_50[7];
            float _unpack_bf16_51[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_51[_pair * 2])[0]), "=f"((&_unpack_bf16_51[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[4 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_51[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_51[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_51[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_51[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_51[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_51[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_51[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_51[7];
            float _unpack_bf16_52[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_52[_pair * 2])[0]), "=f"((&_unpack_bf16_52[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[8 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_52[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_52[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_52[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_52[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_52[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_52[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_52[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_52[7];
            float _unpack_bf16_53[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_53[_pair * 2])[0]), "=f"((&_unpack_bf16_53[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[12 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_53[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_53[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_53[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_53[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_53[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_53[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_53[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_53[7];
            float _unpack_bf16_54[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_54[_pair * 2])[0]), "=f"((&_unpack_bf16_54[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[16 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_54[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_54[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_54[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_54[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_54[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_54[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_54[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_54[7];
            float _unpack_bf16_55[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_55[_pair * 2])[0]), "=f"((&_unpack_bf16_55[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[20 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_55[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_55[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_55[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_55[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_55[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_55[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_55[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_55[7];
            float _unpack_bf16_56[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_56[_pair * 2])[0]), "=f"((&_unpack_bf16_56[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[24 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_56[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_56[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_56[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_56[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_56[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_56[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_56[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_56[7];
            float _unpack_bf16_57[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_57[_pair * 2])[0]), "=f"((&_unpack_bf16_57[_pair * 2])[1])
                    : "r"(_sysv_poll_group_6[28 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_57[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_57[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_57[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_57[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_57[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_57[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_57[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_57[7];
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
            float _unpack_bf16_y_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_6[_pair * 2])[0]), "=f"((&_unpack_bf16_y_6[_pair * 2])[1])
                    : "r"(s8_bf16_6[_pair]));
            }
            float ssq_6 = 0.0f;
            float y_48 = ((mode != 0) ? _unpack_bf16_y_6[0] * gw8[0] : _unpack_bf16_y_6[0]);
            ssq_6 = __fmaf_rn(y_48, y_48, ssq_6);
            float y_49 = ((mode != 0) ? _unpack_bf16_y_6[1] * gw8[1] : _unpack_bf16_y_6[1]);
            ssq_6 = __fmaf_rn(y_49, y_49, ssq_6);
            float y_50 = ((mode != 0) ? _unpack_bf16_y_6[2] * gw8[2] : _unpack_bf16_y_6[2]);
            ssq_6 = __fmaf_rn(y_50, y_50, ssq_6);
            float y_51 = ((mode != 0) ? _unpack_bf16_y_6[3] * gw8[3] : _unpack_bf16_y_6[3]);
            ssq_6 = __fmaf_rn(y_51, y_51, ssq_6);
            float y_52 = ((mode != 0) ? _unpack_bf16_y_6[4] * gw8[4] : _unpack_bf16_y_6[4]);
            ssq_6 = __fmaf_rn(y_52, y_52, ssq_6);
            float y_53 = ((mode != 0) ? _unpack_bf16_y_6[5] * gw8[5] : _unpack_bf16_y_6[5]);
            ssq_6 = __fmaf_rn(y_53, y_53, ssq_6);
            float y_54 = ((mode != 0) ? _unpack_bf16_y_6[6] * gw8[6] : _unpack_bf16_y_6[6]);
            ssq_6 = __fmaf_rn(y_54, y_54, ssq_6);
            float y_55 = ((mode != 0) ? _unpack_bf16_y_6[7] * gw8[7] : _unpack_bf16_y_6[7]);
            ssq_6 = __fmaf_rn(y_55, y_55, ssq_6);
            ssq8[6] = ssq_6;
            {
                auto _gst_84 = push_local + (src + 196608);
                __builtin_assume(__isGlobal(_gst_84));
                reinterpret_cast<int4*>(_gst_84)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_85 = push_local + (src + 200704);
                __builtin_assume(__isGlobal(_gst_85));
                reinterpret_cast<int4*>(_gst_85)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_86 = push_local + (src + 204800);
                __builtin_assume(__isGlobal(_gst_86));
                reinterpret_cast<int4*>(_gst_86)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_87 = push_local + (src + 208896);
                __builtin_assume(__isGlobal(_gst_87));
                reinterpret_cast<int4*>(_gst_87)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_88 = push_local + (src + 212992);
                __builtin_assume(__isGlobal(_gst_88));
                reinterpret_cast<int4*>(_gst_88)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_89 = push_local + (src + 217088);
                __builtin_assume(__isGlobal(_gst_89));
                reinterpret_cast<int4*>(_gst_89)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_90 = push_local + (src + 221184);
                __builtin_assume(__isGlobal(_gst_90));
                reinterpret_cast<int4*>(_gst_90)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_91 = push_local + (src + 225280);
                __builtin_assume(__isGlobal(_gst_91));
                reinterpret_cast<int4*>(_gst_91)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_6 = ssq8[6];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_6 += __shfl_xor_sync(0xFFFFFFFF, _wrg_6, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 6] = _wrg_6;
            }
            uint32_t _sysv_poll_group_7[32];
            uint32_t _sysv_min_92;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[0]), "=r"(_sysv_poll_group_7[1]), "=r"(_sysv_poll_group_7[2]), "=r"(_sysv_poll_group_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 229376))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[4]), "=r"(_sysv_poll_group_7[5]), "=r"(_sysv_poll_group_7[6]), "=r"(_sysv_poll_group_7[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 233472))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[8]), "=r"(_sysv_poll_group_7[9]), "=r"(_sysv_poll_group_7[10]), "=r"(_sysv_poll_group_7[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 237568))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[12]), "=r"(_sysv_poll_group_7[13]), "=r"(_sysv_poll_group_7[14]), "=r"(_sysv_poll_group_7[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 241664))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[16]), "=r"(_sysv_poll_group_7[17]), "=r"(_sysv_poll_group_7[18]), "=r"(_sysv_poll_group_7[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 245760))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[20]), "=r"(_sysv_poll_group_7[21]), "=r"(_sysv_poll_group_7[22]), "=r"(_sysv_poll_group_7[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 249856))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[24]), "=r"(_sysv_poll_group_7[25]), "=r"(_sysv_poll_group_7[26]), "=r"(_sysv_poll_group_7[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 253952))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_7[28]), "=r"(_sysv_poll_group_7[29]), "=r"(_sysv_poll_group_7[30]), "=r"(_sysv_poll_group_7[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 258048))) : "memory");
                uint32_t _sysv_min_93_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_0) : "r"(_sysv_poll_group_7[0]), "r"(_sysv_poll_group_7[1]));
                uint32_t _sysv_min_93_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_1) : "r"(_sysv_poll_group_7[2]), "r"(_sysv_poll_group_7[3]));
                uint32_t _sysv_min_93_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_2) : "r"(_sysv_min_93_0), "r"(_sysv_min_93_1));
                uint32_t _sysv_min_93_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_3) : "r"(_sysv_poll_group_7[4]), "r"(_sysv_poll_group_7[5]));
                uint32_t _sysv_min_93_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_4) : "r"(_sysv_poll_group_7[6]), "r"(_sysv_poll_group_7[7]));
                uint32_t _sysv_min_93_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_5) : "r"(_sysv_min_93_3), "r"(_sysv_min_93_4));
                uint32_t _sysv_min_93_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_6) : "r"(_sysv_min_93_2), "r"(_sysv_min_93_5));
                uint32_t _sysv_min_93_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_7) : "r"(_sysv_poll_group_7[8]), "r"(_sysv_poll_group_7[9]));
                uint32_t _sysv_min_93_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_8) : "r"(_sysv_poll_group_7[10]), "r"(_sysv_poll_group_7[11]));
                uint32_t _sysv_min_93_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_9) : "r"(_sysv_min_93_7), "r"(_sysv_min_93_8));
                uint32_t _sysv_min_93_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_10) : "r"(_sysv_poll_group_7[12]), "r"(_sysv_poll_group_7[13]));
                uint32_t _sysv_min_93_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_11) : "r"(_sysv_poll_group_7[14]), "r"(_sysv_poll_group_7[15]));
                uint32_t _sysv_min_93_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_12) : "r"(_sysv_min_93_10), "r"(_sysv_min_93_11));
                uint32_t _sysv_min_93_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_13) : "r"(_sysv_min_93_9), "r"(_sysv_min_93_12));
                uint32_t _sysv_min_93_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_14) : "r"(_sysv_min_93_6), "r"(_sysv_min_93_13));
                uint32_t _sysv_min_93_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_15) : "r"(_sysv_poll_group_7[16]), "r"(_sysv_poll_group_7[17]));
                uint32_t _sysv_min_93_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_16) : "r"(_sysv_poll_group_7[18]), "r"(_sysv_poll_group_7[19]));
                uint32_t _sysv_min_93_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_17) : "r"(_sysv_min_93_15), "r"(_sysv_min_93_16));
                uint32_t _sysv_min_93_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_18) : "r"(_sysv_poll_group_7[20]), "r"(_sysv_poll_group_7[21]));
                uint32_t _sysv_min_93_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_19) : "r"(_sysv_poll_group_7[22]), "r"(_sysv_poll_group_7[23]));
                uint32_t _sysv_min_93_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_20) : "r"(_sysv_min_93_18), "r"(_sysv_min_93_19));
                uint32_t _sysv_min_93_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_21) : "r"(_sysv_min_93_17), "r"(_sysv_min_93_20));
                uint32_t _sysv_min_93_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_22) : "r"(_sysv_poll_group_7[24]), "r"(_sysv_poll_group_7[25]));
                uint32_t _sysv_min_93_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_23) : "r"(_sysv_poll_group_7[26]), "r"(_sysv_poll_group_7[27]));
                uint32_t _sysv_min_93_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_24) : "r"(_sysv_min_93_22), "r"(_sysv_min_93_23));
                uint32_t _sysv_min_93_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_25) : "r"(_sysv_poll_group_7[28]), "r"(_sysv_poll_group_7[29]));
                uint32_t _sysv_min_93_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_26) : "r"(_sysv_poll_group_7[30]), "r"(_sysv_poll_group_7[31]));
                uint32_t _sysv_min_93_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_27) : "r"(_sysv_min_93_25), "r"(_sysv_min_93_26));
                uint32_t _sysv_min_93_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_28) : "r"(_sysv_min_93_24), "r"(_sysv_min_93_27));
                uint32_t _sysv_min_93_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_29) : "r"(_sysv_min_93_21), "r"(_sysv_min_93_28));
                uint32_t _sysv_min_93_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_93_30) : "r"(_sysv_min_93_14), "r"(_sysv_min_93_29));
                _sysv_min_92 = _sysv_min_93_30;
            } while (
                ((_sysv_min_92 & 0xffffu) == 0x8000u) || ((_sysv_min_92 >> 16) == 0x8000u)
            );
            float s8_7[8];
            float _unpack_bf16_58[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_58[_pair * 2])[0]), "=f"((&_unpack_bf16_58[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[_pair]));
            }
            s8_7[0] = _unpack_bf16_58[0];
            s8_7[1] = _unpack_bf16_58[1];
            s8_7[2] = _unpack_bf16_58[2];
            s8_7[3] = _unpack_bf16_58[3];
            s8_7[4] = _unpack_bf16_58[4];
            s8_7[5] = _unpack_bf16_58[5];
            s8_7[6] = _unpack_bf16_58[6];
            s8_7[7] = _unpack_bf16_58[7];
            float _unpack_bf16_59[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_59[_pair * 2])[0]), "=f"((&_unpack_bf16_59[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[4 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_59[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_59[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_59[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_59[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_59[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_59[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_59[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_59[7];
            float _unpack_bf16_60[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_60[_pair * 2])[0]), "=f"((&_unpack_bf16_60[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[8 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_60[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_60[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_60[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_60[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_60[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_60[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_60[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_60[7];
            float _unpack_bf16_61[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_61[_pair * 2])[0]), "=f"((&_unpack_bf16_61[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[12 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_61[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_61[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_61[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_61[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_61[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_61[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_61[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_61[7];
            float _unpack_bf16_62[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_62[_pair * 2])[0]), "=f"((&_unpack_bf16_62[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[16 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_62[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_62[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_62[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_62[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_62[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_62[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_62[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_62[7];
            float _unpack_bf16_63[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_63[_pair * 2])[0]), "=f"((&_unpack_bf16_63[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[20 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_63[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_63[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_63[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_63[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_63[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_63[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_63[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_63[7];
            float _unpack_bf16_64[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_64[_pair * 2])[0]), "=f"((&_unpack_bf16_64[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[24 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_64[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_64[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_64[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_64[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_64[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_64[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_64[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_64[7];
            float _unpack_bf16_65[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_65[_pair * 2])[0]), "=f"((&_unpack_bf16_65[_pair * 2])[1])
                    : "r"(_sysv_poll_group_7[28 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_65[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_65[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_65[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_65[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_65[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_65[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_65[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_65[7];
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
            float _unpack_bf16_y_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_y_7[_pair * 2])[0]), "=f"((&_unpack_bf16_y_7[_pair * 2])[1])
                    : "r"(s8_bf16_7[_pair]));
            }
            float ssq_7 = 0.0f;
            float y_56 = ((mode != 0) ? _unpack_bf16_y_7[0] * gw8[0] : _unpack_bf16_y_7[0]);
            ssq_7 = __fmaf_rn(y_56, y_56, ssq_7);
            float y_57 = ((mode != 0) ? _unpack_bf16_y_7[1] * gw8[1] : _unpack_bf16_y_7[1]);
            ssq_7 = __fmaf_rn(y_57, y_57, ssq_7);
            float y_58 = ((mode != 0) ? _unpack_bf16_y_7[2] * gw8[2] : _unpack_bf16_y_7[2]);
            ssq_7 = __fmaf_rn(y_58, y_58, ssq_7);
            float y_59 = ((mode != 0) ? _unpack_bf16_y_7[3] * gw8[3] : _unpack_bf16_y_7[3]);
            ssq_7 = __fmaf_rn(y_59, y_59, ssq_7);
            float y_60 = ((mode != 0) ? _unpack_bf16_y_7[4] * gw8[4] : _unpack_bf16_y_7[4]);
            ssq_7 = __fmaf_rn(y_60, y_60, ssq_7);
            float y_61 = ((mode != 0) ? _unpack_bf16_y_7[5] * gw8[5] : _unpack_bf16_y_7[5]);
            ssq_7 = __fmaf_rn(y_61, y_61, ssq_7);
            float y_62 = ((mode != 0) ? _unpack_bf16_y_7[6] * gw8[6] : _unpack_bf16_y_7[6]);
            ssq_7 = __fmaf_rn(y_62, y_62, ssq_7);
            float y_63 = ((mode != 0) ? _unpack_bf16_y_7[7] * gw8[7] : _unpack_bf16_y_7[7]);
            ssq_7 = __fmaf_rn(y_63, y_63, ssq_7);
            ssq8[7] = ssq_7;
            {
                auto _gst_94 = push_local + (src + 229376);
                __builtin_assume(__isGlobal(_gst_94));
                reinterpret_cast<int4*>(_gst_94)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_95 = push_local + (src + 233472);
                __builtin_assume(__isGlobal(_gst_95));
                reinterpret_cast<int4*>(_gst_95)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_96 = push_local + (src + 237568);
                __builtin_assume(__isGlobal(_gst_96));
                reinterpret_cast<int4*>(_gst_96)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_97 = push_local + (src + 241664);
                __builtin_assume(__isGlobal(_gst_97));
                reinterpret_cast<int4*>(_gst_97)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_98 = push_local + (src + 245760);
                __builtin_assume(__isGlobal(_gst_98));
                reinterpret_cast<int4*>(_gst_98)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_99 = push_local + (src + 249856);
                __builtin_assume(__isGlobal(_gst_99));
                reinterpret_cast<int4*>(_gst_99)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_100 = push_local + (src + 253952);
                __builtin_assume(__isGlobal(_gst_100));
                reinterpret_cast<int4*>(_gst_100)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_101 = push_local + (src + 258048);
                __builtin_assume(__isGlobal(_gst_101));
                reinterpret_cast<int4*>(_gst_101)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _wrg_7 = ssq8[7];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_7 += __shfl_xor_sync(0xFFFFFFFF, _wrg_7, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 7] = _wrg_7;
            }
            __syncthreads();
            int l8 = lane & 7;
            float tl = warp_tab[parity + l8];
            tl = tl + warp_tab[parity + 8 + l8];
            tl = tl + warp_tab[parity + 16 + l8];
            tl = tl + warp_tab[parity + 24 + l8];
            tl = tl + warp_tab[parity + 32 + l8];
            tl = tl + warp_tab[parity + 40 + l8];
            tl = tl + warp_tab[parity + 48 + l8];
            tl = tl + warp_tab[parity + 56 + l8];
            tl = tl + warp_tab[parity + 64 + l8];
            tl = tl + warp_tab[parity + 72 + l8];
            tl = tl + warp_tab[parity + 80 + l8];
            tl = tl + warp_tab[parity + 88 + l8];
            tl = tl + warp_tab[parity + 96 + l8];
            tl = tl + warp_tab[parity + 104 + l8];
            tl = tl + warp_tab[parity + 112 + l8];
            tl = tl + warp_tab[parity + 120 + l8];
            float _sqrt_102;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_102) : "f"(__fmaf_rn(tl, 0.00024414062f, eps)));
            float rsl = __frcp_rn(_sqrt_102);
            float rs8[8];
            float _shfl_sync_103 = __shfl_sync(0xFFFFFFFF, rsl, 0);
            rs8[0] = _shfl_sync_103;
            float _shfl_sync_104 = __shfl_sync(0xFFFFFFFF, rsl, 1);
            rs8[1] = _shfl_sync_104;
            float _shfl_sync_105 = __shfl_sync(0xFFFFFFFF, rsl, 2);
            rs8[2] = _shfl_sync_105;
            float _shfl_sync_106 = __shfl_sync(0xFFFFFFFF, rsl, 3);
            rs8[3] = _shfl_sync_106;
            float _shfl_sync_107 = __shfl_sync(0xFFFFFFFF, rsl, 4);
            rs8[4] = _shfl_sync_107;
            float _shfl_sync_108 = __shfl_sync(0xFFFFFFFF, rsl, 5);
            rs8[5] = _shfl_sync_108;
            float _shfl_sync_109 = __shfl_sync(0xFFFFFFFF, rsl, 6);
            rs8[6] = _shfl_sync_109;
            float _shfl_sync_110 = __shfl_sync(0xFFFFFFFF, rsl, 7);
            rs8[7] = _shfl_sync_110;
            float acc8[8];
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            float rs_0 = rs8[0];
            float _unpack_bf16_66[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_66[_pair * 2])[0]), "=f"((&_unpack_bf16_66[_pair * 2])[1])
                    : "r"(rw[_pair]));
            }
            float yn_0 = ((mode != 0) ? _unpack_bf16_66[0] * gw8[0] : _unpack_bf16_66[0]);
            float o_0 = yn_0 * rs_0;
            float o2_8_0 = ((mode != 0) ? o_0 : o_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_0, w8[0], acc8[0]);
            float yn_1 = ((mode != 0) ? _unpack_bf16_66[1] * gw8[1] : _unpack_bf16_66[1]);
            float o_1 = yn_1 * rs_0;
            float o2_8_1 = ((mode != 0) ? o_1 : o_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_1, w8[0], acc8[1]);
            float yn_2 = ((mode != 0) ? _unpack_bf16_66[2] * gw8[2] : _unpack_bf16_66[2]);
            float o_2 = yn_2 * rs_0;
            float o2_8_2 = ((mode != 0) ? o_2 : o_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_2, w8[0], acc8[2]);
            float yn_3 = ((mode != 0) ? _unpack_bf16_66[3] * gw8[3] : _unpack_bf16_66[3]);
            float o_3 = yn_3 * rs_0;
            float o2_8_3 = ((mode != 0) ? o_3 : o_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_3, w8[0], acc8[3]);
            float yn_4 = ((mode != 0) ? _unpack_bf16_66[4] * gw8[4] : _unpack_bf16_66[4]);
            float o_4 = yn_4 * rs_0;
            float o2_8_4 = ((mode != 0) ? o_4 : o_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_4, w8[0], acc8[4]);
            float yn_5 = ((mode != 0) ? _unpack_bf16_66[5] * gw8[5] : _unpack_bf16_66[5]);
            float o_5 = yn_5 * rs_0;
            float o2_8_5 = ((mode != 0) ? o_5 : o_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_5, w8[0], acc8[5]);
            float yn_6 = ((mode != 0) ? _unpack_bf16_66[6] * gw8[6] : _unpack_bf16_66[6]);
            float o_6 = yn_6 * rs_0;
            float o2_8_6 = ((mode != 0) ? o_6 : o_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_6, w8[0], acc8[6]);
            float yn_7 = ((mode != 0) ? _unpack_bf16_66[7] * gw8[7] : _unpack_bf16_66[7]);
            float o_7 = yn_7 * rs_0;
            float o2_8_7 = ((mode != 0) ? o_7 : o_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_7, w8[0], acc8[7]);
            float rs_1 = rs8[1];
            float _unpack_bf16_67[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_67[_pair * 2])[0]), "=f"((&_unpack_bf16_67[_pair * 2])[1])
                    : "r"(rw[4 + _pair]));
            }
            float yn_8 = ((mode != 0) ? _unpack_bf16_67[0] * gw8[0] : _unpack_bf16_67[0]);
            float o_8 = yn_8 * rs_1;
            float o2_8_8 = ((mode != 0) ? o_8 : o_8 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_8, w8[1], acc8[0]);
            float yn_9 = ((mode != 0) ? _unpack_bf16_67[1] * gw8[1] : _unpack_bf16_67[1]);
            float o_9 = yn_9 * rs_1;
            float o2_8_9 = ((mode != 0) ? o_9 : o_9 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_9, w8[1], acc8[1]);
            float yn_10 = ((mode != 0) ? _unpack_bf16_67[2] * gw8[2] : _unpack_bf16_67[2]);
            float o_10 = yn_10 * rs_1;
            float o2_8_10 = ((mode != 0) ? o_10 : o_10 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_10, w8[1], acc8[2]);
            float yn_11 = ((mode != 0) ? _unpack_bf16_67[3] * gw8[3] : _unpack_bf16_67[3]);
            float o_11 = yn_11 * rs_1;
            float o2_8_11 = ((mode != 0) ? o_11 : o_11 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_11, w8[1], acc8[3]);
            float yn_12 = ((mode != 0) ? _unpack_bf16_67[4] * gw8[4] : _unpack_bf16_67[4]);
            float o_12 = yn_12 * rs_1;
            float o2_8_12 = ((mode != 0) ? o_12 : o_12 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_12, w8[1], acc8[4]);
            float yn_13 = ((mode != 0) ? _unpack_bf16_67[5] * gw8[5] : _unpack_bf16_67[5]);
            float o_13 = yn_13 * rs_1;
            float o2_8_13 = ((mode != 0) ? o_13 : o_13 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_13, w8[1], acc8[5]);
            float yn_14 = ((mode != 0) ? _unpack_bf16_67[6] * gw8[6] : _unpack_bf16_67[6]);
            float o_14 = yn_14 * rs_1;
            float o2_8_14 = ((mode != 0) ? o_14 : o_14 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_14, w8[1], acc8[6]);
            float yn_15 = ((mode != 0) ? _unpack_bf16_67[7] * gw8[7] : _unpack_bf16_67[7]);
            float o_15 = yn_15 * rs_1;
            float o2_8_15 = ((mode != 0) ? o_15 : o_15 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_15, w8[1], acc8[7]);
            float rs_2 = rs8[2];
            float _unpack_bf16_68[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_68[_pair * 2])[0]), "=f"((&_unpack_bf16_68[_pair * 2])[1])
                    : "r"(rw[8 + _pair]));
            }
            float yn_16 = ((mode != 0) ? _unpack_bf16_68[0] * gw8[0] : _unpack_bf16_68[0]);
            float o_16 = yn_16 * rs_2;
            float o2_8_16 = ((mode != 0) ? o_16 : o_16 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_16, w8[2], acc8[0]);
            float yn_17 = ((mode != 0) ? _unpack_bf16_68[1] * gw8[1] : _unpack_bf16_68[1]);
            float o_17 = yn_17 * rs_2;
            float o2_8_17 = ((mode != 0) ? o_17 : o_17 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_17, w8[2], acc8[1]);
            float yn_18 = ((mode != 0) ? _unpack_bf16_68[2] * gw8[2] : _unpack_bf16_68[2]);
            float o_18 = yn_18 * rs_2;
            float o2_8_18 = ((mode != 0) ? o_18 : o_18 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_18, w8[2], acc8[2]);
            float yn_19 = ((mode != 0) ? _unpack_bf16_68[3] * gw8[3] : _unpack_bf16_68[3]);
            float o_19 = yn_19 * rs_2;
            float o2_8_19 = ((mode != 0) ? o_19 : o_19 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_19, w8[2], acc8[3]);
            float yn_20 = ((mode != 0) ? _unpack_bf16_68[4] * gw8[4] : _unpack_bf16_68[4]);
            float o_20 = yn_20 * rs_2;
            float o2_8_20 = ((mode != 0) ? o_20 : o_20 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_20, w8[2], acc8[4]);
            float yn_21 = ((mode != 0) ? _unpack_bf16_68[5] * gw8[5] : _unpack_bf16_68[5]);
            float o_21 = yn_21 * rs_2;
            float o2_8_21 = ((mode != 0) ? o_21 : o_21 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_21, w8[2], acc8[5]);
            float yn_22 = ((mode != 0) ? _unpack_bf16_68[6] * gw8[6] : _unpack_bf16_68[6]);
            float o_22 = yn_22 * rs_2;
            float o2_8_22 = ((mode != 0) ? o_22 : o_22 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_22, w8[2], acc8[6]);
            float yn_23 = ((mode != 0) ? _unpack_bf16_68[7] * gw8[7] : _unpack_bf16_68[7]);
            float o_23 = yn_23 * rs_2;
            float o2_8_23 = ((mode != 0) ? o_23 : o_23 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_23, w8[2], acc8[7]);
            float rs_3 = rs8[3];
            float _unpack_bf16_69[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_69[_pair * 2])[0]), "=f"((&_unpack_bf16_69[_pair * 2])[1])
                    : "r"(rw[12 + _pair]));
            }
            float yn_24 = ((mode != 0) ? _unpack_bf16_69[0] * gw8[0] : _unpack_bf16_69[0]);
            float o_24 = yn_24 * rs_3;
            float o2_8_24 = ((mode != 0) ? o_24 : o_24 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_24, w8[3], acc8[0]);
            float yn_25 = ((mode != 0) ? _unpack_bf16_69[1] * gw8[1] : _unpack_bf16_69[1]);
            float o_25 = yn_25 * rs_3;
            float o2_8_25 = ((mode != 0) ? o_25 : o_25 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_25, w8[3], acc8[1]);
            float yn_26 = ((mode != 0) ? _unpack_bf16_69[2] * gw8[2] : _unpack_bf16_69[2]);
            float o_26 = yn_26 * rs_3;
            float o2_8_26 = ((mode != 0) ? o_26 : o_26 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_26, w8[3], acc8[2]);
            float yn_27 = ((mode != 0) ? _unpack_bf16_69[3] * gw8[3] : _unpack_bf16_69[3]);
            float o_27 = yn_27 * rs_3;
            float o2_8_27 = ((mode != 0) ? o_27 : o_27 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_27, w8[3], acc8[3]);
            float yn_28 = ((mode != 0) ? _unpack_bf16_69[4] * gw8[4] : _unpack_bf16_69[4]);
            float o_28 = yn_28 * rs_3;
            float o2_8_28 = ((mode != 0) ? o_28 : o_28 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_28, w8[3], acc8[4]);
            float yn_29 = ((mode != 0) ? _unpack_bf16_69[5] * gw8[5] : _unpack_bf16_69[5]);
            float o_29 = yn_29 * rs_3;
            float o2_8_29 = ((mode != 0) ? o_29 : o_29 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_29, w8[3], acc8[5]);
            float yn_30 = ((mode != 0) ? _unpack_bf16_69[6] * gw8[6] : _unpack_bf16_69[6]);
            float o_30 = yn_30 * rs_3;
            float o2_8_30 = ((mode != 0) ? o_30 : o_30 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_30, w8[3], acc8[6]);
            float yn_31 = ((mode != 0) ? _unpack_bf16_69[7] * gw8[7] : _unpack_bf16_69[7]);
            float o_31 = yn_31 * rs_3;
            float o2_8_31 = ((mode != 0) ? o_31 : o_31 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_31, w8[3], acc8[7]);
            float rs_4 = rs8[4];
            float _unpack_bf16_70[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_70[_pair * 2])[0]), "=f"((&_unpack_bf16_70[_pair * 2])[1])
                    : "r"(rw[16 + _pair]));
            }
            float yn_32 = ((mode != 0) ? _unpack_bf16_70[0] * gw8[0] : _unpack_bf16_70[0]);
            float o_32 = yn_32 * rs_4;
            float o2_8_32 = ((mode != 0) ? o_32 : o_32 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_32, w8[4], acc8[0]);
            float yn_33 = ((mode != 0) ? _unpack_bf16_70[1] * gw8[1] : _unpack_bf16_70[1]);
            float o_33 = yn_33 * rs_4;
            float o2_8_33 = ((mode != 0) ? o_33 : o_33 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_33, w8[4], acc8[1]);
            float yn_34 = ((mode != 0) ? _unpack_bf16_70[2] * gw8[2] : _unpack_bf16_70[2]);
            float o_34 = yn_34 * rs_4;
            float o2_8_34 = ((mode != 0) ? o_34 : o_34 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_34, w8[4], acc8[2]);
            float yn_35 = ((mode != 0) ? _unpack_bf16_70[3] * gw8[3] : _unpack_bf16_70[3]);
            float o_35 = yn_35 * rs_4;
            float o2_8_35 = ((mode != 0) ? o_35 : o_35 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_35, w8[4], acc8[3]);
            float yn_36 = ((mode != 0) ? _unpack_bf16_70[4] * gw8[4] : _unpack_bf16_70[4]);
            float o_36 = yn_36 * rs_4;
            float o2_8_36 = ((mode != 0) ? o_36 : o_36 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_36, w8[4], acc8[4]);
            float yn_37 = ((mode != 0) ? _unpack_bf16_70[5] * gw8[5] : _unpack_bf16_70[5]);
            float o_37 = yn_37 * rs_4;
            float o2_8_37 = ((mode != 0) ? o_37 : o_37 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_37, w8[4], acc8[5]);
            float yn_38 = ((mode != 0) ? _unpack_bf16_70[6] * gw8[6] : _unpack_bf16_70[6]);
            float o_38 = yn_38 * rs_4;
            float o2_8_38 = ((mode != 0) ? o_38 : o_38 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_38, w8[4], acc8[6]);
            float yn_39 = ((mode != 0) ? _unpack_bf16_70[7] * gw8[7] : _unpack_bf16_70[7]);
            float o_39 = yn_39 * rs_4;
            float o2_8_39 = ((mode != 0) ? o_39 : o_39 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_39, w8[4], acc8[7]);
            float rs_5 = rs8[5];
            float _unpack_bf16_71[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_71[_pair * 2])[0]), "=f"((&_unpack_bf16_71[_pair * 2])[1])
                    : "r"(rw[20 + _pair]));
            }
            float yn_40 = ((mode != 0) ? _unpack_bf16_71[0] * gw8[0] : _unpack_bf16_71[0]);
            float o_40 = yn_40 * rs_5;
            float o2_8_40 = ((mode != 0) ? o_40 : o_40 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_40, w8[5], acc8[0]);
            float yn_41 = ((mode != 0) ? _unpack_bf16_71[1] * gw8[1] : _unpack_bf16_71[1]);
            float o_41 = yn_41 * rs_5;
            float o2_8_41 = ((mode != 0) ? o_41 : o_41 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_41, w8[5], acc8[1]);
            float yn_42 = ((mode != 0) ? _unpack_bf16_71[2] * gw8[2] : _unpack_bf16_71[2]);
            float o_42 = yn_42 * rs_5;
            float o2_8_42 = ((mode != 0) ? o_42 : o_42 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_42, w8[5], acc8[2]);
            float yn_43 = ((mode != 0) ? _unpack_bf16_71[3] * gw8[3] : _unpack_bf16_71[3]);
            float o_43 = yn_43 * rs_5;
            float o2_8_43 = ((mode != 0) ? o_43 : o_43 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_43, w8[5], acc8[3]);
            float yn_44 = ((mode != 0) ? _unpack_bf16_71[4] * gw8[4] : _unpack_bf16_71[4]);
            float o_44 = yn_44 * rs_5;
            float o2_8_44 = ((mode != 0) ? o_44 : o_44 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_44, w8[5], acc8[4]);
            float yn_45 = ((mode != 0) ? _unpack_bf16_71[5] * gw8[5] : _unpack_bf16_71[5]);
            float o_45 = yn_45 * rs_5;
            float o2_8_45 = ((mode != 0) ? o_45 : o_45 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_45, w8[5], acc8[5]);
            float yn_46 = ((mode != 0) ? _unpack_bf16_71[6] * gw8[6] : _unpack_bf16_71[6]);
            float o_46 = yn_46 * rs_5;
            float o2_8_46 = ((mode != 0) ? o_46 : o_46 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_46, w8[5], acc8[6]);
            float yn_47 = ((mode != 0) ? _unpack_bf16_71[7] * gw8[7] : _unpack_bf16_71[7]);
            float o_47 = yn_47 * rs_5;
            float o2_8_47 = ((mode != 0) ? o_47 : o_47 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_47, w8[5], acc8[7]);
            float rs_6 = rs8[6];
            float _unpack_bf16_72[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_72[_pair * 2])[0]), "=f"((&_unpack_bf16_72[_pair * 2])[1])
                    : "r"(rw[24 + _pair]));
            }
            float yn_48 = ((mode != 0) ? _unpack_bf16_72[0] * gw8[0] : _unpack_bf16_72[0]);
            float o_48 = yn_48 * rs_6;
            float o2_8_48 = ((mode != 0) ? o_48 : o_48 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_48, w8[6], acc8[0]);
            float yn_49 = ((mode != 0) ? _unpack_bf16_72[1] * gw8[1] : _unpack_bf16_72[1]);
            float o_49 = yn_49 * rs_6;
            float o2_8_49 = ((mode != 0) ? o_49 : o_49 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_49, w8[6], acc8[1]);
            float yn_50 = ((mode != 0) ? _unpack_bf16_72[2] * gw8[2] : _unpack_bf16_72[2]);
            float o_50 = yn_50 * rs_6;
            float o2_8_50 = ((mode != 0) ? o_50 : o_50 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_50, w8[6], acc8[2]);
            float yn_51 = ((mode != 0) ? _unpack_bf16_72[3] * gw8[3] : _unpack_bf16_72[3]);
            float o_51 = yn_51 * rs_6;
            float o2_8_51 = ((mode != 0) ? o_51 : o_51 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_51, w8[6], acc8[3]);
            float yn_52 = ((mode != 0) ? _unpack_bf16_72[4] * gw8[4] : _unpack_bf16_72[4]);
            float o_52 = yn_52 * rs_6;
            float o2_8_52 = ((mode != 0) ? o_52 : o_52 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_52, w8[6], acc8[4]);
            float yn_53 = ((mode != 0) ? _unpack_bf16_72[5] * gw8[5] : _unpack_bf16_72[5]);
            float o_53 = yn_53 * rs_6;
            float o2_8_53 = ((mode != 0) ? o_53 : o_53 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_53, w8[6], acc8[5]);
            float yn_54 = ((mode != 0) ? _unpack_bf16_72[6] * gw8[6] : _unpack_bf16_72[6]);
            float o_54 = yn_54 * rs_6;
            float o2_8_54 = ((mode != 0) ? o_54 : o_54 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_54, w8[6], acc8[6]);
            float yn_55 = ((mode != 0) ? _unpack_bf16_72[7] * gw8[7] : _unpack_bf16_72[7]);
            float o_55 = yn_55 * rs_6;
            float o2_8_55 = ((mode != 0) ? o_55 : o_55 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_55, w8[6], acc8[7]);
            float rs_7 = rs8[7];
            float _unpack_bf16_73[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_73[_pair * 2])[0]), "=f"((&_unpack_bf16_73[_pair * 2])[1])
                    : "r"(rw[28 + _pair]));
            }
            float yn_56 = ((mode != 0) ? _unpack_bf16_73[0] * gw8[0] : _unpack_bf16_73[0]);
            float o_56 = yn_56 * rs_7;
            float o2_8_56 = ((mode != 0) ? o_56 : o_56 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8_56, w8[7], acc8[0]);
            float yn_57 = ((mode != 0) ? _unpack_bf16_73[1] * gw8[1] : _unpack_bf16_73[1]);
            float o_57 = yn_57 * rs_7;
            float o2_8_57 = ((mode != 0) ? o_57 : o_57 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_8_57, w8[7], acc8[1]);
            float yn_58 = ((mode != 0) ? _unpack_bf16_73[2] * gw8[2] : _unpack_bf16_73[2]);
            float o_58 = yn_58 * rs_7;
            float o2_8_58 = ((mode != 0) ? o_58 : o_58 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_8_58, w8[7], acc8[2]);
            float yn_59 = ((mode != 0) ? _unpack_bf16_73[3] * gw8[3] : _unpack_bf16_73[3]);
            float o_59 = yn_59 * rs_7;
            float o2_8_59 = ((mode != 0) ? o_59 : o_59 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_8_59, w8[7], acc8[3]);
            float yn_60 = ((mode != 0) ? _unpack_bf16_73[4] * gw8[4] : _unpack_bf16_73[4]);
            float o_60 = yn_60 * rs_7;
            float o2_8_60 = ((mode != 0) ? o_60 : o_60 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_8_60, w8[7], acc8[4]);
            float yn_61 = ((mode != 0) ? _unpack_bf16_73[5] * gw8[5] : _unpack_bf16_73[5]);
            float o_61 = yn_61 * rs_7;
            float o2_8_61 = ((mode != 0) ? o_61 : o_61 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_8_61, w8[7], acc8[5]);
            float yn_62 = ((mode != 0) ? _unpack_bf16_73[6] * gw8[6] : _unpack_bf16_73[6]);
            float o_62 = yn_62 * rs_7;
            float o2_8_62 = ((mode != 0) ? o_62 : o_62 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_8_62, w8[7], acc8[6]);
            float yn_63 = ((mode != 0) ? _unpack_bf16_73[7] * gw8[7] : _unpack_bf16_73[7]);
            float o_63 = yn_63 * rs_7;
            float o2_8_63 = ((mode != 0) ? o_63 : o_63 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_8_63, w8[7], acc8[7]);
            unsigned int acc8_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 0], acc8[_lp*2+1 + 0]));
                acc8_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _packed16_sanitize_8[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_111 = static_cast<uint32_t>(acc8_bf16[_p16s_word]);
                uint32_t _p16s_lo_111 = _p16s_carrier_111 & 0x0000ffffu;
                uint32_t _p16s_hi_111 = (_p16s_carrier_111 >> 16) & 0x0000ffffu;
                if (_p16s_lo_111 == 0x00008000u) _p16s_lo_111 = 0u;
                if (_p16s_hi_111 == 0x00008000u) _p16s_hi_111 = 0u;
                _packed16_sanitize_8[_p16s_word] = _p16s_lo_111 | (_p16s_hi_111 << 16);
            }
            int gcr = sr & 7;
            long long pub = bcast_gen_elems * (long long)gcr + bcast_cta;
            asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<float*>(c_mcast + (unsigned long long)(pub * 2))), "r"(_packed16_sanitize_8[0]), "r"(_packed16_sanitize_8[1]), "r"(_packed16_sanitize_8[2]), "r"(_packed16_sanitize_8[3]) : "memory");
        }
        int jd = step - 6;
        if (jd >= 0 && jd < n_mine) {
            int tok_d = b + jd * grid;
            int sd = s0 + jd;
            int gcd = sd & 7;
            long long land = bcast_gen_elems * (long long)gcd + bcast_cta;
            uint32_t _sysv_poll_group_8[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_8[0]), "=r"(_sysv_poll_group_8[1]), "=r"(_sysv_poll_group_8[2]), "=r"(_sysv_poll_group_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(bcast_local + land)) : "memory");
            } while (
                (((_sysv_poll_group_8[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_8[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_8[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_8[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_8[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_8[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_8[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_8[3] >> 16) & 0xffffu) == 0x8000u)
            );
            unsigned int row4[4];
            row4[0] = _sysv_poll_group_8[0];
            row4[1] = _sysv_poll_group_8[1];
            row4[2] = _sysv_poll_group_8[2];
            row4[3] = _sysv_poll_group_8[3];
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output + ((long long)tok_d * 4096 + col64)))[0] = reinterpret_cast<int4*>(row4)[0];
            {
                auto _gst_112 = bcast_local + land;
                __builtin_assume(__isGlobal(_gst_112));
                reinterpret_cast<int4*>(_gst_112)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
        }
    }
    int s_end = s0 + steps;
    __syncthreads();
    if (tid == 0) {
        *(reinterpret_cast<int*>(ctl + b) + (0)) = s_end;
    }
}

} // extern "C"
