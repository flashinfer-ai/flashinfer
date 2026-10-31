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
kernel_cake_tp_moe_finalize_rmsnorm_62b5c5b3c6a275712199(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int max_ctas, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, unsigned long long c_mcast, int* __restrict__ ctl)
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
    int steps = n_mine + 4;
    long long push_gen_elems = (long long)max_ctas * 131072;
    long long push_cta = (long long)b * 131072 + col64;
    long long bcast_gen_elems = (long long)max_ctas * 4096;
    long long bcast_cta = (long long)b * 4096 + col64;
    long long rank64 = (long long)rank;
    long long push_addr_0 = peer_table[0];
    long long push_addr_1 = peer_table[1];
    long long push_addr_2 = peer_table[2];
    long long push_addr_3 = peer_table[3];
    __nv_bfloat16* push_peers_0 = reinterpret_cast<__nv_bfloat16*>(push_addr_0);
    __nv_bfloat16* push_peers_1 = reinterpret_cast<__nv_bfloat16*>(push_addr_1);
    __nv_bfloat16* push_peers_2 = reinterpret_cast<__nv_bfloat16*>(push_addr_2);
    __nv_bfloat16* push_peers_3 = reinterpret_cast<__nv_bfloat16*>(push_addr_3);
    long long bcast_addr_0 = peer_table[4];
    long long bcast_addr_1 = peer_table[5];
    long long bcast_addr_2 = peer_table[6];
    long long bcast_addr_3 = peer_table[7];
    __nv_bfloat16* bcast_peers_0 = reinterpret_cast<__nv_bfloat16*>(bcast_addr_0);
    __nv_bfloat16* bcast_peers_1 = reinterpret_cast<__nv_bfloat16*>(bcast_addr_1);
    __nv_bfloat16* bcast_peers_2 = reinterpret_cast<__nv_bfloat16*>(bcast_addr_2);
    __nv_bfloat16* bcast_peers_3 = reinterpret_cast<__nv_bfloat16*>(bcast_addr_3);
    long long push_local_addr = peer_table[rank];
    __nv_bfloat16* push_local = reinterpret_cast<__nv_bfloat16*>(push_local_addr);
    long long bcast_local_addr = peer_table[4 + rank];
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
            int owner = b + step & 3;
            int gb = s >> 2 & 3;
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
            int ridx_0 = idx8[0];
            if (ridx_0 >= 0 && ridx_0 < rows) {
                {
                    uint4 _uv4_3 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_0 * (long long)row_stride + col64) + 0);
                    packets[0 + 0] = _uv4_3.x;
                    packets[0 + 1] = _uv4_3.y;
                    packets[0 + 2] = _uv4_3.z;
                    packets[0 + 3] = _uv4_3.w;
                }
            }
            int ridx_1 = idx8[1];
            if (ridx_1 >= 0 && ridx_1 < rows) {
                {
                    uint4 _uv4_4 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_1 * (long long)row_stride + col64) + 0);
                    packets[4 + 0] = _uv4_4.x;
                    packets[4 + 1] = _uv4_4.y;
                    packets[4 + 2] = _uv4_4.z;
                    packets[4 + 3] = _uv4_4.w;
                }
            }
            int ridx_2 = idx8[2];
            if (ridx_2 >= 0 && ridx_2 < rows) {
                {
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_2 * (long long)row_stride + col64) + 0);
                    packets[8 + 0] = _uv4_5.x;
                    packets[8 + 1] = _uv4_5.y;
                    packets[8 + 2] = _uv4_5.z;
                    packets[8 + 3] = _uv4_5.w;
                }
            }
            int ridx_3 = idx8[3];
            if (ridx_3 >= 0 && ridx_3 < rows) {
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_3 * (long long)row_stride + col64) + 0);
                    packets[12 + 0] = _uv4_6.x;
                    packets[12 + 1] = _uv4_6.y;
                    packets[12 + 2] = _uv4_6.z;
                    packets[12 + 3] = _uv4_6.w;
                }
            }
            int ridx_4 = idx8[4];
            if (ridx_4 >= 0 && ridx_4 < rows) {
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_4 * (long long)row_stride + col64) + 0);
                    packets[16 + 0] = _uv4_7.x;
                    packets[16 + 1] = _uv4_7.y;
                    packets[16 + 2] = _uv4_7.z;
                    packets[16 + 3] = _uv4_7.w;
                }
            }
            int ridx_5 = idx8[5];
            if (ridx_5 >= 0 && ridx_5 < rows) {
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_5 * (long long)row_stride + col64) + 0);
                    packets[20 + 0] = _uv4_8.x;
                    packets[20 + 1] = _uv4_8.y;
                    packets[20 + 2] = _uv4_8.z;
                    packets[20 + 3] = _uv4_8.w;
                }
            }
            int ridx_6 = idx8[6];
            if (ridx_6 >= 0 && ridx_6 < rows) {
                {
                    uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_6 * (long long)row_stride + col64) + 0);
                    packets[24 + 0] = _uv4_9.x;
                    packets[24 + 1] = _uv4_9.y;
                    packets[24 + 2] = _uv4_9.z;
                    packets[24 + 3] = _uv4_9.w;
                }
            }
            int ridx_7 = idx8[7];
            if (ridx_7 >= 0 && ridx_7 < rows) {
                {
                    uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_7 * (long long)row_stride + col64) + 0);
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + dst), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
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
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_0 + (dst + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_1 + (dst + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_2 + (dst + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(push_peers_3 + (dst + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
            }
        }
        int jr = step - 2;
        if (jr >= 0 && jr < n_mine && (b + jr & 3) == rank) {
            int tok_r = b + jr * grid;
            int sr = s0 + jr;
            int gbr = sr >> 2 & 3;
            long long src = push_gen_elems * (long long)gbr + push_cta;
            int parity = (jr >> 2 & 1) * 128;
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
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + src)) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 4096))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 8192))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 12288))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 16384))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 20480))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 24576))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 28672))) : "memory");
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
            float s8_0[8];
            float part_0[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_0[_pair * 2])[0]), "=f"((&part_0[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[_pair]));
            }
            s8_0[0] = part_0[0];
            s8_0[1] = part_0[1];
            s8_0[2] = part_0[2];
            s8_0[3] = part_0[3];
            s8_0[4] = part_0[4];
            s8_0[5] = part_0[5];
            s8_0[6] = part_0[6];
            s8_0[7] = part_0[7];
            float part_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_1[_pair * 2])[0]), "=f"((&part_1[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[4 + _pair]));
            }
            s8_0[0] = s8_0[0] + part_1[0];
            s8_0[1] = s8_0[1] + part_1[1];
            s8_0[2] = s8_0[2] + part_1[2];
            s8_0[3] = s8_0[3] + part_1[3];
            s8_0[4] = s8_0[4] + part_1[4];
            s8_0[5] = s8_0[5] + part_1[5];
            s8_0[6] = s8_0[6] + part_1[6];
            s8_0[7] = s8_0[7] + part_1[7];
            float part_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_2[_pair * 2])[0]), "=f"((&part_2[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[8 + _pair]));
            }
            s8_0[0] = s8_0[0] + part_2[0];
            s8_0[1] = s8_0[1] + part_2[1];
            s8_0[2] = s8_0[2] + part_2[2];
            s8_0[3] = s8_0[3] + part_2[3];
            s8_0[4] = s8_0[4] + part_2[4];
            s8_0[5] = s8_0[5] + part_2[5];
            s8_0[6] = s8_0[6] + part_2[6];
            s8_0[7] = s8_0[7] + part_2[7];
            float part_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_3[_pair * 2])[0]), "=f"((&part_3[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[12 + _pair]));
            }
            s8_0[0] = s8_0[0] + part_3[0];
            s8_0[1] = s8_0[1] + part_3[1];
            s8_0[2] = s8_0[2] + part_3[2];
            s8_0[3] = s8_0[3] + part_3[3];
            s8_0[4] = s8_0[4] + part_3[4];
            s8_0[5] = s8_0[5] + part_3[5];
            s8_0[6] = s8_0[6] + part_3[6];
            s8_0[7] = s8_0[7] + part_3[7];
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
            float y_0 = ((mode != 0) ? y8_0[0] * gw8[0] : y8_0[0]);
            ssq_0 = __fmaf_rn(y_0, y_0, ssq_0);
            float y_1 = ((mode != 0) ? y8_0[1] * gw8[1] : y8_0[1]);
            ssq_0 = __fmaf_rn(y_1, y_1, ssq_0);
            float y_2 = ((mode != 0) ? y8_0[2] * gw8[2] : y8_0[2]);
            ssq_0 = __fmaf_rn(y_2, y_2, ssq_0);
            float y_3 = ((mode != 0) ? y8_0[3] * gw8[3] : y8_0[3]);
            ssq_0 = __fmaf_rn(y_3, y_3, ssq_0);
            float y_4 = ((mode != 0) ? y8_0[4] * gw8[4] : y8_0[4]);
            ssq_0 = __fmaf_rn(y_4, y_4, ssq_0);
            float y_5 = ((mode != 0) ? y8_0[5] * gw8[5] : y8_0[5]);
            ssq_0 = __fmaf_rn(y_5, y_5, ssq_0);
            float y_6 = ((mode != 0) ? y8_0[6] * gw8[6] : y8_0[6]);
            ssq_0 = __fmaf_rn(y_6, y_6, ssq_0);
            float y_7 = ((mode != 0) ? y8_0[7] * gw8[7] : y8_0[7]);
            ssq_0 = __fmaf_rn(y_7, y_7, ssq_0);
            ssq8[0] = ssq_0;
            {
                auto _gst_22 = push_local + src;
                __builtin_assume(__isGlobal(_gst_22));
                reinterpret_cast<int4*>(_gst_22)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_23 = push_local + (src + 4096);
                __builtin_assume(__isGlobal(_gst_23));
                reinterpret_cast<int4*>(_gst_23)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_24 = push_local + (src + 8192);
                __builtin_assume(__isGlobal(_gst_24));
                reinterpret_cast<int4*>(_gst_24)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_25 = push_local + (src + 12288);
                __builtin_assume(__isGlobal(_gst_25));
                reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float s8_1[8];
            float part_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_4[_pair * 2])[0]), "=f"((&part_4[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[16 + _pair]));
            }
            s8_1[0] = part_4[0];
            s8_1[1] = part_4[1];
            s8_1[2] = part_4[2];
            s8_1[3] = part_4[3];
            s8_1[4] = part_4[4];
            s8_1[5] = part_4[5];
            s8_1[6] = part_4[6];
            s8_1[7] = part_4[7];
            float part_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_5[_pair * 2])[0]), "=f"((&part_5[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[20 + _pair]));
            }
            s8_1[0] = s8_1[0] + part_5[0];
            s8_1[1] = s8_1[1] + part_5[1];
            s8_1[2] = s8_1[2] + part_5[2];
            s8_1[3] = s8_1[3] + part_5[3];
            s8_1[4] = s8_1[4] + part_5[4];
            s8_1[5] = s8_1[5] + part_5[5];
            s8_1[6] = s8_1[6] + part_5[6];
            s8_1[7] = s8_1[7] + part_5[7];
            float part_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_6[_pair * 2])[0]), "=f"((&part_6[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[24 + _pair]));
            }
            s8_1[0] = s8_1[0] + part_6[0];
            s8_1[1] = s8_1[1] + part_6[1];
            s8_1[2] = s8_1[2] + part_6[2];
            s8_1[3] = s8_1[3] + part_6[3];
            s8_1[4] = s8_1[4] + part_6[4];
            s8_1[5] = s8_1[5] + part_6[5];
            s8_1[6] = s8_1[6] + part_6[6];
            s8_1[7] = s8_1[7] + part_6[7];
            float part_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_7[_pair * 2])[0]), "=f"((&part_7[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[28 + _pair]));
            }
            s8_1[0] = s8_1[0] + part_7[0];
            s8_1[1] = s8_1[1] + part_7[1];
            s8_1[2] = s8_1[2] + part_7[2];
            s8_1[3] = s8_1[3] + part_7[3];
            s8_1[4] = s8_1[4] + part_7[4];
            s8_1[5] = s8_1[5] + part_7[5];
            s8_1[6] = s8_1[6] + part_7[6];
            s8_1[7] = s8_1[7] + part_7[7];
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
            float y_8 = ((mode != 0) ? y8_1[0] * gw8[0] : y8_1[0]);
            ssq_1 = __fmaf_rn(y_8, y_8, ssq_1);
            float y_9 = ((mode != 0) ? y8_1[1] * gw8[1] : y8_1[1]);
            ssq_1 = __fmaf_rn(y_9, y_9, ssq_1);
            float y_10 = ((mode != 0) ? y8_1[2] * gw8[2] : y8_1[2]);
            ssq_1 = __fmaf_rn(y_10, y_10, ssq_1);
            float y_11 = ((mode != 0) ? y8_1[3] * gw8[3] : y8_1[3]);
            ssq_1 = __fmaf_rn(y_11, y_11, ssq_1);
            float y_12 = ((mode != 0) ? y8_1[4] * gw8[4] : y8_1[4]);
            ssq_1 = __fmaf_rn(y_12, y_12, ssq_1);
            float y_13 = ((mode != 0) ? y8_1[5] * gw8[5] : y8_1[5]);
            ssq_1 = __fmaf_rn(y_13, y_13, ssq_1);
            float y_14 = ((mode != 0) ? y8_1[6] * gw8[6] : y8_1[6]);
            ssq_1 = __fmaf_rn(y_14, y_14, ssq_1);
            float y_15 = ((mode != 0) ? y8_1[7] * gw8[7] : y8_1[7]);
            ssq_1 = __fmaf_rn(y_15, y_15, ssq_1);
            ssq8[1] = ssq_1;
            {
                auto _gst_26 = push_local + (src + 16384);
                __builtin_assume(__isGlobal(_gst_26));
                reinterpret_cast<int4*>(_gst_26)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_27 = push_local + (src + 20480);
                __builtin_assume(__isGlobal(_gst_27));
                reinterpret_cast<int4*>(_gst_27)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_28 = push_local + (src + 24576);
                __builtin_assume(__isGlobal(_gst_28));
                reinterpret_cast<int4*>(_gst_28)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_29 = push_local + (src + 28672);
                __builtin_assume(__isGlobal(_gst_29));
                reinterpret_cast<int4*>(_gst_29)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            uint32_t _sysv_poll_group_1[32];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 32768))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 36864))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 40960))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 45056))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 49152))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 53248))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 57344))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 61440))) : "memory");
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
                (((_sysv_poll_group_1[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[31] >> 16) & 0xffffu) == 0x8000u)
            );
            float s8_2[8];
            float part_8[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_8[_pair * 2])[0]), "=f"((&part_8[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[_pair]));
            }
            s8_2[0] = part_8[0];
            s8_2[1] = part_8[1];
            s8_2[2] = part_8[2];
            s8_2[3] = part_8[3];
            s8_2[4] = part_8[4];
            s8_2[5] = part_8[5];
            s8_2[6] = part_8[6];
            s8_2[7] = part_8[7];
            float part_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_9[_pair * 2])[0]), "=f"((&part_9[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[4 + _pair]));
            }
            s8_2[0] = s8_2[0] + part_9[0];
            s8_2[1] = s8_2[1] + part_9[1];
            s8_2[2] = s8_2[2] + part_9[2];
            s8_2[3] = s8_2[3] + part_9[3];
            s8_2[4] = s8_2[4] + part_9[4];
            s8_2[5] = s8_2[5] + part_9[5];
            s8_2[6] = s8_2[6] + part_9[6];
            s8_2[7] = s8_2[7] + part_9[7];
            float part_10[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_10[_pair * 2])[0]), "=f"((&part_10[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[8 + _pair]));
            }
            s8_2[0] = s8_2[0] + part_10[0];
            s8_2[1] = s8_2[1] + part_10[1];
            s8_2[2] = s8_2[2] + part_10[2];
            s8_2[3] = s8_2[3] + part_10[3];
            s8_2[4] = s8_2[4] + part_10[4];
            s8_2[5] = s8_2[5] + part_10[5];
            s8_2[6] = s8_2[6] + part_10[6];
            s8_2[7] = s8_2[7] + part_10[7];
            float part_11[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_11[_pair * 2])[0]), "=f"((&part_11[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[12 + _pair]));
            }
            s8_2[0] = s8_2[0] + part_11[0];
            s8_2[1] = s8_2[1] + part_11[1];
            s8_2[2] = s8_2[2] + part_11[2];
            s8_2[3] = s8_2[3] + part_11[3];
            s8_2[4] = s8_2[4] + part_11[4];
            s8_2[5] = s8_2[5] + part_11[5];
            s8_2[6] = s8_2[6] + part_11[6];
            s8_2[7] = s8_2[7] + part_11[7];
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
            float y_16 = ((mode != 0) ? y8_2[0] * gw8[0] : y8_2[0]);
            ssq_2 = __fmaf_rn(y_16, y_16, ssq_2);
            float y_17 = ((mode != 0) ? y8_2[1] * gw8[1] : y8_2[1]);
            ssq_2 = __fmaf_rn(y_17, y_17, ssq_2);
            float y_18 = ((mode != 0) ? y8_2[2] * gw8[2] : y8_2[2]);
            ssq_2 = __fmaf_rn(y_18, y_18, ssq_2);
            float y_19 = ((mode != 0) ? y8_2[3] * gw8[3] : y8_2[3]);
            ssq_2 = __fmaf_rn(y_19, y_19, ssq_2);
            float y_20 = ((mode != 0) ? y8_2[4] * gw8[4] : y8_2[4]);
            ssq_2 = __fmaf_rn(y_20, y_20, ssq_2);
            float y_21 = ((mode != 0) ? y8_2[5] * gw8[5] : y8_2[5]);
            ssq_2 = __fmaf_rn(y_21, y_21, ssq_2);
            float y_22 = ((mode != 0) ? y8_2[6] * gw8[6] : y8_2[6]);
            ssq_2 = __fmaf_rn(y_22, y_22, ssq_2);
            float y_23 = ((mode != 0) ? y8_2[7] * gw8[7] : y8_2[7]);
            ssq_2 = __fmaf_rn(y_23, y_23, ssq_2);
            ssq8[2] = ssq_2;
            {
                auto _gst_30 = push_local + (src + 32768);
                __builtin_assume(__isGlobal(_gst_30));
                reinterpret_cast<int4*>(_gst_30)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_31 = push_local + (src + 36864);
                __builtin_assume(__isGlobal(_gst_31));
                reinterpret_cast<int4*>(_gst_31)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_32 = push_local + (src + 40960);
                __builtin_assume(__isGlobal(_gst_32));
                reinterpret_cast<int4*>(_gst_32)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_33 = push_local + (src + 45056);
                __builtin_assume(__isGlobal(_gst_33));
                reinterpret_cast<int4*>(_gst_33)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float s8_3[8];
            float part_12[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_12[_pair * 2])[0]), "=f"((&part_12[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[16 + _pair]));
            }
            s8_3[0] = part_12[0];
            s8_3[1] = part_12[1];
            s8_3[2] = part_12[2];
            s8_3[3] = part_12[3];
            s8_3[4] = part_12[4];
            s8_3[5] = part_12[5];
            s8_3[6] = part_12[6];
            s8_3[7] = part_12[7];
            float part_13[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_13[_pair * 2])[0]), "=f"((&part_13[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[20 + _pair]));
            }
            s8_3[0] = s8_3[0] + part_13[0];
            s8_3[1] = s8_3[1] + part_13[1];
            s8_3[2] = s8_3[2] + part_13[2];
            s8_3[3] = s8_3[3] + part_13[3];
            s8_3[4] = s8_3[4] + part_13[4];
            s8_3[5] = s8_3[5] + part_13[5];
            s8_3[6] = s8_3[6] + part_13[6];
            s8_3[7] = s8_3[7] + part_13[7];
            float part_14[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_14[_pair * 2])[0]), "=f"((&part_14[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[24 + _pair]));
            }
            s8_3[0] = s8_3[0] + part_14[0];
            s8_3[1] = s8_3[1] + part_14[1];
            s8_3[2] = s8_3[2] + part_14[2];
            s8_3[3] = s8_3[3] + part_14[3];
            s8_3[4] = s8_3[4] + part_14[4];
            s8_3[5] = s8_3[5] + part_14[5];
            s8_3[6] = s8_3[6] + part_14[6];
            s8_3[7] = s8_3[7] + part_14[7];
            float part_15[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_15[_pair * 2])[0]), "=f"((&part_15[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[28 + _pair]));
            }
            s8_3[0] = s8_3[0] + part_15[0];
            s8_3[1] = s8_3[1] + part_15[1];
            s8_3[2] = s8_3[2] + part_15[2];
            s8_3[3] = s8_3[3] + part_15[3];
            s8_3[4] = s8_3[4] + part_15[4];
            s8_3[5] = s8_3[5] + part_15[5];
            s8_3[6] = s8_3[6] + part_15[6];
            s8_3[7] = s8_3[7] + part_15[7];
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
            float y_24 = ((mode != 0) ? y8_3[0] * gw8[0] : y8_3[0]);
            ssq_3 = __fmaf_rn(y_24, y_24, ssq_3);
            float y_25 = ((mode != 0) ? y8_3[1] * gw8[1] : y8_3[1]);
            ssq_3 = __fmaf_rn(y_25, y_25, ssq_3);
            float y_26 = ((mode != 0) ? y8_3[2] * gw8[2] : y8_3[2]);
            ssq_3 = __fmaf_rn(y_26, y_26, ssq_3);
            float y_27 = ((mode != 0) ? y8_3[3] * gw8[3] : y8_3[3]);
            ssq_3 = __fmaf_rn(y_27, y_27, ssq_3);
            float y_28 = ((mode != 0) ? y8_3[4] * gw8[4] : y8_3[4]);
            ssq_3 = __fmaf_rn(y_28, y_28, ssq_3);
            float y_29 = ((mode != 0) ? y8_3[5] * gw8[5] : y8_3[5]);
            ssq_3 = __fmaf_rn(y_29, y_29, ssq_3);
            float y_30 = ((mode != 0) ? y8_3[6] * gw8[6] : y8_3[6]);
            ssq_3 = __fmaf_rn(y_30, y_30, ssq_3);
            float y_31 = ((mode != 0) ? y8_3[7] * gw8[7] : y8_3[7]);
            ssq_3 = __fmaf_rn(y_31, y_31, ssq_3);
            ssq8[3] = ssq_3;
            {
                auto _gst_34 = push_local + (src + 49152);
                __builtin_assume(__isGlobal(_gst_34));
                reinterpret_cast<int4*>(_gst_34)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_35 = push_local + (src + 53248);
                __builtin_assume(__isGlobal(_gst_35));
                reinterpret_cast<int4*>(_gst_35)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_36 = push_local + (src + 57344);
                __builtin_assume(__isGlobal(_gst_36));
                reinterpret_cast<int4*>(_gst_36)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_37 = push_local + (src + 61440);
                __builtin_assume(__isGlobal(_gst_37));
                reinterpret_cast<int4*>(_gst_37)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            uint32_t _sysv_poll_group_2[32];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[0]), "=r"(_sysv_poll_group_2[1]), "=r"(_sysv_poll_group_2[2]), "=r"(_sysv_poll_group_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 65536))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[4]), "=r"(_sysv_poll_group_2[5]), "=r"(_sysv_poll_group_2[6]), "=r"(_sysv_poll_group_2[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 69632))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[8]), "=r"(_sysv_poll_group_2[9]), "=r"(_sysv_poll_group_2[10]), "=r"(_sysv_poll_group_2[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 73728))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[12]), "=r"(_sysv_poll_group_2[13]), "=r"(_sysv_poll_group_2[14]), "=r"(_sysv_poll_group_2[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 77824))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[16]), "=r"(_sysv_poll_group_2[17]), "=r"(_sysv_poll_group_2[18]), "=r"(_sysv_poll_group_2[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 81920))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[20]), "=r"(_sysv_poll_group_2[21]), "=r"(_sysv_poll_group_2[22]), "=r"(_sysv_poll_group_2[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 86016))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[24]), "=r"(_sysv_poll_group_2[25]), "=r"(_sysv_poll_group_2[26]), "=r"(_sysv_poll_group_2[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 90112))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[28]), "=r"(_sysv_poll_group_2[29]), "=r"(_sysv_poll_group_2[30]), "=r"(_sysv_poll_group_2[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 94208))) : "memory");
            } while (
                (((_sysv_poll_group_2[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[7] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[9] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[11] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[13] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[15] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[17] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[19] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[21] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[23] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[25] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[27] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[29] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_2[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_2[31] >> 16) & 0xffffu) == 0x8000u)
            );
            float s8_4[8];
            float part_16[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_16[_pair * 2])[0]), "=f"((&part_16[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[_pair]));
            }
            s8_4[0] = part_16[0];
            s8_4[1] = part_16[1];
            s8_4[2] = part_16[2];
            s8_4[3] = part_16[3];
            s8_4[4] = part_16[4];
            s8_4[5] = part_16[5];
            s8_4[6] = part_16[6];
            s8_4[7] = part_16[7];
            float part_17[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_17[_pair * 2])[0]), "=f"((&part_17[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[4 + _pair]));
            }
            s8_4[0] = s8_4[0] + part_17[0];
            s8_4[1] = s8_4[1] + part_17[1];
            s8_4[2] = s8_4[2] + part_17[2];
            s8_4[3] = s8_4[3] + part_17[3];
            s8_4[4] = s8_4[4] + part_17[4];
            s8_4[5] = s8_4[5] + part_17[5];
            s8_4[6] = s8_4[6] + part_17[6];
            s8_4[7] = s8_4[7] + part_17[7];
            float part_18[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_18[_pair * 2])[0]), "=f"((&part_18[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[8 + _pair]));
            }
            s8_4[0] = s8_4[0] + part_18[0];
            s8_4[1] = s8_4[1] + part_18[1];
            s8_4[2] = s8_4[2] + part_18[2];
            s8_4[3] = s8_4[3] + part_18[3];
            s8_4[4] = s8_4[4] + part_18[4];
            s8_4[5] = s8_4[5] + part_18[5];
            s8_4[6] = s8_4[6] + part_18[6];
            s8_4[7] = s8_4[7] + part_18[7];
            float part_19[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_19[_pair * 2])[0]), "=f"((&part_19[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[12 + _pair]));
            }
            s8_4[0] = s8_4[0] + part_19[0];
            s8_4[1] = s8_4[1] + part_19[1];
            s8_4[2] = s8_4[2] + part_19[2];
            s8_4[3] = s8_4[3] + part_19[3];
            s8_4[4] = s8_4[4] + part_19[4];
            s8_4[5] = s8_4[5] + part_19[5];
            s8_4[6] = s8_4[6] + part_19[6];
            s8_4[7] = s8_4[7] + part_19[7];
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
            float y_32 = ((mode != 0) ? y8_4[0] * gw8[0] : y8_4[0]);
            ssq_4 = __fmaf_rn(y_32, y_32, ssq_4);
            float y_33 = ((mode != 0) ? y8_4[1] * gw8[1] : y8_4[1]);
            ssq_4 = __fmaf_rn(y_33, y_33, ssq_4);
            float y_34 = ((mode != 0) ? y8_4[2] * gw8[2] : y8_4[2]);
            ssq_4 = __fmaf_rn(y_34, y_34, ssq_4);
            float y_35 = ((mode != 0) ? y8_4[3] * gw8[3] : y8_4[3]);
            ssq_4 = __fmaf_rn(y_35, y_35, ssq_4);
            float y_36 = ((mode != 0) ? y8_4[4] * gw8[4] : y8_4[4]);
            ssq_4 = __fmaf_rn(y_36, y_36, ssq_4);
            float y_37 = ((mode != 0) ? y8_4[5] * gw8[5] : y8_4[5]);
            ssq_4 = __fmaf_rn(y_37, y_37, ssq_4);
            float y_38 = ((mode != 0) ? y8_4[6] * gw8[6] : y8_4[6]);
            ssq_4 = __fmaf_rn(y_38, y_38, ssq_4);
            float y_39 = ((mode != 0) ? y8_4[7] * gw8[7] : y8_4[7]);
            ssq_4 = __fmaf_rn(y_39, y_39, ssq_4);
            ssq8[4] = ssq_4;
            {
                auto _gst_38 = push_local + (src + 65536);
                __builtin_assume(__isGlobal(_gst_38));
                reinterpret_cast<int4*>(_gst_38)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_39 = push_local + (src + 69632);
                __builtin_assume(__isGlobal(_gst_39));
                reinterpret_cast<int4*>(_gst_39)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_40 = push_local + (src + 73728);
                __builtin_assume(__isGlobal(_gst_40));
                reinterpret_cast<int4*>(_gst_40)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_41 = push_local + (src + 77824);
                __builtin_assume(__isGlobal(_gst_41));
                reinterpret_cast<int4*>(_gst_41)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float s8_5[8];
            float part_20[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_20[_pair * 2])[0]), "=f"((&part_20[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[16 + _pair]));
            }
            s8_5[0] = part_20[0];
            s8_5[1] = part_20[1];
            s8_5[2] = part_20[2];
            s8_5[3] = part_20[3];
            s8_5[4] = part_20[4];
            s8_5[5] = part_20[5];
            s8_5[6] = part_20[6];
            s8_5[7] = part_20[7];
            float part_21[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_21[_pair * 2])[0]), "=f"((&part_21[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[20 + _pair]));
            }
            s8_5[0] = s8_5[0] + part_21[0];
            s8_5[1] = s8_5[1] + part_21[1];
            s8_5[2] = s8_5[2] + part_21[2];
            s8_5[3] = s8_5[3] + part_21[3];
            s8_5[4] = s8_5[4] + part_21[4];
            s8_5[5] = s8_5[5] + part_21[5];
            s8_5[6] = s8_5[6] + part_21[6];
            s8_5[7] = s8_5[7] + part_21[7];
            float part_22[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_22[_pair * 2])[0]), "=f"((&part_22[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[24 + _pair]));
            }
            s8_5[0] = s8_5[0] + part_22[0];
            s8_5[1] = s8_5[1] + part_22[1];
            s8_5[2] = s8_5[2] + part_22[2];
            s8_5[3] = s8_5[3] + part_22[3];
            s8_5[4] = s8_5[4] + part_22[4];
            s8_5[5] = s8_5[5] + part_22[5];
            s8_5[6] = s8_5[6] + part_22[6];
            s8_5[7] = s8_5[7] + part_22[7];
            float part_23[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_23[_pair * 2])[0]), "=f"((&part_23[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[28 + _pair]));
            }
            s8_5[0] = s8_5[0] + part_23[0];
            s8_5[1] = s8_5[1] + part_23[1];
            s8_5[2] = s8_5[2] + part_23[2];
            s8_5[3] = s8_5[3] + part_23[3];
            s8_5[4] = s8_5[4] + part_23[4];
            s8_5[5] = s8_5[5] + part_23[5];
            s8_5[6] = s8_5[6] + part_23[6];
            s8_5[7] = s8_5[7] + part_23[7];
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
            float y_40 = ((mode != 0) ? y8_5[0] * gw8[0] : y8_5[0]);
            ssq_5 = __fmaf_rn(y_40, y_40, ssq_5);
            float y_41 = ((mode != 0) ? y8_5[1] * gw8[1] : y8_5[1]);
            ssq_5 = __fmaf_rn(y_41, y_41, ssq_5);
            float y_42 = ((mode != 0) ? y8_5[2] * gw8[2] : y8_5[2]);
            ssq_5 = __fmaf_rn(y_42, y_42, ssq_5);
            float y_43 = ((mode != 0) ? y8_5[3] * gw8[3] : y8_5[3]);
            ssq_5 = __fmaf_rn(y_43, y_43, ssq_5);
            float y_44 = ((mode != 0) ? y8_5[4] * gw8[4] : y8_5[4]);
            ssq_5 = __fmaf_rn(y_44, y_44, ssq_5);
            float y_45 = ((mode != 0) ? y8_5[5] * gw8[5] : y8_5[5]);
            ssq_5 = __fmaf_rn(y_45, y_45, ssq_5);
            float y_46 = ((mode != 0) ? y8_5[6] * gw8[6] : y8_5[6]);
            ssq_5 = __fmaf_rn(y_46, y_46, ssq_5);
            float y_47 = ((mode != 0) ? y8_5[7] * gw8[7] : y8_5[7]);
            ssq_5 = __fmaf_rn(y_47, y_47, ssq_5);
            ssq8[5] = ssq_5;
            {
                auto _gst_42 = push_local + (src + 81920);
                __builtin_assume(__isGlobal(_gst_42));
                reinterpret_cast<int4*>(_gst_42)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_43 = push_local + (src + 86016);
                __builtin_assume(__isGlobal(_gst_43));
                reinterpret_cast<int4*>(_gst_43)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_44 = push_local + (src + 90112);
                __builtin_assume(__isGlobal(_gst_44));
                reinterpret_cast<int4*>(_gst_44)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_45 = push_local + (src + 94208);
                __builtin_assume(__isGlobal(_gst_45));
                reinterpret_cast<int4*>(_gst_45)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            uint32_t _sysv_poll_group_3[32];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[0]), "=r"(_sysv_poll_group_3[1]), "=r"(_sysv_poll_group_3[2]), "=r"(_sysv_poll_group_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 98304))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[4]), "=r"(_sysv_poll_group_3[5]), "=r"(_sysv_poll_group_3[6]), "=r"(_sysv_poll_group_3[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 102400))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[8]), "=r"(_sysv_poll_group_3[9]), "=r"(_sysv_poll_group_3[10]), "=r"(_sysv_poll_group_3[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 106496))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[12]), "=r"(_sysv_poll_group_3[13]), "=r"(_sysv_poll_group_3[14]), "=r"(_sysv_poll_group_3[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 110592))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[16]), "=r"(_sysv_poll_group_3[17]), "=r"(_sysv_poll_group_3[18]), "=r"(_sysv_poll_group_3[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 114688))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[20]), "=r"(_sysv_poll_group_3[21]), "=r"(_sysv_poll_group_3[22]), "=r"(_sysv_poll_group_3[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 118784))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[24]), "=r"(_sysv_poll_group_3[25]), "=r"(_sysv_poll_group_3[26]), "=r"(_sysv_poll_group_3[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 122880))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[28]), "=r"(_sysv_poll_group_3[29]), "=r"(_sysv_poll_group_3[30]), "=r"(_sysv_poll_group_3[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 126976))) : "memory");
            } while (
                (((_sysv_poll_group_3[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[7] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[9] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[11] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[13] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[15] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[17] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[19] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[21] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[23] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[25] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[27] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[29] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_3[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_3[31] >> 16) & 0xffffu) == 0x8000u)
            );
            float s8_6[8];
            float part_24[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_24[_pair * 2])[0]), "=f"((&part_24[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[_pair]));
            }
            s8_6[0] = part_24[0];
            s8_6[1] = part_24[1];
            s8_6[2] = part_24[2];
            s8_6[3] = part_24[3];
            s8_6[4] = part_24[4];
            s8_6[5] = part_24[5];
            s8_6[6] = part_24[6];
            s8_6[7] = part_24[7];
            float part_25[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_25[_pair * 2])[0]), "=f"((&part_25[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[4 + _pair]));
            }
            s8_6[0] = s8_6[0] + part_25[0];
            s8_6[1] = s8_6[1] + part_25[1];
            s8_6[2] = s8_6[2] + part_25[2];
            s8_6[3] = s8_6[3] + part_25[3];
            s8_6[4] = s8_6[4] + part_25[4];
            s8_6[5] = s8_6[5] + part_25[5];
            s8_6[6] = s8_6[6] + part_25[6];
            s8_6[7] = s8_6[7] + part_25[7];
            float part_26[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_26[_pair * 2])[0]), "=f"((&part_26[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[8 + _pair]));
            }
            s8_6[0] = s8_6[0] + part_26[0];
            s8_6[1] = s8_6[1] + part_26[1];
            s8_6[2] = s8_6[2] + part_26[2];
            s8_6[3] = s8_6[3] + part_26[3];
            s8_6[4] = s8_6[4] + part_26[4];
            s8_6[5] = s8_6[5] + part_26[5];
            s8_6[6] = s8_6[6] + part_26[6];
            s8_6[7] = s8_6[7] + part_26[7];
            float part_27[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_27[_pair * 2])[0]), "=f"((&part_27[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[12 + _pair]));
            }
            s8_6[0] = s8_6[0] + part_27[0];
            s8_6[1] = s8_6[1] + part_27[1];
            s8_6[2] = s8_6[2] + part_27[2];
            s8_6[3] = s8_6[3] + part_27[3];
            s8_6[4] = s8_6[4] + part_27[4];
            s8_6[5] = s8_6[5] + part_27[5];
            s8_6[6] = s8_6[6] + part_27[6];
            s8_6[7] = s8_6[7] + part_27[7];
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
            float y_48 = ((mode != 0) ? y8_6[0] * gw8[0] : y8_6[0]);
            ssq_6 = __fmaf_rn(y_48, y_48, ssq_6);
            float y_49 = ((mode != 0) ? y8_6[1] * gw8[1] : y8_6[1]);
            ssq_6 = __fmaf_rn(y_49, y_49, ssq_6);
            float y_50 = ((mode != 0) ? y8_6[2] * gw8[2] : y8_6[2]);
            ssq_6 = __fmaf_rn(y_50, y_50, ssq_6);
            float y_51 = ((mode != 0) ? y8_6[3] * gw8[3] : y8_6[3]);
            ssq_6 = __fmaf_rn(y_51, y_51, ssq_6);
            float y_52 = ((mode != 0) ? y8_6[4] * gw8[4] : y8_6[4]);
            ssq_6 = __fmaf_rn(y_52, y_52, ssq_6);
            float y_53 = ((mode != 0) ? y8_6[5] * gw8[5] : y8_6[5]);
            ssq_6 = __fmaf_rn(y_53, y_53, ssq_6);
            float y_54 = ((mode != 0) ? y8_6[6] * gw8[6] : y8_6[6]);
            ssq_6 = __fmaf_rn(y_54, y_54, ssq_6);
            float y_55 = ((mode != 0) ? y8_6[7] * gw8[7] : y8_6[7]);
            ssq_6 = __fmaf_rn(y_55, y_55, ssq_6);
            ssq8[6] = ssq_6;
            {
                auto _gst_46 = push_local + (src + 98304);
                __builtin_assume(__isGlobal(_gst_46));
                reinterpret_cast<int4*>(_gst_46)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_47 = push_local + (src + 102400);
                __builtin_assume(__isGlobal(_gst_47));
                reinterpret_cast<int4*>(_gst_47)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_48 = push_local + (src + 106496);
                __builtin_assume(__isGlobal(_gst_48));
                reinterpret_cast<int4*>(_gst_48)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_49 = push_local + (src + 110592);
                __builtin_assume(__isGlobal(_gst_49));
                reinterpret_cast<int4*>(_gst_49)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float s8_7[8];
            float part_28[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_28[_pair * 2])[0]), "=f"((&part_28[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[16 + _pair]));
            }
            s8_7[0] = part_28[0];
            s8_7[1] = part_28[1];
            s8_7[2] = part_28[2];
            s8_7[3] = part_28[3];
            s8_7[4] = part_28[4];
            s8_7[5] = part_28[5];
            s8_7[6] = part_28[6];
            s8_7[7] = part_28[7];
            float part_29[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_29[_pair * 2])[0]), "=f"((&part_29[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[20 + _pair]));
            }
            s8_7[0] = s8_7[0] + part_29[0];
            s8_7[1] = s8_7[1] + part_29[1];
            s8_7[2] = s8_7[2] + part_29[2];
            s8_7[3] = s8_7[3] + part_29[3];
            s8_7[4] = s8_7[4] + part_29[4];
            s8_7[5] = s8_7[5] + part_29[5];
            s8_7[6] = s8_7[6] + part_29[6];
            s8_7[7] = s8_7[7] + part_29[7];
            float part_30[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_30[_pair * 2])[0]), "=f"((&part_30[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[24 + _pair]));
            }
            s8_7[0] = s8_7[0] + part_30[0];
            s8_7[1] = s8_7[1] + part_30[1];
            s8_7[2] = s8_7[2] + part_30[2];
            s8_7[3] = s8_7[3] + part_30[3];
            s8_7[4] = s8_7[4] + part_30[4];
            s8_7[5] = s8_7[5] + part_30[5];
            s8_7[6] = s8_7[6] + part_30[6];
            s8_7[7] = s8_7[7] + part_30[7];
            float part_31[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_31[_pair * 2])[0]), "=f"((&part_31[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[28 + _pair]));
            }
            s8_7[0] = s8_7[0] + part_31[0];
            s8_7[1] = s8_7[1] + part_31[1];
            s8_7[2] = s8_7[2] + part_31[2];
            s8_7[3] = s8_7[3] + part_31[3];
            s8_7[4] = s8_7[4] + part_31[4];
            s8_7[5] = s8_7[5] + part_31[5];
            s8_7[6] = s8_7[6] + part_31[6];
            s8_7[7] = s8_7[7] + part_31[7];
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
            float y8_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&y8_7[_pair * 2])[0]), "=f"((&y8_7[_pair * 2])[1])
                    : "r"(s8_bf16_7[_pair]));
            }
            float ssq_7 = 0.0f;
            float y_56 = ((mode != 0) ? y8_7[0] * gw8[0] : y8_7[0]);
            ssq_7 = __fmaf_rn(y_56, y_56, ssq_7);
            float y_57 = ((mode != 0) ? y8_7[1] * gw8[1] : y8_7[1]);
            ssq_7 = __fmaf_rn(y_57, y_57, ssq_7);
            float y_58 = ((mode != 0) ? y8_7[2] * gw8[2] : y8_7[2]);
            ssq_7 = __fmaf_rn(y_58, y_58, ssq_7);
            float y_59 = ((mode != 0) ? y8_7[3] * gw8[3] : y8_7[3]);
            ssq_7 = __fmaf_rn(y_59, y_59, ssq_7);
            float y_60 = ((mode != 0) ? y8_7[4] * gw8[4] : y8_7[4]);
            ssq_7 = __fmaf_rn(y_60, y_60, ssq_7);
            float y_61 = ((mode != 0) ? y8_7[5] * gw8[5] : y8_7[5]);
            ssq_7 = __fmaf_rn(y_61, y_61, ssq_7);
            float y_62 = ((mode != 0) ? y8_7[6] * gw8[6] : y8_7[6]);
            ssq_7 = __fmaf_rn(y_62, y_62, ssq_7);
            float y_63 = ((mode != 0) ? y8_7[7] * gw8[7] : y8_7[7]);
            ssq_7 = __fmaf_rn(y_63, y_63, ssq_7);
            ssq8[7] = ssq_7;
            {
                auto _gst_50 = push_local + (src + 114688);
                __builtin_assume(__isGlobal(_gst_50));
                reinterpret_cast<int4*>(_gst_50)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_51 = push_local + (src + 118784);
                __builtin_assume(__isGlobal(_gst_51));
                reinterpret_cast<int4*>(_gst_51)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_52 = push_local + (src + 122880);
                __builtin_assume(__isGlobal(_gst_52));
                reinterpret_cast<int4*>(_gst_52)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_53 = push_local + (src + 126976);
                __builtin_assume(__isGlobal(_gst_53));
                reinterpret_cast<int4*>(_gst_53)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            float _warp_reduce_0 = ssq8[0];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8] = _warp_reduce_0;
            }
            float _warp_reduce_1 = ssq8[1];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 1] = _warp_reduce_1;
            }
            float _warp_reduce_2 = ssq8[2];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 2] = _warp_reduce_2;
            }
            float _warp_reduce_3 = ssq8[3];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 3] = _warp_reduce_3;
            }
            float _warp_reduce_4 = ssq8[4];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 4] = _warp_reduce_4;
            }
            float _warp_reduce_5 = ssq8[5];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 5] = _warp_reduce_5;
            }
            float _warp_reduce_6 = ssq8[6];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_6 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 6] = _warp_reduce_6;
            }
            float _warp_reduce_7 = ssq8[7];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 7] = _warp_reduce_7;
            }
            __syncthreads();
            float acc8[8];
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            float tot_0 = warp_tab[parity];
            tot_0 = tot_0 + warp_tab[parity + 8];
            tot_0 = tot_0 + warp_tab[parity + 16];
            tot_0 = tot_0 + warp_tab[parity + 24];
            tot_0 = tot_0 + warp_tab[parity + 32];
            tot_0 = tot_0 + warp_tab[parity + 40];
            tot_0 = tot_0 + warp_tab[parity + 48];
            tot_0 = tot_0 + warp_tab[parity + 56];
            tot_0 = tot_0 + warp_tab[parity + 64];
            tot_0 = tot_0 + warp_tab[parity + 72];
            tot_0 = tot_0 + warp_tab[parity + 80];
            tot_0 = tot_0 + warp_tab[parity + 88];
            tot_0 = tot_0 + warp_tab[parity + 96];
            tot_0 = tot_0 + warp_tab[parity + 104];
            tot_0 = tot_0 + warp_tab[parity + 112];
            tot_0 = tot_0 + warp_tab[parity + 120];
            float _sqrt_54;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_54) : "f"(__fmaf_rn(tot_0, 0.00024414062f, eps)));
            float rs_0 = __frcp_rn(_sqrt_54);
            float yn8_0[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_0[_pair * 2])[0]), "=f"((&yn8_0[_pair * 2])[1])
                    : "r"(rw[_pair]));
            }
            float yn_0 = ((mode != 0) ? yn8_0[0] * gw8[0] : yn8_0[0]);
            float o_0 = yn_0 * rs_0;
            float o2_0 = ((mode != 0) ? o_0 : o_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_0, w8[0], acc8[0]);
            float yn_1 = ((mode != 0) ? yn8_0[1] * gw8[1] : yn8_0[1]);
            float o_1 = yn_1 * rs_0;
            float o2_1 = ((mode != 0) ? o_1 : o_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_1, w8[0], acc8[1]);
            float yn_2 = ((mode != 0) ? yn8_0[2] * gw8[2] : yn8_0[2]);
            float o_2 = yn_2 * rs_0;
            float o2_2 = ((mode != 0) ? o_2 : o_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_2, w8[0], acc8[2]);
            float yn_3 = ((mode != 0) ? yn8_0[3] * gw8[3] : yn8_0[3]);
            float o_3 = yn_3 * rs_0;
            float o2_3 = ((mode != 0) ? o_3 : o_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_3, w8[0], acc8[3]);
            float yn_4 = ((mode != 0) ? yn8_0[4] * gw8[4] : yn8_0[4]);
            float o_4 = yn_4 * rs_0;
            float o2_4 = ((mode != 0) ? o_4 : o_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_4, w8[0], acc8[4]);
            float yn_5 = ((mode != 0) ? yn8_0[5] * gw8[5] : yn8_0[5]);
            float o_5 = yn_5 * rs_0;
            float o2_5 = ((mode != 0) ? o_5 : o_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_5, w8[0], acc8[5]);
            float yn_6 = ((mode != 0) ? yn8_0[6] * gw8[6] : yn8_0[6]);
            float o_6 = yn_6 * rs_0;
            float o2_6 = ((mode != 0) ? o_6 : o_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_6, w8[0], acc8[6]);
            float yn_7 = ((mode != 0) ? yn8_0[7] * gw8[7] : yn8_0[7]);
            float o_7 = yn_7 * rs_0;
            float o2_7 = ((mode != 0) ? o_7 : o_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_7, w8[0], acc8[7]);
            float tot_1 = warp_tab[parity + 1];
            tot_1 = tot_1 + warp_tab[parity + 8 + 1];
            tot_1 = tot_1 + warp_tab[parity + 16 + 1];
            tot_1 = tot_1 + warp_tab[parity + 24 + 1];
            tot_1 = tot_1 + warp_tab[parity + 32 + 1];
            tot_1 = tot_1 + warp_tab[parity + 40 + 1];
            tot_1 = tot_1 + warp_tab[parity + 48 + 1];
            tot_1 = tot_1 + warp_tab[parity + 56 + 1];
            tot_1 = tot_1 + warp_tab[parity + 64 + 1];
            tot_1 = tot_1 + warp_tab[parity + 72 + 1];
            tot_1 = tot_1 + warp_tab[parity + 80 + 1];
            tot_1 = tot_1 + warp_tab[parity + 88 + 1];
            tot_1 = tot_1 + warp_tab[parity + 96 + 1];
            tot_1 = tot_1 + warp_tab[parity + 104 + 1];
            tot_1 = tot_1 + warp_tab[parity + 112 + 1];
            tot_1 = tot_1 + warp_tab[parity + 120 + 1];
            float _sqrt_55;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_55) : "f"(__fmaf_rn(tot_1, 0.00024414062f, eps)));
            float rs_1 = __frcp_rn(_sqrt_55);
            float yn8_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_1[_pair * 2])[0]), "=f"((&yn8_1[_pair * 2])[1])
                    : "r"(rw[4 + _pair]));
            }
            float yn_8 = ((mode != 0) ? yn8_1[0] * gw8[0] : yn8_1[0]);
            float o_8 = yn_8 * rs_1;
            float o2_8 = ((mode != 0) ? o_8 : o_8 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_8, w8[1], acc8[0]);
            float yn_9 = ((mode != 0) ? yn8_1[1] * gw8[1] : yn8_1[1]);
            float o_9 = yn_9 * rs_1;
            float o2_9 = ((mode != 0) ? o_9 : o_9 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_9, w8[1], acc8[1]);
            float yn_10 = ((mode != 0) ? yn8_1[2] * gw8[2] : yn8_1[2]);
            float o_10 = yn_10 * rs_1;
            float o2_10 = ((mode != 0) ? o_10 : o_10 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_10, w8[1], acc8[2]);
            float yn_11 = ((mode != 0) ? yn8_1[3] * gw8[3] : yn8_1[3]);
            float o_11 = yn_11 * rs_1;
            float o2_11 = ((mode != 0) ? o_11 : o_11 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_11, w8[1], acc8[3]);
            float yn_12 = ((mode != 0) ? yn8_1[4] * gw8[4] : yn8_1[4]);
            float o_12 = yn_12 * rs_1;
            float o2_12 = ((mode != 0) ? o_12 : o_12 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_12, w8[1], acc8[4]);
            float yn_13 = ((mode != 0) ? yn8_1[5] * gw8[5] : yn8_1[5]);
            float o_13 = yn_13 * rs_1;
            float o2_13 = ((mode != 0) ? o_13 : o_13 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_13, w8[1], acc8[5]);
            float yn_14 = ((mode != 0) ? yn8_1[6] * gw8[6] : yn8_1[6]);
            float o_14 = yn_14 * rs_1;
            float o2_14 = ((mode != 0) ? o_14 : o_14 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_14, w8[1], acc8[6]);
            float yn_15 = ((mode != 0) ? yn8_1[7] * gw8[7] : yn8_1[7]);
            float o_15 = yn_15 * rs_1;
            float o2_15 = ((mode != 0) ? o_15 : o_15 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_15, w8[1], acc8[7]);
            float tot_2 = warp_tab[parity + 2];
            tot_2 = tot_2 + warp_tab[parity + 8 + 2];
            tot_2 = tot_2 + warp_tab[parity + 16 + 2];
            tot_2 = tot_2 + warp_tab[parity + 24 + 2];
            tot_2 = tot_2 + warp_tab[parity + 32 + 2];
            tot_2 = tot_2 + warp_tab[parity + 40 + 2];
            tot_2 = tot_2 + warp_tab[parity + 48 + 2];
            tot_2 = tot_2 + warp_tab[parity + 56 + 2];
            tot_2 = tot_2 + warp_tab[parity + 64 + 2];
            tot_2 = tot_2 + warp_tab[parity + 72 + 2];
            tot_2 = tot_2 + warp_tab[parity + 80 + 2];
            tot_2 = tot_2 + warp_tab[parity + 88 + 2];
            tot_2 = tot_2 + warp_tab[parity + 96 + 2];
            tot_2 = tot_2 + warp_tab[parity + 104 + 2];
            tot_2 = tot_2 + warp_tab[parity + 112 + 2];
            tot_2 = tot_2 + warp_tab[parity + 120 + 2];
            float _sqrt_56;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_56) : "f"(__fmaf_rn(tot_2, 0.00024414062f, eps)));
            float rs_2 = __frcp_rn(_sqrt_56);
            float yn8_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_2[_pair * 2])[0]), "=f"((&yn8_2[_pair * 2])[1])
                    : "r"(rw[8 + _pair]));
            }
            float yn_16 = ((mode != 0) ? yn8_2[0] * gw8[0] : yn8_2[0]);
            float o_16 = yn_16 * rs_2;
            float o2_16 = ((mode != 0) ? o_16 : o_16 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_16, w8[2], acc8[0]);
            float yn_17 = ((mode != 0) ? yn8_2[1] * gw8[1] : yn8_2[1]);
            float o_17 = yn_17 * rs_2;
            float o2_17 = ((mode != 0) ? o_17 : o_17 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_17, w8[2], acc8[1]);
            float yn_18 = ((mode != 0) ? yn8_2[2] * gw8[2] : yn8_2[2]);
            float o_18 = yn_18 * rs_2;
            float o2_18 = ((mode != 0) ? o_18 : o_18 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_18, w8[2], acc8[2]);
            float yn_19 = ((mode != 0) ? yn8_2[3] * gw8[3] : yn8_2[3]);
            float o_19 = yn_19 * rs_2;
            float o2_19 = ((mode != 0) ? o_19 : o_19 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_19, w8[2], acc8[3]);
            float yn_20 = ((mode != 0) ? yn8_2[4] * gw8[4] : yn8_2[4]);
            float o_20 = yn_20 * rs_2;
            float o2_20 = ((mode != 0) ? o_20 : o_20 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_20, w8[2], acc8[4]);
            float yn_21 = ((mode != 0) ? yn8_2[5] * gw8[5] : yn8_2[5]);
            float o_21 = yn_21 * rs_2;
            float o2_21 = ((mode != 0) ? o_21 : o_21 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_21, w8[2], acc8[5]);
            float yn_22 = ((mode != 0) ? yn8_2[6] * gw8[6] : yn8_2[6]);
            float o_22 = yn_22 * rs_2;
            float o2_22 = ((mode != 0) ? o_22 : o_22 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_22, w8[2], acc8[6]);
            float yn_23 = ((mode != 0) ? yn8_2[7] * gw8[7] : yn8_2[7]);
            float o_23 = yn_23 * rs_2;
            float o2_23 = ((mode != 0) ? o_23 : o_23 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_23, w8[2], acc8[7]);
            float tot_3 = warp_tab[parity + 3];
            tot_3 = tot_3 + warp_tab[parity + 8 + 3];
            tot_3 = tot_3 + warp_tab[parity + 16 + 3];
            tot_3 = tot_3 + warp_tab[parity + 24 + 3];
            tot_3 = tot_3 + warp_tab[parity + 32 + 3];
            tot_3 = tot_3 + warp_tab[parity + 40 + 3];
            tot_3 = tot_3 + warp_tab[parity + 48 + 3];
            tot_3 = tot_3 + warp_tab[parity + 56 + 3];
            tot_3 = tot_3 + warp_tab[parity + 64 + 3];
            tot_3 = tot_3 + warp_tab[parity + 72 + 3];
            tot_3 = tot_3 + warp_tab[parity + 80 + 3];
            tot_3 = tot_3 + warp_tab[parity + 88 + 3];
            tot_3 = tot_3 + warp_tab[parity + 96 + 3];
            tot_3 = tot_3 + warp_tab[parity + 104 + 3];
            tot_3 = tot_3 + warp_tab[parity + 112 + 3];
            tot_3 = tot_3 + warp_tab[parity + 120 + 3];
            float _sqrt_57;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_57) : "f"(__fmaf_rn(tot_3, 0.00024414062f, eps)));
            float rs_3 = __frcp_rn(_sqrt_57);
            float yn8_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_3[_pair * 2])[0]), "=f"((&yn8_3[_pair * 2])[1])
                    : "r"(rw[12 + _pair]));
            }
            float yn_24 = ((mode != 0) ? yn8_3[0] * gw8[0] : yn8_3[0]);
            float o_24 = yn_24 * rs_3;
            float o2_24 = ((mode != 0) ? o_24 : o_24 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_24, w8[3], acc8[0]);
            float yn_25 = ((mode != 0) ? yn8_3[1] * gw8[1] : yn8_3[1]);
            float o_25 = yn_25 * rs_3;
            float o2_25 = ((mode != 0) ? o_25 : o_25 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_25, w8[3], acc8[1]);
            float yn_26 = ((mode != 0) ? yn8_3[2] * gw8[2] : yn8_3[2]);
            float o_26 = yn_26 * rs_3;
            float o2_26 = ((mode != 0) ? o_26 : o_26 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_26, w8[3], acc8[2]);
            float yn_27 = ((mode != 0) ? yn8_3[3] * gw8[3] : yn8_3[3]);
            float o_27 = yn_27 * rs_3;
            float o2_27 = ((mode != 0) ? o_27 : o_27 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_27, w8[3], acc8[3]);
            float yn_28 = ((mode != 0) ? yn8_3[4] * gw8[4] : yn8_3[4]);
            float o_28 = yn_28 * rs_3;
            float o2_28 = ((mode != 0) ? o_28 : o_28 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_28, w8[3], acc8[4]);
            float yn_29 = ((mode != 0) ? yn8_3[5] * gw8[5] : yn8_3[5]);
            float o_29 = yn_29 * rs_3;
            float o2_29 = ((mode != 0) ? o_29 : o_29 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_29, w8[3], acc8[5]);
            float yn_30 = ((mode != 0) ? yn8_3[6] * gw8[6] : yn8_3[6]);
            float o_30 = yn_30 * rs_3;
            float o2_30 = ((mode != 0) ? o_30 : o_30 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_30, w8[3], acc8[6]);
            float yn_31 = ((mode != 0) ? yn8_3[7] * gw8[7] : yn8_3[7]);
            float o_31 = yn_31 * rs_3;
            float o2_31 = ((mode != 0) ? o_31 : o_31 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_31, w8[3], acc8[7]);
            float tot_4 = warp_tab[parity + 4];
            tot_4 = tot_4 + warp_tab[parity + 8 + 4];
            tot_4 = tot_4 + warp_tab[parity + 16 + 4];
            tot_4 = tot_4 + warp_tab[parity + 24 + 4];
            tot_4 = tot_4 + warp_tab[parity + 32 + 4];
            tot_4 = tot_4 + warp_tab[parity + 40 + 4];
            tot_4 = tot_4 + warp_tab[parity + 48 + 4];
            tot_4 = tot_4 + warp_tab[parity + 56 + 4];
            tot_4 = tot_4 + warp_tab[parity + 64 + 4];
            tot_4 = tot_4 + warp_tab[parity + 72 + 4];
            tot_4 = tot_4 + warp_tab[parity + 80 + 4];
            tot_4 = tot_4 + warp_tab[parity + 88 + 4];
            tot_4 = tot_4 + warp_tab[parity + 96 + 4];
            tot_4 = tot_4 + warp_tab[parity + 104 + 4];
            tot_4 = tot_4 + warp_tab[parity + 112 + 4];
            tot_4 = tot_4 + warp_tab[parity + 120 + 4];
            float _sqrt_58;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_58) : "f"(__fmaf_rn(tot_4, 0.00024414062f, eps)));
            float rs_4 = __frcp_rn(_sqrt_58);
            float yn8_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_4[_pair * 2])[0]), "=f"((&yn8_4[_pair * 2])[1])
                    : "r"(rw[16 + _pair]));
            }
            float yn_32 = ((mode != 0) ? yn8_4[0] * gw8[0] : yn8_4[0]);
            float o_32 = yn_32 * rs_4;
            float o2_32 = ((mode != 0) ? o_32 : o_32 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_32, w8[4], acc8[0]);
            float yn_33 = ((mode != 0) ? yn8_4[1] * gw8[1] : yn8_4[1]);
            float o_33 = yn_33 * rs_4;
            float o2_33 = ((mode != 0) ? o_33 : o_33 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_33, w8[4], acc8[1]);
            float yn_34 = ((mode != 0) ? yn8_4[2] * gw8[2] : yn8_4[2]);
            float o_34 = yn_34 * rs_4;
            float o2_34 = ((mode != 0) ? o_34 : o_34 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_34, w8[4], acc8[2]);
            float yn_35 = ((mode != 0) ? yn8_4[3] * gw8[3] : yn8_4[3]);
            float o_35 = yn_35 * rs_4;
            float o2_35 = ((mode != 0) ? o_35 : o_35 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_35, w8[4], acc8[3]);
            float yn_36 = ((mode != 0) ? yn8_4[4] * gw8[4] : yn8_4[4]);
            float o_36 = yn_36 * rs_4;
            float o2_36 = ((mode != 0) ? o_36 : o_36 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_36, w8[4], acc8[4]);
            float yn_37 = ((mode != 0) ? yn8_4[5] * gw8[5] : yn8_4[5]);
            float o_37 = yn_37 * rs_4;
            float o2_37 = ((mode != 0) ? o_37 : o_37 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_37, w8[4], acc8[5]);
            float yn_38 = ((mode != 0) ? yn8_4[6] * gw8[6] : yn8_4[6]);
            float o_38 = yn_38 * rs_4;
            float o2_38 = ((mode != 0) ? o_38 : o_38 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_38, w8[4], acc8[6]);
            float yn_39 = ((mode != 0) ? yn8_4[7] * gw8[7] : yn8_4[7]);
            float o_39 = yn_39 * rs_4;
            float o2_39 = ((mode != 0) ? o_39 : o_39 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_39, w8[4], acc8[7]);
            float tot_5 = warp_tab[parity + 5];
            tot_5 = tot_5 + warp_tab[parity + 8 + 5];
            tot_5 = tot_5 + warp_tab[parity + 16 + 5];
            tot_5 = tot_5 + warp_tab[parity + 24 + 5];
            tot_5 = tot_5 + warp_tab[parity + 32 + 5];
            tot_5 = tot_5 + warp_tab[parity + 40 + 5];
            tot_5 = tot_5 + warp_tab[parity + 48 + 5];
            tot_5 = tot_5 + warp_tab[parity + 56 + 5];
            tot_5 = tot_5 + warp_tab[parity + 64 + 5];
            tot_5 = tot_5 + warp_tab[parity + 72 + 5];
            tot_5 = tot_5 + warp_tab[parity + 80 + 5];
            tot_5 = tot_5 + warp_tab[parity + 88 + 5];
            tot_5 = tot_5 + warp_tab[parity + 96 + 5];
            tot_5 = tot_5 + warp_tab[parity + 104 + 5];
            tot_5 = tot_5 + warp_tab[parity + 112 + 5];
            tot_5 = tot_5 + warp_tab[parity + 120 + 5];
            float _sqrt_59;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_59) : "f"(__fmaf_rn(tot_5, 0.00024414062f, eps)));
            float rs_5 = __frcp_rn(_sqrt_59);
            float yn8_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_5[_pair * 2])[0]), "=f"((&yn8_5[_pair * 2])[1])
                    : "r"(rw[20 + _pair]));
            }
            float yn_40 = ((mode != 0) ? yn8_5[0] * gw8[0] : yn8_5[0]);
            float o_40 = yn_40 * rs_5;
            float o2_40 = ((mode != 0) ? o_40 : o_40 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_40, w8[5], acc8[0]);
            float yn_41 = ((mode != 0) ? yn8_5[1] * gw8[1] : yn8_5[1]);
            float o_41 = yn_41 * rs_5;
            float o2_41 = ((mode != 0) ? o_41 : o_41 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_41, w8[5], acc8[1]);
            float yn_42 = ((mode != 0) ? yn8_5[2] * gw8[2] : yn8_5[2]);
            float o_42 = yn_42 * rs_5;
            float o2_42 = ((mode != 0) ? o_42 : o_42 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_42, w8[5], acc8[2]);
            float yn_43 = ((mode != 0) ? yn8_5[3] * gw8[3] : yn8_5[3]);
            float o_43 = yn_43 * rs_5;
            float o2_43 = ((mode != 0) ? o_43 : o_43 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_43, w8[5], acc8[3]);
            float yn_44 = ((mode != 0) ? yn8_5[4] * gw8[4] : yn8_5[4]);
            float o_44 = yn_44 * rs_5;
            float o2_44 = ((mode != 0) ? o_44 : o_44 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_44, w8[5], acc8[4]);
            float yn_45 = ((mode != 0) ? yn8_5[5] * gw8[5] : yn8_5[5]);
            float o_45 = yn_45 * rs_5;
            float o2_45 = ((mode != 0) ? o_45 : o_45 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_45, w8[5], acc8[5]);
            float yn_46 = ((mode != 0) ? yn8_5[6] * gw8[6] : yn8_5[6]);
            float o_46 = yn_46 * rs_5;
            float o2_46 = ((mode != 0) ? o_46 : o_46 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_46, w8[5], acc8[6]);
            float yn_47 = ((mode != 0) ? yn8_5[7] * gw8[7] : yn8_5[7]);
            float o_47 = yn_47 * rs_5;
            float o2_47 = ((mode != 0) ? o_47 : o_47 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_47, w8[5], acc8[7]);
            float tot_6 = warp_tab[parity + 6];
            tot_6 = tot_6 + warp_tab[parity + 8 + 6];
            tot_6 = tot_6 + warp_tab[parity + 16 + 6];
            tot_6 = tot_6 + warp_tab[parity + 24 + 6];
            tot_6 = tot_6 + warp_tab[parity + 32 + 6];
            tot_6 = tot_6 + warp_tab[parity + 40 + 6];
            tot_6 = tot_6 + warp_tab[parity + 48 + 6];
            tot_6 = tot_6 + warp_tab[parity + 56 + 6];
            tot_6 = tot_6 + warp_tab[parity + 64 + 6];
            tot_6 = tot_6 + warp_tab[parity + 72 + 6];
            tot_6 = tot_6 + warp_tab[parity + 80 + 6];
            tot_6 = tot_6 + warp_tab[parity + 88 + 6];
            tot_6 = tot_6 + warp_tab[parity + 96 + 6];
            tot_6 = tot_6 + warp_tab[parity + 104 + 6];
            tot_6 = tot_6 + warp_tab[parity + 112 + 6];
            tot_6 = tot_6 + warp_tab[parity + 120 + 6];
            float _sqrt_60;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_60) : "f"(__fmaf_rn(tot_6, 0.00024414062f, eps)));
            float rs_6 = __frcp_rn(_sqrt_60);
            float yn8_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_6[_pair * 2])[0]), "=f"((&yn8_6[_pair * 2])[1])
                    : "r"(rw[24 + _pair]));
            }
            float yn_48 = ((mode != 0) ? yn8_6[0] * gw8[0] : yn8_6[0]);
            float o_48 = yn_48 * rs_6;
            float o2_48 = ((mode != 0) ? o_48 : o_48 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_48, w8[6], acc8[0]);
            float yn_49 = ((mode != 0) ? yn8_6[1] * gw8[1] : yn8_6[1]);
            float o_49 = yn_49 * rs_6;
            float o2_49 = ((mode != 0) ? o_49 : o_49 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_49, w8[6], acc8[1]);
            float yn_50 = ((mode != 0) ? yn8_6[2] * gw8[2] : yn8_6[2]);
            float o_50 = yn_50 * rs_6;
            float o2_50 = ((mode != 0) ? o_50 : o_50 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_50, w8[6], acc8[2]);
            float yn_51 = ((mode != 0) ? yn8_6[3] * gw8[3] : yn8_6[3]);
            float o_51 = yn_51 * rs_6;
            float o2_51 = ((mode != 0) ? o_51 : o_51 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_51, w8[6], acc8[3]);
            float yn_52 = ((mode != 0) ? yn8_6[4] * gw8[4] : yn8_6[4]);
            float o_52 = yn_52 * rs_6;
            float o2_52 = ((mode != 0) ? o_52 : o_52 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_52, w8[6], acc8[4]);
            float yn_53 = ((mode != 0) ? yn8_6[5] * gw8[5] : yn8_6[5]);
            float o_53 = yn_53 * rs_6;
            float o2_53 = ((mode != 0) ? o_53 : o_53 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_53, w8[6], acc8[5]);
            float yn_54 = ((mode != 0) ? yn8_6[6] * gw8[6] : yn8_6[6]);
            float o_54 = yn_54 * rs_6;
            float o2_54 = ((mode != 0) ? o_54 : o_54 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_54, w8[6], acc8[6]);
            float yn_55 = ((mode != 0) ? yn8_6[7] * gw8[7] : yn8_6[7]);
            float o_55 = yn_55 * rs_6;
            float o2_55 = ((mode != 0) ? o_55 : o_55 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_55, w8[6], acc8[7]);
            float tot_7 = warp_tab[parity + 7];
            tot_7 = tot_7 + warp_tab[parity + 8 + 7];
            tot_7 = tot_7 + warp_tab[parity + 16 + 7];
            tot_7 = tot_7 + warp_tab[parity + 24 + 7];
            tot_7 = tot_7 + warp_tab[parity + 32 + 7];
            tot_7 = tot_7 + warp_tab[parity + 40 + 7];
            tot_7 = tot_7 + warp_tab[parity + 48 + 7];
            tot_7 = tot_7 + warp_tab[parity + 56 + 7];
            tot_7 = tot_7 + warp_tab[parity + 64 + 7];
            tot_7 = tot_7 + warp_tab[parity + 72 + 7];
            tot_7 = tot_7 + warp_tab[parity + 80 + 7];
            tot_7 = tot_7 + warp_tab[parity + 88 + 7];
            tot_7 = tot_7 + warp_tab[parity + 96 + 7];
            tot_7 = tot_7 + warp_tab[parity + 104 + 7];
            tot_7 = tot_7 + warp_tab[parity + 112 + 7];
            tot_7 = tot_7 + warp_tab[parity + 120 + 7];
            float _sqrt_61;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_61) : "f"(__fmaf_rn(tot_7, 0.00024414062f, eps)));
            float rs_7 = __frcp_rn(_sqrt_61);
            float yn8_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yn8_7[_pair * 2])[0]), "=f"((&yn8_7[_pair * 2])[1])
                    : "r"(rw[28 + _pair]));
            }
            float yn_56 = ((mode != 0) ? yn8_7[0] * gw8[0] : yn8_7[0]);
            float o_56 = yn_56 * rs_7;
            float o2_56 = ((mode != 0) ? o_56 : o_56 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_56, w8[7], acc8[0]);
            float yn_57 = ((mode != 0) ? yn8_7[1] * gw8[1] : yn8_7[1]);
            float o_57 = yn_57 * rs_7;
            float o2_57 = ((mode != 0) ? o_57 : o_57 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_57, w8[7], acc8[1]);
            float yn_58 = ((mode != 0) ? yn8_7[2] * gw8[2] : yn8_7[2]);
            float o_58 = yn_58 * rs_7;
            float o2_58 = ((mode != 0) ? o_58 : o_58 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_58, w8[7], acc8[2]);
            float yn_59 = ((mode != 0) ? yn8_7[3] * gw8[3] : yn8_7[3]);
            float o_59 = yn_59 * rs_7;
            float o2_59 = ((mode != 0) ? o_59 : o_59 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_59, w8[7], acc8[3]);
            float yn_60 = ((mode != 0) ? yn8_7[4] * gw8[4] : yn8_7[4]);
            float o_60 = yn_60 * rs_7;
            float o2_60 = ((mode != 0) ? o_60 : o_60 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_60, w8[7], acc8[4]);
            float yn_61 = ((mode != 0) ? yn8_7[5] * gw8[5] : yn8_7[5]);
            float o_61 = yn_61 * rs_7;
            float o2_61 = ((mode != 0) ? o_61 : o_61 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_61, w8[7], acc8[5]);
            float yn_62 = ((mode != 0) ? yn8_7[6] * gw8[6] : yn8_7[6]);
            float o_62 = yn_62 * rs_7;
            float o2_62 = ((mode != 0) ? o_62 : o_62 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_62, w8[7], acc8[6]);
            float yn_63 = ((mode != 0) ? yn8_7[7] * gw8[7] : yn8_7[7]);
            float o_63 = yn_63 * rs_7;
            float o2_63 = ((mode != 0) ? o_63 : o_63 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_63, w8[7], acc8[7]);
            unsigned int acc8_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 0], acc8[_lp*2+1 + 0]));
                acc8_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _packed16_sanitize_8[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_62 = static_cast<uint32_t>(acc8_bf16[_p16s_word]);
                uint32_t _p16s_lo_62 = _p16s_carrier_62 & 0x0000ffffu;
                uint32_t _p16s_hi_62 = (_p16s_carrier_62 >> 16) & 0x0000ffffu;
                if (_p16s_lo_62 == 0x00008000u) _p16s_lo_62 = 0u;
                if (_p16s_hi_62 == 0x00008000u) _p16s_hi_62 = 0u;
                _packed16_sanitize_8[_p16s_word] = _p16s_lo_62 | (_p16s_hi_62 << 16);
            }
            int gcr = sr & 7;
            long long pub = bcast_gen_elems * (long long)gcr + bcast_cta;
            asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<float*>(c_mcast + (unsigned long long)(pub * 2))), "r"(_packed16_sanitize_8[0]), "r"(_packed16_sanitize_8[1]), "r"(_packed16_sanitize_8[2]), "r"(_packed16_sanitize_8[3]) : "memory");
        }
        int jd = step - 4;
        if (jd >= 0 && jd < n_mine) {
            int tok_d = b + jd * grid;
            int sd = s0 + jd;
            int gcd = sd & 7;
            long long land = bcast_gen_elems * (long long)gcd + bcast_cta;
            uint32_t _sysv_poll_group_4[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_4[0]), "=r"(_sysv_poll_group_4[1]), "=r"(_sysv_poll_group_4[2]), "=r"(_sysv_poll_group_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(bcast_local + land)) : "memory");
            } while (
                (((_sysv_poll_group_4[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_4[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_4[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_4[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll_group_4[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_4[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_4[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_4[3] >> 16) & 0xffffu) == 0x8000u)
            );
            unsigned int row4[4];
            row4[0] = _sysv_poll_group_4[0];
            row4[1] = _sysv_poll_group_4[1];
            row4[2] = _sysv_poll_group_4[2];
            row4[3] = _sysv_poll_group_4[3];
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output + ((long long)tok_d * 4096 + col64)))[0] = reinterpret_cast<int4*>(row4)[0];
            {
                auto _gst_63 = bcast_local + land;
                __builtin_assume(__isGlobal(_gst_63));
                reinterpret_cast<int4*>(_gst_63)[0] = reinterpret_cast<int4*>(sent4)[0];
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
