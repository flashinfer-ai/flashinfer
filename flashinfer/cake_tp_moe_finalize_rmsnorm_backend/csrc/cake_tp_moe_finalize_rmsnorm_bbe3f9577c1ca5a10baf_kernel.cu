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
kernel_cake_tp_moe_finalize_rmsnorm_bbe3f9577c1ca5a10baf(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int max_ctas, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, unsigned long long c_mcast, int* __restrict__ ctl)
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
            int ridx1 = idx8[1];
            if (ridx1 >= 0 && ridx1 < rows) {
                {
                    uint4 _uv4_4 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1 * (long long)row_stride + col64) + 0);
                    packets[4 + 0] = _uv4_4.x;
                    packets[4 + 1] = _uv4_4.y;
                    packets[4 + 2] = _uv4_4.z;
                    packets[4 + 3] = _uv4_4.w;
                }
            }
            int ridx2 = idx8[2];
            if (ridx2 >= 0 && ridx2 < rows) {
                {
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx2 * (long long)row_stride + col64) + 0);
                    packets[8 + 0] = _uv4_5.x;
                    packets[8 + 1] = _uv4_5.y;
                    packets[8 + 2] = _uv4_5.z;
                    packets[8 + 3] = _uv4_5.w;
                }
            }
            int ridx3 = idx8[3];
            if (ridx3 >= 0 && ridx3 < rows) {
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx3 * (long long)row_stride + col64) + 0);
                    packets[12 + 0] = _uv4_6.x;
                    packets[12 + 1] = _uv4_6.y;
                    packets[12 + 2] = _uv4_6.z;
                    packets[12 + 3] = _uv4_6.w;
                }
            }
            int ridx4 = idx8[4];
            if (ridx4 >= 0 && ridx4 < rows) {
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx4 * (long long)row_stride + col64) + 0);
                    packets[16 + 0] = _uv4_7.x;
                    packets[16 + 1] = _uv4_7.y;
                    packets[16 + 2] = _uv4_7.z;
                    packets[16 + 3] = _uv4_7.w;
                }
            }
            int ridx5 = idx8[5];
            if (ridx5 >= 0 && ridx5 < rows) {
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx5 * (long long)row_stride + col64) + 0);
                    packets[20 + 0] = _uv4_8.x;
                    packets[20 + 1] = _uv4_8.y;
                    packets[20 + 2] = _uv4_8.z;
                    packets[20 + 3] = _uv4_8.w;
                }
            }
            int ridx6 = idx8[6];
            if (ridx6 >= 0 && ridx6 < rows) {
                {
                    uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx6 * (long long)row_stride + col64) + 0);
                    packets[24 + 0] = _uv4_9.x;
                    packets[24 + 1] = _uv4_9.y;
                    packets[24 + 2] = _uv4_9.z;
                    packets[24 + 3] = _uv4_9.w;
                }
            }
            int ridx7 = idx8[7];
            if (ridx7 >= 0 && ridx7 < rows) {
                {
                    uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx7 * (long long)row_stride + col64) + 0);
                    packets[28 + 0] = _uv4_10.x;
                    packets[28 + 1] = _uv4_10.y;
                    packets[28 + 2] = _uv4_10.z;
                    packets[28 + 3] = _uv4_10.w;
                }
            }
            unsigned int raw0[4];
            raw0[0] = packets[0];
            raw0[1] = packets[1];
            raw0[2] = packets[2];
            raw0[3] = packets[3];
            uint32_t _packed16_sanitize_0[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_11 = static_cast<uint32_t>(raw0[_p16s_word]);
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
            unsigned int raw1[4];
            raw1[0] = packets[4];
            raw1[1] = packets[5];
            raw1[2] = packets[6];
            raw1[3] = packets[7];
            uint32_t _packed16_sanitize_1[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_12 = static_cast<uint32_t>(raw1[_p16s_word]);
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
            unsigned int raw2[4];
            raw2[0] = packets[8];
            raw2[1] = packets[9];
            raw2[2] = packets[10];
            raw2[3] = packets[11];
            uint32_t _packed16_sanitize_2[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_13 = static_cast<uint32_t>(raw2[_p16s_word]);
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
            unsigned int raw3[4];
            raw3[0] = packets[12];
            raw3[1] = packets[13];
            raw3[2] = packets[14];
            raw3[3] = packets[15];
            uint32_t _packed16_sanitize_3[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_14 = static_cast<uint32_t>(raw3[_p16s_word]);
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
            unsigned int raw4[4];
            raw4[0] = packets[16];
            raw4[1] = packets[17];
            raw4[2] = packets[18];
            raw4[3] = packets[19];
            uint32_t _packed16_sanitize_4[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_15 = static_cast<uint32_t>(raw4[_p16s_word]);
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
            unsigned int raw5[4];
            raw5[0] = packets[20];
            raw5[1] = packets[21];
            raw5[2] = packets[22];
            raw5[3] = packets[23];
            uint32_t _packed16_sanitize_5[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_16 = static_cast<uint32_t>(raw5[_p16s_word]);
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
            unsigned int raw6[4];
            raw6[0] = packets[24];
            raw6[1] = packets[25];
            raw6[2] = packets[26];
            raw6[3] = packets[27];
            uint32_t _packed16_sanitize_6[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_17 = static_cast<uint32_t>(raw6[_p16s_word]);
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
            unsigned int raw7[4];
            raw7[0] = packets[28];
            raw7[1] = packets[29];
            raw7[2] = packets[30];
            raw7[3] = packets[31];
            uint32_t _packed16_sanitize_7[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_18 = static_cast<uint32_t>(raw7[_p16s_word]);
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
            float _unpack_bf16_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                    : "r"(s8_0_bf16[_pair]));
            }
            float ssq0 = 0.0f;
            float y0_0 = ((mode != 0) ? _unpack_bf16_6[0] * gw8[0] : _unpack_bf16_6[0]);
            ssq0 = __fmaf_rn(y0_0, y0_0, ssq0);
            float y0_1 = ((mode != 0) ? _unpack_bf16_6[1] * gw8[1] : _unpack_bf16_6[1]);
            ssq0 = __fmaf_rn(y0_1, y0_1, ssq0);
            float y0_2 = ((mode != 0) ? _unpack_bf16_6[2] * gw8[2] : _unpack_bf16_6[2]);
            ssq0 = __fmaf_rn(y0_2, y0_2, ssq0);
            float y0_3 = ((mode != 0) ? _unpack_bf16_6[3] * gw8[3] : _unpack_bf16_6[3]);
            ssq0 = __fmaf_rn(y0_3, y0_3, ssq0);
            float y0_4 = ((mode != 0) ? _unpack_bf16_6[4] * gw8[4] : _unpack_bf16_6[4]);
            ssq0 = __fmaf_rn(y0_4, y0_4, ssq0);
            float y0_5 = ((mode != 0) ? _unpack_bf16_6[5] * gw8[5] : _unpack_bf16_6[5]);
            ssq0 = __fmaf_rn(y0_5, y0_5, ssq0);
            float y0_6 = ((mode != 0) ? _unpack_bf16_6[6] * gw8[6] : _unpack_bf16_6[6]);
            ssq0 = __fmaf_rn(y0_6, y0_6, ssq0);
            float y0_7 = ((mode != 0) ? _unpack_bf16_6[7] * gw8[7] : _unpack_bf16_6[7]);
            ssq0 = __fmaf_rn(y0_7, y0_7, ssq0);
            ssq8[0] = ssq0;
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
            float _unpack_bf16_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[16 + _pair]));
            }
            s8_1[0] = _unpack_bf16_7[0];
            s8_1[1] = _unpack_bf16_7[1];
            s8_1[2] = _unpack_bf16_7[2];
            s8_1[3] = _unpack_bf16_7[3];
            s8_1[4] = _unpack_bf16_7[4];
            s8_1[5] = _unpack_bf16_7[5];
            s8_1[6] = _unpack_bf16_7[6];
            s8_1[7] = _unpack_bf16_7[7];
            float _unpack_bf16_8[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[20 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_8[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_8[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_8[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_8[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_8[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_8[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_8[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_8[7];
            float _unpack_bf16_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[24 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_9[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_9[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_9[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_9[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_9[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_9[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_9[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_9[7];
            float _unpack_bf16_10[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                    : "r"(_sysv_poll_group_0[28 + _pair]));
            }
            s8_1[0] = s8_1[0] + _unpack_bf16_10[0];
            s8_1[1] = s8_1[1] + _unpack_bf16_10[1];
            s8_1[2] = s8_1[2] + _unpack_bf16_10[2];
            s8_1[3] = s8_1[3] + _unpack_bf16_10[3];
            s8_1[4] = s8_1[4] + _unpack_bf16_10[4];
            s8_1[5] = s8_1[5] + _unpack_bf16_10[5];
            s8_1[6] = s8_1[6] + _unpack_bf16_10[6];
            s8_1[7] = s8_1[7] + _unpack_bf16_10[7];
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
            float _unpack_bf16_11[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                    : "r"(s8_1_bf16[_pair]));
            }
            float ssq1 = 0.0f;
            float y1_0 = ((mode != 0) ? _unpack_bf16_11[0] * gw8[0] : _unpack_bf16_11[0]);
            ssq1 = __fmaf_rn(y1_0, y1_0, ssq1);
            float y1_1 = ((mode != 0) ? _unpack_bf16_11[1] * gw8[1] : _unpack_bf16_11[1]);
            ssq1 = __fmaf_rn(y1_1, y1_1, ssq1);
            float y1_2 = ((mode != 0) ? _unpack_bf16_11[2] * gw8[2] : _unpack_bf16_11[2]);
            ssq1 = __fmaf_rn(y1_2, y1_2, ssq1);
            float y1_3 = ((mode != 0) ? _unpack_bf16_11[3] * gw8[3] : _unpack_bf16_11[3]);
            ssq1 = __fmaf_rn(y1_3, y1_3, ssq1);
            float y1_4 = ((mode != 0) ? _unpack_bf16_11[4] * gw8[4] : _unpack_bf16_11[4]);
            ssq1 = __fmaf_rn(y1_4, y1_4, ssq1);
            float y1_5 = ((mode != 0) ? _unpack_bf16_11[5] * gw8[5] : _unpack_bf16_11[5]);
            ssq1 = __fmaf_rn(y1_5, y1_5, ssq1);
            float y1_6 = ((mode != 0) ? _unpack_bf16_11[6] * gw8[6] : _unpack_bf16_11[6]);
            ssq1 = __fmaf_rn(y1_6, y1_6, ssq1);
            float y1_7 = ((mode != 0) ? _unpack_bf16_11[7] * gw8[7] : _unpack_bf16_11[7]);
            ssq1 = __fmaf_rn(y1_7, y1_7, ssq1);
            ssq8[1] = ssq1;
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
            float _unpack_bf16_12[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[_pair]));
            }
            s8_2[0] = _unpack_bf16_12[0];
            s8_2[1] = _unpack_bf16_12[1];
            s8_2[2] = _unpack_bf16_12[2];
            s8_2[3] = _unpack_bf16_12[3];
            s8_2[4] = _unpack_bf16_12[4];
            s8_2[5] = _unpack_bf16_12[5];
            s8_2[6] = _unpack_bf16_12[6];
            s8_2[7] = _unpack_bf16_12[7];
            float _unpack_bf16_13[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[4 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_13[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_13[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_13[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_13[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_13[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_13[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_13[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_13[7];
            float _unpack_bf16_14[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[8 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_14[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_14[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_14[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_14[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_14[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_14[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_14[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_14[7];
            float _unpack_bf16_15[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[12 + _pair]));
            }
            s8_2[0] = s8_2[0] + _unpack_bf16_15[0];
            s8_2[1] = s8_2[1] + _unpack_bf16_15[1];
            s8_2[2] = s8_2[2] + _unpack_bf16_15[2];
            s8_2[3] = s8_2[3] + _unpack_bf16_15[3];
            s8_2[4] = s8_2[4] + _unpack_bf16_15[4];
            s8_2[5] = s8_2[5] + _unpack_bf16_15[5];
            s8_2[6] = s8_2[6] + _unpack_bf16_15[6];
            s8_2[7] = s8_2[7] + _unpack_bf16_15[7];
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
            float _unpack_bf16_16[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                    : "r"(s8_2_bf16[_pair]));
            }
            float ssq2 = 0.0f;
            float y2_0 = ((mode != 0) ? _unpack_bf16_16[0] * gw8[0] : _unpack_bf16_16[0]);
            ssq2 = __fmaf_rn(y2_0, y2_0, ssq2);
            float y2_1 = ((mode != 0) ? _unpack_bf16_16[1] * gw8[1] : _unpack_bf16_16[1]);
            ssq2 = __fmaf_rn(y2_1, y2_1, ssq2);
            float y2_2 = ((mode != 0) ? _unpack_bf16_16[2] * gw8[2] : _unpack_bf16_16[2]);
            ssq2 = __fmaf_rn(y2_2, y2_2, ssq2);
            float y2_3 = ((mode != 0) ? _unpack_bf16_16[3] * gw8[3] : _unpack_bf16_16[3]);
            ssq2 = __fmaf_rn(y2_3, y2_3, ssq2);
            float y2_4 = ((mode != 0) ? _unpack_bf16_16[4] * gw8[4] : _unpack_bf16_16[4]);
            ssq2 = __fmaf_rn(y2_4, y2_4, ssq2);
            float y2_5 = ((mode != 0) ? _unpack_bf16_16[5] * gw8[5] : _unpack_bf16_16[5]);
            ssq2 = __fmaf_rn(y2_5, y2_5, ssq2);
            float y2_6 = ((mode != 0) ? _unpack_bf16_16[6] * gw8[6] : _unpack_bf16_16[6]);
            ssq2 = __fmaf_rn(y2_6, y2_6, ssq2);
            float y2_7 = ((mode != 0) ? _unpack_bf16_16[7] * gw8[7] : _unpack_bf16_16[7]);
            ssq2 = __fmaf_rn(y2_7, y2_7, ssq2);
            ssq8[2] = ssq2;
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
            float _unpack_bf16_17[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[16 + _pair]));
            }
            s8_3[0] = _unpack_bf16_17[0];
            s8_3[1] = _unpack_bf16_17[1];
            s8_3[2] = _unpack_bf16_17[2];
            s8_3[3] = _unpack_bf16_17[3];
            s8_3[4] = _unpack_bf16_17[4];
            s8_3[5] = _unpack_bf16_17[5];
            s8_3[6] = _unpack_bf16_17[6];
            s8_3[7] = _unpack_bf16_17[7];
            float _unpack_bf16_18[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[20 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_18[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_18[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_18[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_18[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_18[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_18[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_18[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_18[7];
            float _unpack_bf16_19[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[24 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_19[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_19[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_19[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_19[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_19[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_19[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_19[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_19[7];
            float _unpack_bf16_20[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[28 + _pair]));
            }
            s8_3[0] = s8_3[0] + _unpack_bf16_20[0];
            s8_3[1] = s8_3[1] + _unpack_bf16_20[1];
            s8_3[2] = s8_3[2] + _unpack_bf16_20[2];
            s8_3[3] = s8_3[3] + _unpack_bf16_20[3];
            s8_3[4] = s8_3[4] + _unpack_bf16_20[4];
            s8_3[5] = s8_3[5] + _unpack_bf16_20[5];
            s8_3[6] = s8_3[6] + _unpack_bf16_20[6];
            s8_3[7] = s8_3[7] + _unpack_bf16_20[7];
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
            float _unpack_bf16_21[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                    : "r"(s8_3_bf16[_pair]));
            }
            float ssq3 = 0.0f;
            float y3_0 = ((mode != 0) ? _unpack_bf16_21[0] * gw8[0] : _unpack_bf16_21[0]);
            ssq3 = __fmaf_rn(y3_0, y3_0, ssq3);
            float y3_1 = ((mode != 0) ? _unpack_bf16_21[1] * gw8[1] : _unpack_bf16_21[1]);
            ssq3 = __fmaf_rn(y3_1, y3_1, ssq3);
            float y3_2 = ((mode != 0) ? _unpack_bf16_21[2] * gw8[2] : _unpack_bf16_21[2]);
            ssq3 = __fmaf_rn(y3_2, y3_2, ssq3);
            float y3_3 = ((mode != 0) ? _unpack_bf16_21[3] * gw8[3] : _unpack_bf16_21[3]);
            ssq3 = __fmaf_rn(y3_3, y3_3, ssq3);
            float y3_4 = ((mode != 0) ? _unpack_bf16_21[4] * gw8[4] : _unpack_bf16_21[4]);
            ssq3 = __fmaf_rn(y3_4, y3_4, ssq3);
            float y3_5 = ((mode != 0) ? _unpack_bf16_21[5] * gw8[5] : _unpack_bf16_21[5]);
            ssq3 = __fmaf_rn(y3_5, y3_5, ssq3);
            float y3_6 = ((mode != 0) ? _unpack_bf16_21[6] * gw8[6] : _unpack_bf16_21[6]);
            ssq3 = __fmaf_rn(y3_6, y3_6, ssq3);
            float y3_7 = ((mode != 0) ? _unpack_bf16_21[7] * gw8[7] : _unpack_bf16_21[7]);
            ssq3 = __fmaf_rn(y3_7, y3_7, ssq3);
            ssq8[3] = ssq3;
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
            float _unpack_bf16_22[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[_pair]));
            }
            s8_4[0] = _unpack_bf16_22[0];
            s8_4[1] = _unpack_bf16_22[1];
            s8_4[2] = _unpack_bf16_22[2];
            s8_4[3] = _unpack_bf16_22[3];
            s8_4[4] = _unpack_bf16_22[4];
            s8_4[5] = _unpack_bf16_22[5];
            s8_4[6] = _unpack_bf16_22[6];
            s8_4[7] = _unpack_bf16_22[7];
            float _unpack_bf16_23[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[4 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_23[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_23[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_23[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_23[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_23[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_23[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_23[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_23[7];
            float _unpack_bf16_24[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[8 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_24[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_24[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_24[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_24[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_24[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_24[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_24[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_24[7];
            float _unpack_bf16_25[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[12 + _pair]));
            }
            s8_4[0] = s8_4[0] + _unpack_bf16_25[0];
            s8_4[1] = s8_4[1] + _unpack_bf16_25[1];
            s8_4[2] = s8_4[2] + _unpack_bf16_25[2];
            s8_4[3] = s8_4[3] + _unpack_bf16_25[3];
            s8_4[4] = s8_4[4] + _unpack_bf16_25[4];
            s8_4[5] = s8_4[5] + _unpack_bf16_25[5];
            s8_4[6] = s8_4[6] + _unpack_bf16_25[6];
            s8_4[7] = s8_4[7] + _unpack_bf16_25[7];
            unsigned int s8_4_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_4[_lp*2 + 0], s8_4[_lp*2+1 + 0]));
                s8_4_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            rw[16] = s8_4_bf16[0];
            rw[17] = s8_4_bf16[1];
            rw[18] = s8_4_bf16[2];
            rw[19] = s8_4_bf16[3];
            float _unpack_bf16_26[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                    : "r"(s8_4_bf16[_pair]));
            }
            float ssq4 = 0.0f;
            float y4_0 = ((mode != 0) ? _unpack_bf16_26[0] * gw8[0] : _unpack_bf16_26[0]);
            ssq4 = __fmaf_rn(y4_0, y4_0, ssq4);
            float y4_1 = ((mode != 0) ? _unpack_bf16_26[1] * gw8[1] : _unpack_bf16_26[1]);
            ssq4 = __fmaf_rn(y4_1, y4_1, ssq4);
            float y4_2 = ((mode != 0) ? _unpack_bf16_26[2] * gw8[2] : _unpack_bf16_26[2]);
            ssq4 = __fmaf_rn(y4_2, y4_2, ssq4);
            float y4_3 = ((mode != 0) ? _unpack_bf16_26[3] * gw8[3] : _unpack_bf16_26[3]);
            ssq4 = __fmaf_rn(y4_3, y4_3, ssq4);
            float y4_4 = ((mode != 0) ? _unpack_bf16_26[4] * gw8[4] : _unpack_bf16_26[4]);
            ssq4 = __fmaf_rn(y4_4, y4_4, ssq4);
            float y4_5 = ((mode != 0) ? _unpack_bf16_26[5] * gw8[5] : _unpack_bf16_26[5]);
            ssq4 = __fmaf_rn(y4_5, y4_5, ssq4);
            float y4_6 = ((mode != 0) ? _unpack_bf16_26[6] * gw8[6] : _unpack_bf16_26[6]);
            ssq4 = __fmaf_rn(y4_6, y4_6, ssq4);
            float y4_7 = ((mode != 0) ? _unpack_bf16_26[7] * gw8[7] : _unpack_bf16_26[7]);
            ssq4 = __fmaf_rn(y4_7, y4_7, ssq4);
            ssq8[4] = ssq4;
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
            float _unpack_bf16_27[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[16 + _pair]));
            }
            s8_5[0] = _unpack_bf16_27[0];
            s8_5[1] = _unpack_bf16_27[1];
            s8_5[2] = _unpack_bf16_27[2];
            s8_5[3] = _unpack_bf16_27[3];
            s8_5[4] = _unpack_bf16_27[4];
            s8_5[5] = _unpack_bf16_27[5];
            s8_5[6] = _unpack_bf16_27[6];
            s8_5[7] = _unpack_bf16_27[7];
            float _unpack_bf16_28[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[20 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_28[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_28[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_28[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_28[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_28[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_28[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_28[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_28[7];
            float _unpack_bf16_29[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[24 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_29[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_29[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_29[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_29[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_29[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_29[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_29[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_29[7];
            float _unpack_bf16_30[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[28 + _pair]));
            }
            s8_5[0] = s8_5[0] + _unpack_bf16_30[0];
            s8_5[1] = s8_5[1] + _unpack_bf16_30[1];
            s8_5[2] = s8_5[2] + _unpack_bf16_30[2];
            s8_5[3] = s8_5[3] + _unpack_bf16_30[3];
            s8_5[4] = s8_5[4] + _unpack_bf16_30[4];
            s8_5[5] = s8_5[5] + _unpack_bf16_30[5];
            s8_5[6] = s8_5[6] + _unpack_bf16_30[6];
            s8_5[7] = s8_5[7] + _unpack_bf16_30[7];
            unsigned int s8_5_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_5[_lp*2 + 0], s8_5[_lp*2+1 + 0]));
                s8_5_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            rw[20] = s8_5_bf16[0];
            rw[21] = s8_5_bf16[1];
            rw[22] = s8_5_bf16[2];
            rw[23] = s8_5_bf16[3];
            float _unpack_bf16_31[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                    : "r"(s8_5_bf16[_pair]));
            }
            float ssq5 = 0.0f;
            float y5_0 = ((mode != 0) ? _unpack_bf16_31[0] * gw8[0] : _unpack_bf16_31[0]);
            ssq5 = __fmaf_rn(y5_0, y5_0, ssq5);
            float y5_1 = ((mode != 0) ? _unpack_bf16_31[1] * gw8[1] : _unpack_bf16_31[1]);
            ssq5 = __fmaf_rn(y5_1, y5_1, ssq5);
            float y5_2 = ((mode != 0) ? _unpack_bf16_31[2] * gw8[2] : _unpack_bf16_31[2]);
            ssq5 = __fmaf_rn(y5_2, y5_2, ssq5);
            float y5_3 = ((mode != 0) ? _unpack_bf16_31[3] * gw8[3] : _unpack_bf16_31[3]);
            ssq5 = __fmaf_rn(y5_3, y5_3, ssq5);
            float y5_4 = ((mode != 0) ? _unpack_bf16_31[4] * gw8[4] : _unpack_bf16_31[4]);
            ssq5 = __fmaf_rn(y5_4, y5_4, ssq5);
            float y5_5 = ((mode != 0) ? _unpack_bf16_31[5] * gw8[5] : _unpack_bf16_31[5]);
            ssq5 = __fmaf_rn(y5_5, y5_5, ssq5);
            float y5_6 = ((mode != 0) ? _unpack_bf16_31[6] * gw8[6] : _unpack_bf16_31[6]);
            ssq5 = __fmaf_rn(y5_6, y5_6, ssq5);
            float y5_7 = ((mode != 0) ? _unpack_bf16_31[7] * gw8[7] : _unpack_bf16_31[7]);
            ssq5 = __fmaf_rn(y5_7, y5_7, ssq5);
            ssq8[5] = ssq5;
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
            float _unpack_bf16_32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[_pair]));
            }
            s8_6[0] = _unpack_bf16_32[0];
            s8_6[1] = _unpack_bf16_32[1];
            s8_6[2] = _unpack_bf16_32[2];
            s8_6[3] = _unpack_bf16_32[3];
            s8_6[4] = _unpack_bf16_32[4];
            s8_6[5] = _unpack_bf16_32[5];
            s8_6[6] = _unpack_bf16_32[6];
            s8_6[7] = _unpack_bf16_32[7];
            float _unpack_bf16_33[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[4 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_33[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_33[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_33[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_33[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_33[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_33[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_33[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_33[7];
            float _unpack_bf16_34[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[8 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_34[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_34[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_34[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_34[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_34[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_34[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_34[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_34[7];
            float _unpack_bf16_35[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[12 + _pair]));
            }
            s8_6[0] = s8_6[0] + _unpack_bf16_35[0];
            s8_6[1] = s8_6[1] + _unpack_bf16_35[1];
            s8_6[2] = s8_6[2] + _unpack_bf16_35[2];
            s8_6[3] = s8_6[3] + _unpack_bf16_35[3];
            s8_6[4] = s8_6[4] + _unpack_bf16_35[4];
            s8_6[5] = s8_6[5] + _unpack_bf16_35[5];
            s8_6[6] = s8_6[6] + _unpack_bf16_35[6];
            s8_6[7] = s8_6[7] + _unpack_bf16_35[7];
            unsigned int s8_6_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_6[_lp*2 + 0], s8_6[_lp*2+1 + 0]));
                s8_6_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            rw[24] = s8_6_bf16[0];
            rw[25] = s8_6_bf16[1];
            rw[26] = s8_6_bf16[2];
            rw[27] = s8_6_bf16[3];
            float _unpack_bf16_36[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                    : "r"(s8_6_bf16[_pair]));
            }
            float ssq6 = 0.0f;
            float y6_0 = ((mode != 0) ? _unpack_bf16_36[0] * gw8[0] : _unpack_bf16_36[0]);
            ssq6 = __fmaf_rn(y6_0, y6_0, ssq6);
            float y6_1 = ((mode != 0) ? _unpack_bf16_36[1] * gw8[1] : _unpack_bf16_36[1]);
            ssq6 = __fmaf_rn(y6_1, y6_1, ssq6);
            float y6_2 = ((mode != 0) ? _unpack_bf16_36[2] * gw8[2] : _unpack_bf16_36[2]);
            ssq6 = __fmaf_rn(y6_2, y6_2, ssq6);
            float y6_3 = ((mode != 0) ? _unpack_bf16_36[3] * gw8[3] : _unpack_bf16_36[3]);
            ssq6 = __fmaf_rn(y6_3, y6_3, ssq6);
            float y6_4 = ((mode != 0) ? _unpack_bf16_36[4] * gw8[4] : _unpack_bf16_36[4]);
            ssq6 = __fmaf_rn(y6_4, y6_4, ssq6);
            float y6_5 = ((mode != 0) ? _unpack_bf16_36[5] * gw8[5] : _unpack_bf16_36[5]);
            ssq6 = __fmaf_rn(y6_5, y6_5, ssq6);
            float y6_6 = ((mode != 0) ? _unpack_bf16_36[6] * gw8[6] : _unpack_bf16_36[6]);
            ssq6 = __fmaf_rn(y6_6, y6_6, ssq6);
            float y6_7 = ((mode != 0) ? _unpack_bf16_36[7] * gw8[7] : _unpack_bf16_36[7]);
            ssq6 = __fmaf_rn(y6_7, y6_7, ssq6);
            ssq8[6] = ssq6;
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
            float _unpack_bf16_37[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[16 + _pair]));
            }
            s8_7[0] = _unpack_bf16_37[0];
            s8_7[1] = _unpack_bf16_37[1];
            s8_7[2] = _unpack_bf16_37[2];
            s8_7[3] = _unpack_bf16_37[3];
            s8_7[4] = _unpack_bf16_37[4];
            s8_7[5] = _unpack_bf16_37[5];
            s8_7[6] = _unpack_bf16_37[6];
            s8_7[7] = _unpack_bf16_37[7];
            float _unpack_bf16_38[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[20 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_38[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_38[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_38[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_38[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_38[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_38[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_38[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_38[7];
            float _unpack_bf16_39[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[24 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_39[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_39[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_39[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_39[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_39[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_39[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_39[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_39[7];
            float _unpack_bf16_40[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[28 + _pair]));
            }
            s8_7[0] = s8_7[0] + _unpack_bf16_40[0];
            s8_7[1] = s8_7[1] + _unpack_bf16_40[1];
            s8_7[2] = s8_7[2] + _unpack_bf16_40[2];
            s8_7[3] = s8_7[3] + _unpack_bf16_40[3];
            s8_7[4] = s8_7[4] + _unpack_bf16_40[4];
            s8_7[5] = s8_7[5] + _unpack_bf16_40[5];
            s8_7[6] = s8_7[6] + _unpack_bf16_40[6];
            s8_7[7] = s8_7[7] + _unpack_bf16_40[7];
            unsigned int s8_7_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_7[_lp*2 + 0], s8_7[_lp*2+1 + 0]));
                s8_7_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            rw[28] = s8_7_bf16[0];
            rw[29] = s8_7_bf16[1];
            rw[30] = s8_7_bf16[2];
            rw[31] = s8_7_bf16[3];
            float _unpack_bf16_41[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                    : "r"(s8_7_bf16[_pair]));
            }
            float ssq7 = 0.0f;
            float y7_0 = ((mode != 0) ? _unpack_bf16_41[0] * gw8[0] : _unpack_bf16_41[0]);
            ssq7 = __fmaf_rn(y7_0, y7_0, ssq7);
            float y7_1 = ((mode != 0) ? _unpack_bf16_41[1] * gw8[1] : _unpack_bf16_41[1]);
            ssq7 = __fmaf_rn(y7_1, y7_1, ssq7);
            float y7_2 = ((mode != 0) ? _unpack_bf16_41[2] * gw8[2] : _unpack_bf16_41[2]);
            ssq7 = __fmaf_rn(y7_2, y7_2, ssq7);
            float y7_3 = ((mode != 0) ? _unpack_bf16_41[3] * gw8[3] : _unpack_bf16_41[3]);
            ssq7 = __fmaf_rn(y7_3, y7_3, ssq7);
            float y7_4 = ((mode != 0) ? _unpack_bf16_41[4] * gw8[4] : _unpack_bf16_41[4]);
            ssq7 = __fmaf_rn(y7_4, y7_4, ssq7);
            float y7_5 = ((mode != 0) ? _unpack_bf16_41[5] * gw8[5] : _unpack_bf16_41[5]);
            ssq7 = __fmaf_rn(y7_5, y7_5, ssq7);
            float y7_6 = ((mode != 0) ? _unpack_bf16_41[6] * gw8[6] : _unpack_bf16_41[6]);
            ssq7 = __fmaf_rn(y7_6, y7_6, ssq7);
            float y7_7 = ((mode != 0) ? _unpack_bf16_41[7] * gw8[7] : _unpack_bf16_41[7]);
            ssq7 = __fmaf_rn(y7_7, y7_7, ssq7);
            ssq8[7] = ssq7;
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
            float tot0 = warp_tab[parity];
            tot0 = tot0 + warp_tab[parity + 8];
            tot0 = tot0 + warp_tab[parity + 16];
            tot0 = tot0 + warp_tab[parity + 24];
            tot0 = tot0 + warp_tab[parity + 32];
            tot0 = tot0 + warp_tab[parity + 40];
            tot0 = tot0 + warp_tab[parity + 48];
            tot0 = tot0 + warp_tab[parity + 56];
            tot0 = tot0 + warp_tab[parity + 64];
            tot0 = tot0 + warp_tab[parity + 72];
            tot0 = tot0 + warp_tab[parity + 80];
            tot0 = tot0 + warp_tab[parity + 88];
            tot0 = tot0 + warp_tab[parity + 96];
            tot0 = tot0 + warp_tab[parity + 104];
            tot0 = tot0 + warp_tab[parity + 112];
            tot0 = tot0 + warp_tab[parity + 120];
            float _sqrt_54;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_54) : "f"(__fmaf_rn(tot0, 0.00024414062f, eps)));
            float rs0 = __frcp_rn(_sqrt_54);
            float _unpack_bf16_42[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                    : "r"(rw[_pair]));
            }
            float yn0_0 = ((mode != 0) ? _unpack_bf16_42[0] * gw8[0] : _unpack_bf16_42[0]);
            float o0_0 = yn0_0 * rs0;
            float o2_0_0 = ((mode != 0) ? o0_0 : o0_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_0_0, w8[0], acc8[0]);
            float yn0_1 = ((mode != 0) ? _unpack_bf16_42[1] * gw8[1] : _unpack_bf16_42[1]);
            float o0_1 = yn0_1 * rs0;
            float o2_0_1 = ((mode != 0) ? o0_1 : o0_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_0_1, w8[0], acc8[1]);
            float yn0_2 = ((mode != 0) ? _unpack_bf16_42[2] * gw8[2] : _unpack_bf16_42[2]);
            float o0_2 = yn0_2 * rs0;
            float o2_0_2 = ((mode != 0) ? o0_2 : o0_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_0_2, w8[0], acc8[2]);
            float yn0_3 = ((mode != 0) ? _unpack_bf16_42[3] * gw8[3] : _unpack_bf16_42[3]);
            float o0_3 = yn0_3 * rs0;
            float o2_0_3 = ((mode != 0) ? o0_3 : o0_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_0_3, w8[0], acc8[3]);
            float yn0_4 = ((mode != 0) ? _unpack_bf16_42[4] * gw8[4] : _unpack_bf16_42[4]);
            float o0_4 = yn0_4 * rs0;
            float o2_0_4 = ((mode != 0) ? o0_4 : o0_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_0_4, w8[0], acc8[4]);
            float yn0_5 = ((mode != 0) ? _unpack_bf16_42[5] * gw8[5] : _unpack_bf16_42[5]);
            float o0_5 = yn0_5 * rs0;
            float o2_0_5 = ((mode != 0) ? o0_5 : o0_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_0_5, w8[0], acc8[5]);
            float yn0_6 = ((mode != 0) ? _unpack_bf16_42[6] * gw8[6] : _unpack_bf16_42[6]);
            float o0_6 = yn0_6 * rs0;
            float o2_0_6 = ((mode != 0) ? o0_6 : o0_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_0_6, w8[0], acc8[6]);
            float yn0_7 = ((mode != 0) ? _unpack_bf16_42[7] * gw8[7] : _unpack_bf16_42[7]);
            float o0_7 = yn0_7 * rs0;
            float o2_0_7 = ((mode != 0) ? o0_7 : o0_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_0_7, w8[0], acc8[7]);
            float tot1 = warp_tab[parity + 1];
            tot1 = tot1 + warp_tab[parity + 8 + 1];
            tot1 = tot1 + warp_tab[parity + 16 + 1];
            tot1 = tot1 + warp_tab[parity + 24 + 1];
            tot1 = tot1 + warp_tab[parity + 32 + 1];
            tot1 = tot1 + warp_tab[parity + 40 + 1];
            tot1 = tot1 + warp_tab[parity + 48 + 1];
            tot1 = tot1 + warp_tab[parity + 56 + 1];
            tot1 = tot1 + warp_tab[parity + 64 + 1];
            tot1 = tot1 + warp_tab[parity + 72 + 1];
            tot1 = tot1 + warp_tab[parity + 80 + 1];
            tot1 = tot1 + warp_tab[parity + 88 + 1];
            tot1 = tot1 + warp_tab[parity + 96 + 1];
            tot1 = tot1 + warp_tab[parity + 104 + 1];
            tot1 = tot1 + warp_tab[parity + 112 + 1];
            tot1 = tot1 + warp_tab[parity + 120 + 1];
            float _sqrt_55;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_55) : "f"(__fmaf_rn(tot1, 0.00024414062f, eps)));
            float rs1 = __frcp_rn(_sqrt_55);
            float _unpack_bf16_43[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                    : "r"(rw[4 + _pair]));
            }
            float yn1_0 = ((mode != 0) ? _unpack_bf16_43[0] * gw8[0] : _unpack_bf16_43[0]);
            float o1_0 = yn1_0 * rs1;
            float o2_1_0 = ((mode != 0) ? o1_0 : o1_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_1_0, w8[1], acc8[0]);
            float yn1_1 = ((mode != 0) ? _unpack_bf16_43[1] * gw8[1] : _unpack_bf16_43[1]);
            float o1_1 = yn1_1 * rs1;
            float o2_1_1 = ((mode != 0) ? o1_1 : o1_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_1_1, w8[1], acc8[1]);
            float yn1_2 = ((mode != 0) ? _unpack_bf16_43[2] * gw8[2] : _unpack_bf16_43[2]);
            float o1_2 = yn1_2 * rs1;
            float o2_1_2 = ((mode != 0) ? o1_2 : o1_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_1_2, w8[1], acc8[2]);
            float yn1_3 = ((mode != 0) ? _unpack_bf16_43[3] * gw8[3] : _unpack_bf16_43[3]);
            float o1_3 = yn1_3 * rs1;
            float o2_1_3 = ((mode != 0) ? o1_3 : o1_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_1_3, w8[1], acc8[3]);
            float yn1_4 = ((mode != 0) ? _unpack_bf16_43[4] * gw8[4] : _unpack_bf16_43[4]);
            float o1_4 = yn1_4 * rs1;
            float o2_1_4 = ((mode != 0) ? o1_4 : o1_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_1_4, w8[1], acc8[4]);
            float yn1_5 = ((mode != 0) ? _unpack_bf16_43[5] * gw8[5] : _unpack_bf16_43[5]);
            float o1_5 = yn1_5 * rs1;
            float o2_1_5 = ((mode != 0) ? o1_5 : o1_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_1_5, w8[1], acc8[5]);
            float yn1_6 = ((mode != 0) ? _unpack_bf16_43[6] * gw8[6] : _unpack_bf16_43[6]);
            float o1_6 = yn1_6 * rs1;
            float o2_1_6 = ((mode != 0) ? o1_6 : o1_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_1_6, w8[1], acc8[6]);
            float yn1_7 = ((mode != 0) ? _unpack_bf16_43[7] * gw8[7] : _unpack_bf16_43[7]);
            float o1_7 = yn1_7 * rs1;
            float o2_1_7 = ((mode != 0) ? o1_7 : o1_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_1_7, w8[1], acc8[7]);
            float tot2 = warp_tab[parity + 2];
            tot2 = tot2 + warp_tab[parity + 8 + 2];
            tot2 = tot2 + warp_tab[parity + 16 + 2];
            tot2 = tot2 + warp_tab[parity + 24 + 2];
            tot2 = tot2 + warp_tab[parity + 32 + 2];
            tot2 = tot2 + warp_tab[parity + 40 + 2];
            tot2 = tot2 + warp_tab[parity + 48 + 2];
            tot2 = tot2 + warp_tab[parity + 56 + 2];
            tot2 = tot2 + warp_tab[parity + 64 + 2];
            tot2 = tot2 + warp_tab[parity + 72 + 2];
            tot2 = tot2 + warp_tab[parity + 80 + 2];
            tot2 = tot2 + warp_tab[parity + 88 + 2];
            tot2 = tot2 + warp_tab[parity + 96 + 2];
            tot2 = tot2 + warp_tab[parity + 104 + 2];
            tot2 = tot2 + warp_tab[parity + 112 + 2];
            tot2 = tot2 + warp_tab[parity + 120 + 2];
            float _sqrt_56;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_56) : "f"(__fmaf_rn(tot2, 0.00024414062f, eps)));
            float rs2 = __frcp_rn(_sqrt_56);
            float _unpack_bf16_44[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_44[_pair * 2])[0]), "=f"((&_unpack_bf16_44[_pair * 2])[1])
                    : "r"(rw[8 + _pair]));
            }
            float yn2_0 = ((mode != 0) ? _unpack_bf16_44[0] * gw8[0] : _unpack_bf16_44[0]);
            float o2_0 = yn2_0 * rs2;
            float o2_2_0 = ((mode != 0) ? o2_0 : o2_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_2_0, w8[2], acc8[0]);
            float yn2_1 = ((mode != 0) ? _unpack_bf16_44[1] * gw8[1] : _unpack_bf16_44[1]);
            float o2_1 = yn2_1 * rs2;
            float o2_2_1 = ((mode != 0) ? o2_1 : o2_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_2_1, w8[2], acc8[1]);
            float yn2_2 = ((mode != 0) ? _unpack_bf16_44[2] * gw8[2] : _unpack_bf16_44[2]);
            float o2_2 = yn2_2 * rs2;
            float o2_2_2 = ((mode != 0) ? o2_2 : o2_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_2_2, w8[2], acc8[2]);
            float yn2_3 = ((mode != 0) ? _unpack_bf16_44[3] * gw8[3] : _unpack_bf16_44[3]);
            float o2_3 = yn2_3 * rs2;
            float o2_2_3 = ((mode != 0) ? o2_3 : o2_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_2_3, w8[2], acc8[3]);
            float yn2_4 = ((mode != 0) ? _unpack_bf16_44[4] * gw8[4] : _unpack_bf16_44[4]);
            float o2_4 = yn2_4 * rs2;
            float o2_2_4 = ((mode != 0) ? o2_4 : o2_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_2_4, w8[2], acc8[4]);
            float yn2_5 = ((mode != 0) ? _unpack_bf16_44[5] * gw8[5] : _unpack_bf16_44[5]);
            float o2_5 = yn2_5 * rs2;
            float o2_2_5 = ((mode != 0) ? o2_5 : o2_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_2_5, w8[2], acc8[5]);
            float yn2_6 = ((mode != 0) ? _unpack_bf16_44[6] * gw8[6] : _unpack_bf16_44[6]);
            float o2_6 = yn2_6 * rs2;
            float o2_2_6 = ((mode != 0) ? o2_6 : o2_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_2_6, w8[2], acc8[6]);
            float yn2_7 = ((mode != 0) ? _unpack_bf16_44[7] * gw8[7] : _unpack_bf16_44[7]);
            float o2_7 = yn2_7 * rs2;
            float o2_2_7 = ((mode != 0) ? o2_7 : o2_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_2_7, w8[2], acc8[7]);
            float tot3 = warp_tab[parity + 3];
            tot3 = tot3 + warp_tab[parity + 8 + 3];
            tot3 = tot3 + warp_tab[parity + 16 + 3];
            tot3 = tot3 + warp_tab[parity + 24 + 3];
            tot3 = tot3 + warp_tab[parity + 32 + 3];
            tot3 = tot3 + warp_tab[parity + 40 + 3];
            tot3 = tot3 + warp_tab[parity + 48 + 3];
            tot3 = tot3 + warp_tab[parity + 56 + 3];
            tot3 = tot3 + warp_tab[parity + 64 + 3];
            tot3 = tot3 + warp_tab[parity + 72 + 3];
            tot3 = tot3 + warp_tab[parity + 80 + 3];
            tot3 = tot3 + warp_tab[parity + 88 + 3];
            tot3 = tot3 + warp_tab[parity + 96 + 3];
            tot3 = tot3 + warp_tab[parity + 104 + 3];
            tot3 = tot3 + warp_tab[parity + 112 + 3];
            tot3 = tot3 + warp_tab[parity + 120 + 3];
            float _sqrt_57;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_57) : "f"(__fmaf_rn(tot3, 0.00024414062f, eps)));
            float rs3 = __frcp_rn(_sqrt_57);
            float _unpack_bf16_45[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_45[_pair * 2])[0]), "=f"((&_unpack_bf16_45[_pair * 2])[1])
                    : "r"(rw[12 + _pair]));
            }
            float yn3_0 = ((mode != 0) ? _unpack_bf16_45[0] * gw8[0] : _unpack_bf16_45[0]);
            float o3_0 = yn3_0 * rs3;
            float o2_3_0 = ((mode != 0) ? o3_0 : o3_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_3_0, w8[3], acc8[0]);
            float yn3_1 = ((mode != 0) ? _unpack_bf16_45[1] * gw8[1] : _unpack_bf16_45[1]);
            float o3_1 = yn3_1 * rs3;
            float o2_3_1 = ((mode != 0) ? o3_1 : o3_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_3_1, w8[3], acc8[1]);
            float yn3_2 = ((mode != 0) ? _unpack_bf16_45[2] * gw8[2] : _unpack_bf16_45[2]);
            float o3_2 = yn3_2 * rs3;
            float o2_3_2 = ((mode != 0) ? o3_2 : o3_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_3_2, w8[3], acc8[2]);
            float yn3_3 = ((mode != 0) ? _unpack_bf16_45[3] * gw8[3] : _unpack_bf16_45[3]);
            float o3_3 = yn3_3 * rs3;
            float o2_3_3 = ((mode != 0) ? o3_3 : o3_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_3_3, w8[3], acc8[3]);
            float yn3_4 = ((mode != 0) ? _unpack_bf16_45[4] * gw8[4] : _unpack_bf16_45[4]);
            float o3_4 = yn3_4 * rs3;
            float o2_3_4 = ((mode != 0) ? o3_4 : o3_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_3_4, w8[3], acc8[4]);
            float yn3_5 = ((mode != 0) ? _unpack_bf16_45[5] * gw8[5] : _unpack_bf16_45[5]);
            float o3_5 = yn3_5 * rs3;
            float o2_3_5 = ((mode != 0) ? o3_5 : o3_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_3_5, w8[3], acc8[5]);
            float yn3_6 = ((mode != 0) ? _unpack_bf16_45[6] * gw8[6] : _unpack_bf16_45[6]);
            float o3_6 = yn3_6 * rs3;
            float o2_3_6 = ((mode != 0) ? o3_6 : o3_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_3_6, w8[3], acc8[6]);
            float yn3_7 = ((mode != 0) ? _unpack_bf16_45[7] * gw8[7] : _unpack_bf16_45[7]);
            float o3_7 = yn3_7 * rs3;
            float o2_3_7 = ((mode != 0) ? o3_7 : o3_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_3_7, w8[3], acc8[7]);
            float tot4 = warp_tab[parity + 4];
            tot4 = tot4 + warp_tab[parity + 8 + 4];
            tot4 = tot4 + warp_tab[parity + 16 + 4];
            tot4 = tot4 + warp_tab[parity + 24 + 4];
            tot4 = tot4 + warp_tab[parity + 32 + 4];
            tot4 = tot4 + warp_tab[parity + 40 + 4];
            tot4 = tot4 + warp_tab[parity + 48 + 4];
            tot4 = tot4 + warp_tab[parity + 56 + 4];
            tot4 = tot4 + warp_tab[parity + 64 + 4];
            tot4 = tot4 + warp_tab[parity + 72 + 4];
            tot4 = tot4 + warp_tab[parity + 80 + 4];
            tot4 = tot4 + warp_tab[parity + 88 + 4];
            tot4 = tot4 + warp_tab[parity + 96 + 4];
            tot4 = tot4 + warp_tab[parity + 104 + 4];
            tot4 = tot4 + warp_tab[parity + 112 + 4];
            tot4 = tot4 + warp_tab[parity + 120 + 4];
            float _sqrt_58;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_58) : "f"(__fmaf_rn(tot4, 0.00024414062f, eps)));
            float rs4 = __frcp_rn(_sqrt_58);
            float _unpack_bf16_46[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_46[_pair * 2])[0]), "=f"((&_unpack_bf16_46[_pair * 2])[1])
                    : "r"(rw[16 + _pair]));
            }
            float yn4_0 = ((mode != 0) ? _unpack_bf16_46[0] * gw8[0] : _unpack_bf16_46[0]);
            float o4_0 = yn4_0 * rs4;
            float o2_4_0 = ((mode != 0) ? o4_0 : o4_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_4_0, w8[4], acc8[0]);
            float yn4_1 = ((mode != 0) ? _unpack_bf16_46[1] * gw8[1] : _unpack_bf16_46[1]);
            float o4_1 = yn4_1 * rs4;
            float o2_4_1 = ((mode != 0) ? o4_1 : o4_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_4_1, w8[4], acc8[1]);
            float yn4_2 = ((mode != 0) ? _unpack_bf16_46[2] * gw8[2] : _unpack_bf16_46[2]);
            float o4_2 = yn4_2 * rs4;
            float o2_4_2 = ((mode != 0) ? o4_2 : o4_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_4_2, w8[4], acc8[2]);
            float yn4_3 = ((mode != 0) ? _unpack_bf16_46[3] * gw8[3] : _unpack_bf16_46[3]);
            float o4_3 = yn4_3 * rs4;
            float o2_4_3 = ((mode != 0) ? o4_3 : o4_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_4_3, w8[4], acc8[3]);
            float yn4_4 = ((mode != 0) ? _unpack_bf16_46[4] * gw8[4] : _unpack_bf16_46[4]);
            float o4_4 = yn4_4 * rs4;
            float o2_4_4 = ((mode != 0) ? o4_4 : o4_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_4_4, w8[4], acc8[4]);
            float yn4_5 = ((mode != 0) ? _unpack_bf16_46[5] * gw8[5] : _unpack_bf16_46[5]);
            float o4_5 = yn4_5 * rs4;
            float o2_4_5 = ((mode != 0) ? o4_5 : o4_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_4_5, w8[4], acc8[5]);
            float yn4_6 = ((mode != 0) ? _unpack_bf16_46[6] * gw8[6] : _unpack_bf16_46[6]);
            float o4_6 = yn4_6 * rs4;
            float o2_4_6 = ((mode != 0) ? o4_6 : o4_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_4_6, w8[4], acc8[6]);
            float yn4_7 = ((mode != 0) ? _unpack_bf16_46[7] * gw8[7] : _unpack_bf16_46[7]);
            float o4_7 = yn4_7 * rs4;
            float o2_4_7 = ((mode != 0) ? o4_7 : o4_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_4_7, w8[4], acc8[7]);
            float tot5 = warp_tab[parity + 5];
            tot5 = tot5 + warp_tab[parity + 8 + 5];
            tot5 = tot5 + warp_tab[parity + 16 + 5];
            tot5 = tot5 + warp_tab[parity + 24 + 5];
            tot5 = tot5 + warp_tab[parity + 32 + 5];
            tot5 = tot5 + warp_tab[parity + 40 + 5];
            tot5 = tot5 + warp_tab[parity + 48 + 5];
            tot5 = tot5 + warp_tab[parity + 56 + 5];
            tot5 = tot5 + warp_tab[parity + 64 + 5];
            tot5 = tot5 + warp_tab[parity + 72 + 5];
            tot5 = tot5 + warp_tab[parity + 80 + 5];
            tot5 = tot5 + warp_tab[parity + 88 + 5];
            tot5 = tot5 + warp_tab[parity + 96 + 5];
            tot5 = tot5 + warp_tab[parity + 104 + 5];
            tot5 = tot5 + warp_tab[parity + 112 + 5];
            tot5 = tot5 + warp_tab[parity + 120 + 5];
            float _sqrt_59;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_59) : "f"(__fmaf_rn(tot5, 0.00024414062f, eps)));
            float rs5 = __frcp_rn(_sqrt_59);
            float _unpack_bf16_47[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_47[_pair * 2])[0]), "=f"((&_unpack_bf16_47[_pair * 2])[1])
                    : "r"(rw[20 + _pair]));
            }
            float yn5_0 = ((mode != 0) ? _unpack_bf16_47[0] * gw8[0] : _unpack_bf16_47[0]);
            float o5_0 = yn5_0 * rs5;
            float o2_5_0 = ((mode != 0) ? o5_0 : o5_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_5_0, w8[5], acc8[0]);
            float yn5_1 = ((mode != 0) ? _unpack_bf16_47[1] * gw8[1] : _unpack_bf16_47[1]);
            float o5_1 = yn5_1 * rs5;
            float o2_5_1 = ((mode != 0) ? o5_1 : o5_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_5_1, w8[5], acc8[1]);
            float yn5_2 = ((mode != 0) ? _unpack_bf16_47[2] * gw8[2] : _unpack_bf16_47[2]);
            float o5_2 = yn5_2 * rs5;
            float o2_5_2 = ((mode != 0) ? o5_2 : o5_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_5_2, w8[5], acc8[2]);
            float yn5_3 = ((mode != 0) ? _unpack_bf16_47[3] * gw8[3] : _unpack_bf16_47[3]);
            float o5_3 = yn5_3 * rs5;
            float o2_5_3 = ((mode != 0) ? o5_3 : o5_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_5_3, w8[5], acc8[3]);
            float yn5_4 = ((mode != 0) ? _unpack_bf16_47[4] * gw8[4] : _unpack_bf16_47[4]);
            float o5_4 = yn5_4 * rs5;
            float o2_5_4 = ((mode != 0) ? o5_4 : o5_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_5_4, w8[5], acc8[4]);
            float yn5_5 = ((mode != 0) ? _unpack_bf16_47[5] * gw8[5] : _unpack_bf16_47[5]);
            float o5_5 = yn5_5 * rs5;
            float o2_5_5 = ((mode != 0) ? o5_5 : o5_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_5_5, w8[5], acc8[5]);
            float yn5_6 = ((mode != 0) ? _unpack_bf16_47[6] * gw8[6] : _unpack_bf16_47[6]);
            float o5_6 = yn5_6 * rs5;
            float o2_5_6 = ((mode != 0) ? o5_6 : o5_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_5_6, w8[5], acc8[6]);
            float yn5_7 = ((mode != 0) ? _unpack_bf16_47[7] * gw8[7] : _unpack_bf16_47[7]);
            float o5_7 = yn5_7 * rs5;
            float o2_5_7 = ((mode != 0) ? o5_7 : o5_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_5_7, w8[5], acc8[7]);
            float tot6 = warp_tab[parity + 6];
            tot6 = tot6 + warp_tab[parity + 8 + 6];
            tot6 = tot6 + warp_tab[parity + 16 + 6];
            tot6 = tot6 + warp_tab[parity + 24 + 6];
            tot6 = tot6 + warp_tab[parity + 32 + 6];
            tot6 = tot6 + warp_tab[parity + 40 + 6];
            tot6 = tot6 + warp_tab[parity + 48 + 6];
            tot6 = tot6 + warp_tab[parity + 56 + 6];
            tot6 = tot6 + warp_tab[parity + 64 + 6];
            tot6 = tot6 + warp_tab[parity + 72 + 6];
            tot6 = tot6 + warp_tab[parity + 80 + 6];
            tot6 = tot6 + warp_tab[parity + 88 + 6];
            tot6 = tot6 + warp_tab[parity + 96 + 6];
            tot6 = tot6 + warp_tab[parity + 104 + 6];
            tot6 = tot6 + warp_tab[parity + 112 + 6];
            tot6 = tot6 + warp_tab[parity + 120 + 6];
            float _sqrt_60;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_60) : "f"(__fmaf_rn(tot6, 0.00024414062f, eps)));
            float rs6 = __frcp_rn(_sqrt_60);
            float _unpack_bf16_48[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_48[_pair * 2])[0]), "=f"((&_unpack_bf16_48[_pair * 2])[1])
                    : "r"(rw[24 + _pair]));
            }
            float yn6_0 = ((mode != 0) ? _unpack_bf16_48[0] * gw8[0] : _unpack_bf16_48[0]);
            float o6_0 = yn6_0 * rs6;
            float o2_6_0 = ((mode != 0) ? o6_0 : o6_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_6_0, w8[6], acc8[0]);
            float yn6_1 = ((mode != 0) ? _unpack_bf16_48[1] * gw8[1] : _unpack_bf16_48[1]);
            float o6_1 = yn6_1 * rs6;
            float o2_6_1 = ((mode != 0) ? o6_1 : o6_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_6_1, w8[6], acc8[1]);
            float yn6_2 = ((mode != 0) ? _unpack_bf16_48[2] * gw8[2] : _unpack_bf16_48[2]);
            float o6_2 = yn6_2 * rs6;
            float o2_6_2 = ((mode != 0) ? o6_2 : o6_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_6_2, w8[6], acc8[2]);
            float yn6_3 = ((mode != 0) ? _unpack_bf16_48[3] * gw8[3] : _unpack_bf16_48[3]);
            float o6_3 = yn6_3 * rs6;
            float o2_6_3 = ((mode != 0) ? o6_3 : o6_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_6_3, w8[6], acc8[3]);
            float yn6_4 = ((mode != 0) ? _unpack_bf16_48[4] * gw8[4] : _unpack_bf16_48[4]);
            float o6_4 = yn6_4 * rs6;
            float o2_6_4 = ((mode != 0) ? o6_4 : o6_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_6_4, w8[6], acc8[4]);
            float yn6_5 = ((mode != 0) ? _unpack_bf16_48[5] * gw8[5] : _unpack_bf16_48[5]);
            float o6_5 = yn6_5 * rs6;
            float o2_6_5 = ((mode != 0) ? o6_5 : o6_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_6_5, w8[6], acc8[5]);
            float yn6_6 = ((mode != 0) ? _unpack_bf16_48[6] * gw8[6] : _unpack_bf16_48[6]);
            float o6_6 = yn6_6 * rs6;
            float o2_6_6 = ((mode != 0) ? o6_6 : o6_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_6_6, w8[6], acc8[6]);
            float yn6_7 = ((mode != 0) ? _unpack_bf16_48[7] * gw8[7] : _unpack_bf16_48[7]);
            float o6_7 = yn6_7 * rs6;
            float o2_6_7 = ((mode != 0) ? o6_7 : o6_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_6_7, w8[6], acc8[7]);
            float tot7 = warp_tab[parity + 7];
            tot7 = tot7 + warp_tab[parity + 8 + 7];
            tot7 = tot7 + warp_tab[parity + 16 + 7];
            tot7 = tot7 + warp_tab[parity + 24 + 7];
            tot7 = tot7 + warp_tab[parity + 32 + 7];
            tot7 = tot7 + warp_tab[parity + 40 + 7];
            tot7 = tot7 + warp_tab[parity + 48 + 7];
            tot7 = tot7 + warp_tab[parity + 56 + 7];
            tot7 = tot7 + warp_tab[parity + 64 + 7];
            tot7 = tot7 + warp_tab[parity + 72 + 7];
            tot7 = tot7 + warp_tab[parity + 80 + 7];
            tot7 = tot7 + warp_tab[parity + 88 + 7];
            tot7 = tot7 + warp_tab[parity + 96 + 7];
            tot7 = tot7 + warp_tab[parity + 104 + 7];
            tot7 = tot7 + warp_tab[parity + 112 + 7];
            tot7 = tot7 + warp_tab[parity + 120 + 7];
            float _sqrt_61;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_61) : "f"(__fmaf_rn(tot7, 0.00024414062f, eps)));
            float rs7 = __frcp_rn(_sqrt_61);
            float _unpack_bf16_49[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_49[_pair * 2])[0]), "=f"((&_unpack_bf16_49[_pair * 2])[1])
                    : "r"(rw[28 + _pair]));
            }
            float yn7_0 = ((mode != 0) ? _unpack_bf16_49[0] * gw8[0] : _unpack_bf16_49[0]);
            float o7_0 = yn7_0 * rs7;
            float o2_7_0 = ((mode != 0) ? o7_0 : o7_0 * gw8[0]);
            acc8[0] = __fmaf_rn(o2_7_0, w8[7], acc8[0]);
            float yn7_1 = ((mode != 0) ? _unpack_bf16_49[1] * gw8[1] : _unpack_bf16_49[1]);
            float o7_1 = yn7_1 * rs7;
            float o2_7_1 = ((mode != 0) ? o7_1 : o7_1 * gw8[1]);
            acc8[1] = __fmaf_rn(o2_7_1, w8[7], acc8[1]);
            float yn7_2 = ((mode != 0) ? _unpack_bf16_49[2] * gw8[2] : _unpack_bf16_49[2]);
            float o7_2 = yn7_2 * rs7;
            float o2_7_2 = ((mode != 0) ? o7_2 : o7_2 * gw8[2]);
            acc8[2] = __fmaf_rn(o2_7_2, w8[7], acc8[2]);
            float yn7_3 = ((mode != 0) ? _unpack_bf16_49[3] * gw8[3] : _unpack_bf16_49[3]);
            float o7_3 = yn7_3 * rs7;
            float o2_7_3 = ((mode != 0) ? o7_3 : o7_3 * gw8[3]);
            acc8[3] = __fmaf_rn(o2_7_3, w8[7], acc8[3]);
            float yn7_4 = ((mode != 0) ? _unpack_bf16_49[4] * gw8[4] : _unpack_bf16_49[4]);
            float o7_4 = yn7_4 * rs7;
            float o2_7_4 = ((mode != 0) ? o7_4 : o7_4 * gw8[4]);
            acc8[4] = __fmaf_rn(o2_7_4, w8[7], acc8[4]);
            float yn7_5 = ((mode != 0) ? _unpack_bf16_49[5] * gw8[5] : _unpack_bf16_49[5]);
            float o7_5 = yn7_5 * rs7;
            float o2_7_5 = ((mode != 0) ? o7_5 : o7_5 * gw8[5]);
            acc8[5] = __fmaf_rn(o2_7_5, w8[7], acc8[5]);
            float yn7_6 = ((mode != 0) ? _unpack_bf16_49[6] * gw8[6] : _unpack_bf16_49[6]);
            float o7_6 = yn7_6 * rs7;
            float o2_7_6 = ((mode != 0) ? o7_6 : o7_6 * gw8[6]);
            acc8[6] = __fmaf_rn(o2_7_6, w8[7], acc8[6]);
            float yn7_7 = ((mode != 0) ? _unpack_bf16_49[7] * gw8[7] : _unpack_bf16_49[7]);
            float o7_7 = yn7_7 * rs7;
            float o2_7_7 = ((mode != 0) ? o7_7 : o7_7 * gw8[7]);
            acc8[7] = __fmaf_rn(o2_7_7, w8[7], acc8[7]);
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
