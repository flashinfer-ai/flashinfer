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
#define SMEM_WARP_TAB_STAGE_BYTES 1024
#define SMEM_WARP_TAB_STRIDE 1024
#define SMEM_TOTAL 1024
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_tp_moe_finalize_rmsnorm_19d7bdd185b0109424d2(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int max_ctas, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, unsigned long long c_mcast, int* __restrict__ ctl)
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
    unsigned int sent8[8];
    sent8[0] = 2147516416;
    sent8[1] = 2147516416;
    sent8[2] = 2147516416;
    sent8[3] = 2147516416;
    sent8[4] = 2147516416;
    sent8[5] = 2147516416;
    sent8[6] = 2147516416;
    sent8[7] = 2147516416;
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
            long long dstw = push_gen_elems * (long long)gb + push_cta + col64 + rank64 * 8192;
            unsigned int rawa_0[4];
            unsigned int rawb_0[4];
            rawa_0[0] = packets[0];
            rawb_0[0] = packets[4];
            rawa_0[1] = packets[1];
            rawb_0[1] = packets[5];
            rawa_0[2] = packets[2];
            rawb_0[2] = packets[6];
            rawa_0[3] = packets[3];
            rawb_0[3] = packets[7];
            uint32_t _sana_0[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_11 = static_cast<uint32_t>(rawa_0[_p16s_word]);
                uint32_t _p16s_lo_11 = _p16s_carrier_11 & 0x0000ffffu;
                uint32_t _p16s_hi_11 = (_p16s_carrier_11 >> 16) & 0x0000ffffu;
                if (_p16s_lo_11 == 0x00008000u) _p16s_lo_11 = 0u;
                if (_p16s_hi_11 == 0x00008000u) _p16s_hi_11 = 0u;
                _sana_0[_p16s_word] = _p16s_lo_11 | (_p16s_hi_11 << 16);
            }
            uint32_t _sanb_0[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_12 = static_cast<uint32_t>(rawb_0[_p16s_word]);
                uint32_t _p16s_lo_12 = _p16s_carrier_12 & 0x0000ffffu;
                uint32_t _p16s_hi_12 = (_p16s_carrier_12 >> 16) & 0x0000ffffu;
                if (_p16s_lo_12 == 0x00008000u) _p16s_lo_12 = 0u;
                if (_p16s_hi_12 == 0x00008000u) _p16s_hi_12 = 0u;
                _sanb_0[_p16s_word] = _p16s_lo_12 | (_p16s_hi_12 << 16);
            }
            unsigned int pk8_0[8];
            pk8_0[0] = _sana_0[0];
            pk8_0[4] = _sanb_0[0];
            pk8_0[1] = _sana_0[1];
            pk8_0[5] = _sanb_0[1];
            pk8_0[2] = _sana_0[2];
            pk8_0[6] = _sanb_0[2];
            pk8_0[3] = _sana_0[3];
            pk8_0[7] = _sanb_0[3];
            if (owner == 0) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_0 + dstw), "r"((pk8_0)[0]), "r"((pk8_0)[1]), "r"((pk8_0)[2]), "r"((pk8_0)[3]), "r"((pk8_0)[4]), "r"((pk8_0)[5]), "r"((pk8_0)[6]), "r"((pk8_0)[7]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_1 + dstw), "r"((pk8_0)[0]), "r"((pk8_0)[1]), "r"((pk8_0)[2]), "r"((pk8_0)[3]), "r"((pk8_0)[4]), "r"((pk8_0)[5]), "r"((pk8_0)[6]), "r"((pk8_0)[7]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_2 + dstw), "r"((pk8_0)[0]), "r"((pk8_0)[1]), "r"((pk8_0)[2]), "r"((pk8_0)[3]), "r"((pk8_0)[4]), "r"((pk8_0)[5]), "r"((pk8_0)[6]), "r"((pk8_0)[7]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_3 + dstw), "r"((pk8_0)[0]), "r"((pk8_0)[1]), "r"((pk8_0)[2]), "r"((pk8_0)[3]), "r"((pk8_0)[4]), "r"((pk8_0)[5]), "r"((pk8_0)[6]), "r"((pk8_0)[7]) : "memory");
            }
            unsigned int rawa_1[4];
            unsigned int rawb_1[4];
            rawa_1[0] = packets[8];
            rawb_1[0] = packets[12];
            rawa_1[1] = packets[9];
            rawb_1[1] = packets[13];
            rawa_1[2] = packets[10];
            rawb_1[2] = packets[14];
            rawa_1[3] = packets[11];
            rawb_1[3] = packets[15];
            uint32_t _sana_1[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_13 = static_cast<uint32_t>(rawa_1[_p16s_word]);
                uint32_t _p16s_lo_13 = _p16s_carrier_13 & 0x0000ffffu;
                uint32_t _p16s_hi_13 = (_p16s_carrier_13 >> 16) & 0x0000ffffu;
                if (_p16s_lo_13 == 0x00008000u) _p16s_lo_13 = 0u;
                if (_p16s_hi_13 == 0x00008000u) _p16s_hi_13 = 0u;
                _sana_1[_p16s_word] = _p16s_lo_13 | (_p16s_hi_13 << 16);
            }
            uint32_t _sanb_1[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_14 = static_cast<uint32_t>(rawb_1[_p16s_word]);
                uint32_t _p16s_lo_14 = _p16s_carrier_14 & 0x0000ffffu;
                uint32_t _p16s_hi_14 = (_p16s_carrier_14 >> 16) & 0x0000ffffu;
                if (_p16s_lo_14 == 0x00008000u) _p16s_lo_14 = 0u;
                if (_p16s_hi_14 == 0x00008000u) _p16s_hi_14 = 0u;
                _sanb_1[_p16s_word] = _p16s_lo_14 | (_p16s_hi_14 << 16);
            }
            unsigned int pk8_1[8];
            pk8_1[0] = _sana_1[0];
            pk8_1[4] = _sanb_1[0];
            pk8_1[1] = _sana_1[1];
            pk8_1[5] = _sanb_1[1];
            pk8_1[2] = _sana_1[2];
            pk8_1[6] = _sanb_1[2];
            pk8_1[3] = _sana_1[3];
            pk8_1[7] = _sanb_1[3];
            if (owner == 0) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_0 + (dstw + 32768)), "r"((pk8_1)[0]), "r"((pk8_1)[1]), "r"((pk8_1)[2]), "r"((pk8_1)[3]), "r"((pk8_1)[4]), "r"((pk8_1)[5]), "r"((pk8_1)[6]), "r"((pk8_1)[7]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_1 + (dstw + 32768)), "r"((pk8_1)[0]), "r"((pk8_1)[1]), "r"((pk8_1)[2]), "r"((pk8_1)[3]), "r"((pk8_1)[4]), "r"((pk8_1)[5]), "r"((pk8_1)[6]), "r"((pk8_1)[7]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_2 + (dstw + 32768)), "r"((pk8_1)[0]), "r"((pk8_1)[1]), "r"((pk8_1)[2]), "r"((pk8_1)[3]), "r"((pk8_1)[4]), "r"((pk8_1)[5]), "r"((pk8_1)[6]), "r"((pk8_1)[7]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_3 + (dstw + 32768)), "r"((pk8_1)[0]), "r"((pk8_1)[1]), "r"((pk8_1)[2]), "r"((pk8_1)[3]), "r"((pk8_1)[4]), "r"((pk8_1)[5]), "r"((pk8_1)[6]), "r"((pk8_1)[7]) : "memory");
            }
            unsigned int rawa_2[4];
            unsigned int rawb_2[4];
            rawa_2[0] = packets[16];
            rawb_2[0] = packets[20];
            rawa_2[1] = packets[17];
            rawb_2[1] = packets[21];
            rawa_2[2] = packets[18];
            rawb_2[2] = packets[22];
            rawa_2[3] = packets[19];
            rawb_2[3] = packets[23];
            uint32_t _sana_2[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_15 = static_cast<uint32_t>(rawa_2[_p16s_word]);
                uint32_t _p16s_lo_15 = _p16s_carrier_15 & 0x0000ffffu;
                uint32_t _p16s_hi_15 = (_p16s_carrier_15 >> 16) & 0x0000ffffu;
                if (_p16s_lo_15 == 0x00008000u) _p16s_lo_15 = 0u;
                if (_p16s_hi_15 == 0x00008000u) _p16s_hi_15 = 0u;
                _sana_2[_p16s_word] = _p16s_lo_15 | (_p16s_hi_15 << 16);
            }
            uint32_t _sanb_2[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_16 = static_cast<uint32_t>(rawb_2[_p16s_word]);
                uint32_t _p16s_lo_16 = _p16s_carrier_16 & 0x0000ffffu;
                uint32_t _p16s_hi_16 = (_p16s_carrier_16 >> 16) & 0x0000ffffu;
                if (_p16s_lo_16 == 0x00008000u) _p16s_lo_16 = 0u;
                if (_p16s_hi_16 == 0x00008000u) _p16s_hi_16 = 0u;
                _sanb_2[_p16s_word] = _p16s_lo_16 | (_p16s_hi_16 << 16);
            }
            unsigned int pk8_2[8];
            pk8_2[0] = _sana_2[0];
            pk8_2[4] = _sanb_2[0];
            pk8_2[1] = _sana_2[1];
            pk8_2[5] = _sanb_2[1];
            pk8_2[2] = _sana_2[2];
            pk8_2[6] = _sanb_2[2];
            pk8_2[3] = _sana_2[3];
            pk8_2[7] = _sanb_2[3];
            if (owner == 0) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_0 + (dstw + 65536)), "r"((pk8_2)[0]), "r"((pk8_2)[1]), "r"((pk8_2)[2]), "r"((pk8_2)[3]), "r"((pk8_2)[4]), "r"((pk8_2)[5]), "r"((pk8_2)[6]), "r"((pk8_2)[7]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_1 + (dstw + 65536)), "r"((pk8_2)[0]), "r"((pk8_2)[1]), "r"((pk8_2)[2]), "r"((pk8_2)[3]), "r"((pk8_2)[4]), "r"((pk8_2)[5]), "r"((pk8_2)[6]), "r"((pk8_2)[7]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_2 + (dstw + 65536)), "r"((pk8_2)[0]), "r"((pk8_2)[1]), "r"((pk8_2)[2]), "r"((pk8_2)[3]), "r"((pk8_2)[4]), "r"((pk8_2)[5]), "r"((pk8_2)[6]), "r"((pk8_2)[7]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_3 + (dstw + 65536)), "r"((pk8_2)[0]), "r"((pk8_2)[1]), "r"((pk8_2)[2]), "r"((pk8_2)[3]), "r"((pk8_2)[4]), "r"((pk8_2)[5]), "r"((pk8_2)[6]), "r"((pk8_2)[7]) : "memory");
            }
            unsigned int rawa_3[4];
            unsigned int rawb_3[4];
            rawa_3[0] = packets[24];
            rawb_3[0] = packets[28];
            rawa_3[1] = packets[25];
            rawb_3[1] = packets[29];
            rawa_3[2] = packets[26];
            rawb_3[2] = packets[30];
            rawa_3[3] = packets[27];
            rawb_3[3] = packets[31];
            uint32_t _sana_3[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_17 = static_cast<uint32_t>(rawa_3[_p16s_word]);
                uint32_t _p16s_lo_17 = _p16s_carrier_17 & 0x0000ffffu;
                uint32_t _p16s_hi_17 = (_p16s_carrier_17 >> 16) & 0x0000ffffu;
                if (_p16s_lo_17 == 0x00008000u) _p16s_lo_17 = 0u;
                if (_p16s_hi_17 == 0x00008000u) _p16s_hi_17 = 0u;
                _sana_3[_p16s_word] = _p16s_lo_17 | (_p16s_hi_17 << 16);
            }
            uint32_t _sanb_3[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_18 = static_cast<uint32_t>(rawb_3[_p16s_word]);
                uint32_t _p16s_lo_18 = _p16s_carrier_18 & 0x0000ffffu;
                uint32_t _p16s_hi_18 = (_p16s_carrier_18 >> 16) & 0x0000ffffu;
                if (_p16s_lo_18 == 0x00008000u) _p16s_lo_18 = 0u;
                if (_p16s_hi_18 == 0x00008000u) _p16s_hi_18 = 0u;
                _sanb_3[_p16s_word] = _p16s_lo_18 | (_p16s_hi_18 << 16);
            }
            unsigned int pk8_3[8];
            pk8_3[0] = _sana_3[0];
            pk8_3[4] = _sanb_3[0];
            pk8_3[1] = _sana_3[1];
            pk8_3[5] = _sanb_3[1];
            pk8_3[2] = _sana_3[2];
            pk8_3[6] = _sanb_3[2];
            pk8_3[3] = _sana_3[3];
            pk8_3[7] = _sanb_3[3];
            if (owner == 0) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_0 + (dstw + 98304)), "r"((pk8_3)[0]), "r"((pk8_3)[1]), "r"((pk8_3)[2]), "r"((pk8_3)[3]), "r"((pk8_3)[4]), "r"((pk8_3)[5]), "r"((pk8_3)[6]), "r"((pk8_3)[7]) : "memory");
            }
            if (owner == 1) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_1 + (dstw + 98304)), "r"((pk8_3)[0]), "r"((pk8_3)[1]), "r"((pk8_3)[2]), "r"((pk8_3)[3]), "r"((pk8_3)[4]), "r"((pk8_3)[5]), "r"((pk8_3)[6]), "r"((pk8_3)[7]) : "memory");
            }
            if (owner == 2) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_2 + (dstw + 98304)), "r"((pk8_3)[0]), "r"((pk8_3)[1]), "r"((pk8_3)[2]), "r"((pk8_3)[3]), "r"((pk8_3)[4]), "r"((pk8_3)[5]), "r"((pk8_3)[6]), "r"((pk8_3)[7]) : "memory");
            }
            if (owner == 3) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(push_peers_3 + (dstw + 98304)), "r"((pk8_3)[0]), "r"((pk8_3)[1]), "r"((pk8_3)[2]), "r"((pk8_3)[3]), "r"((pk8_3)[4]), "r"((pk8_3)[5]), "r"((pk8_3)[6]), "r"((pk8_3)[7]) : "memory");
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
            uint32_t _sysv_min_22;
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]), "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]), "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 8192))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]), "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 16384))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]), "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 24576))) : "memory");
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
            float sum2_0[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_0[0]), "=&f"(sum2_0[1]) : "r"(_sysv_poll_group_0[8]), "f"(s8_0[0]), "f"(s8_0[1]));
            s8_0[0] = sum2_0[0];
            s8_0[1] = sum2_0[1];
            float sum2_1[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_1[0]), "=&f"(sum2_1[1]) : "r"(_sysv_poll_group_0[9]), "f"(s8_0[2]), "f"(s8_0[3]));
            s8_0[2] = sum2_1[0];
            s8_0[3] = sum2_1[1];
            float sum2_2[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_2[0]), "=&f"(sum2_2[1]) : "r"(_sysv_poll_group_0[10]), "f"(s8_0[4]), "f"(s8_0[5]));
            s8_0[4] = sum2_2[0];
            s8_0[5] = sum2_2[1];
            float sum2_3[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_3[0]), "=&f"(sum2_3[1]) : "r"(_sysv_poll_group_0[11]), "f"(s8_0[6]), "f"(s8_0[7]));
            s8_0[6] = sum2_3[0];
            s8_0[7] = sum2_3[1];
            float sum2_4[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_4[0]), "=&f"(sum2_4[1]) : "r"(_sysv_poll_group_0[16]), "f"(s8_0[0]), "f"(s8_0[1]));
            s8_0[0] = sum2_4[0];
            s8_0[1] = sum2_4[1];
            float sum2_5[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_5[0]), "=&f"(sum2_5[1]) : "r"(_sysv_poll_group_0[17]), "f"(s8_0[2]), "f"(s8_0[3]));
            s8_0[2] = sum2_5[0];
            s8_0[3] = sum2_5[1];
            float sum2_6[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_6[0]), "=&f"(sum2_6[1]) : "r"(_sysv_poll_group_0[18]), "f"(s8_0[4]), "f"(s8_0[5]));
            s8_0[4] = sum2_6[0];
            s8_0[5] = sum2_6[1];
            float sum2_7[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_7[0]), "=&f"(sum2_7[1]) : "r"(_sysv_poll_group_0[19]), "f"(s8_0[6]), "f"(s8_0[7]));
            s8_0[6] = sum2_7[0];
            s8_0[7] = sum2_7[1];
            float sum2_8[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_8[0]), "=&f"(sum2_8[1]) : "r"(_sysv_poll_group_0[24]), "f"(s8_0[0]), "f"(s8_0[1]));
            s8_0[0] = sum2_8[0];
            s8_0[1] = sum2_8[1];
            float sum2_9[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_9[0]), "=&f"(sum2_9[1]) : "r"(_sysv_poll_group_0[25]), "f"(s8_0[2]), "f"(s8_0[3]));
            s8_0[2] = sum2_9[0];
            s8_0[3] = sum2_9[1];
            float sum2_10[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_10[0]), "=&f"(sum2_10[1]) : "r"(_sysv_poll_group_0[26]), "f"(s8_0[4]), "f"(s8_0[5]));
            s8_0[4] = sum2_10[0];
            s8_0[5] = sum2_10[1];
            float sum2_11[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_11[0]), "=&f"(sum2_11[1]) : "r"(_sysv_poll_group_0[27]), "f"(s8_0[6]), "f"(s8_0[7]));
            s8_0[6] = sum2_11[0];
            s8_0[7] = sum2_11[1];
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
            float s8_1[8];
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
            s8_1[0] = part_1[0];
            s8_1[1] = part_1[1];
            s8_1[2] = part_1[2];
            s8_1[3] = part_1[3];
            s8_1[4] = part_1[4];
            s8_1[5] = part_1[5];
            s8_1[6] = part_1[6];
            s8_1[7] = part_1[7];
            float sum2_12[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_12[0]), "=&f"(sum2_12[1]) : "r"(_sysv_poll_group_0[12]), "f"(s8_1[0]), "f"(s8_1[1]));
            s8_1[0] = sum2_12[0];
            s8_1[1] = sum2_12[1];
            float sum2_13[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_13[0]), "=&f"(sum2_13[1]) : "r"(_sysv_poll_group_0[13]), "f"(s8_1[2]), "f"(s8_1[3]));
            s8_1[2] = sum2_13[0];
            s8_1[3] = sum2_13[1];
            float sum2_14[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_14[0]), "=&f"(sum2_14[1]) : "r"(_sysv_poll_group_0[14]), "f"(s8_1[4]), "f"(s8_1[5]));
            s8_1[4] = sum2_14[0];
            s8_1[5] = sum2_14[1];
            float sum2_15[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_15[0]), "=&f"(sum2_15[1]) : "r"(_sysv_poll_group_0[15]), "f"(s8_1[6]), "f"(s8_1[7]));
            s8_1[6] = sum2_15[0];
            s8_1[7] = sum2_15[1];
            float sum2_16[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_16[0]), "=&f"(sum2_16[1]) : "r"(_sysv_poll_group_0[20]), "f"(s8_1[0]), "f"(s8_1[1]));
            s8_1[0] = sum2_16[0];
            s8_1[1] = sum2_16[1];
            float sum2_17[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_17[0]), "=&f"(sum2_17[1]) : "r"(_sysv_poll_group_0[21]), "f"(s8_1[2]), "f"(s8_1[3]));
            s8_1[2] = sum2_17[0];
            s8_1[3] = sum2_17[1];
            float sum2_18[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_18[0]), "=&f"(sum2_18[1]) : "r"(_sysv_poll_group_0[22]), "f"(s8_1[4]), "f"(s8_1[5]));
            s8_1[4] = sum2_18[0];
            s8_1[5] = sum2_18[1];
            float sum2_19[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_19[0]), "=&f"(sum2_19[1]) : "r"(_sysv_poll_group_0[23]), "f"(s8_1[6]), "f"(s8_1[7]));
            s8_1[6] = sum2_19[0];
            s8_1[7] = sum2_19[1];
            float sum2_20[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_20[0]), "=&f"(sum2_20[1]) : "r"(_sysv_poll_group_0[28]), "f"(s8_1[0]), "f"(s8_1[1]));
            s8_1[0] = sum2_20[0];
            s8_1[1] = sum2_20[1];
            float sum2_21[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_21[0]), "=&f"(sum2_21[1]) : "r"(_sysv_poll_group_0[29]), "f"(s8_1[2]), "f"(s8_1[3]));
            s8_1[2] = sum2_21[0];
            s8_1[3] = sum2_21[1];
            float sum2_22[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_22[0]), "=&f"(sum2_22[1]) : "r"(_sysv_poll_group_0[30]), "f"(s8_1[4]), "f"(s8_1[5]));
            s8_1[4] = sum2_22[0];
            s8_1[5] = sum2_22[1];
            float sum2_23[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_23[0]), "=&f"(sum2_23[1]) : "r"(_sysv_poll_group_0[31]), "f"(s8_1[6]), "f"(s8_1[7]));
            s8_1[6] = sum2_23[0];
            s8_1[7] = sum2_23[1];
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
                const unsigned* _raw_stv8_24 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64))), "r"(_raw_stv8_24[0]), "r"(_raw_stv8_24[1]), "r"(_raw_stv8_24[2]), "r"(_raw_stv8_24[3]), "r"(_raw_stv8_24[4]), "r"(_raw_stv8_24[5]), "r"(_raw_stv8_24[6]), "r"(_raw_stv8_24[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_25 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 8192))), "r"(_raw_stv8_25[0]), "r"(_raw_stv8_25[1]), "r"(_raw_stv8_25[2]), "r"(_raw_stv8_25[3]), "r"(_raw_stv8_25[4]), "r"(_raw_stv8_25[5]), "r"(_raw_stv8_25[6]), "r"(_raw_stv8_25[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_26 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 16384))), "r"(_raw_stv8_26[0]), "r"(_raw_stv8_26[1]), "r"(_raw_stv8_26[2]), "r"(_raw_stv8_26[3]), "r"(_raw_stv8_26[4]), "r"(_raw_stv8_26[5]), "r"(_raw_stv8_26[6]), "r"(_raw_stv8_26[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_27 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 24576))), "r"(_raw_stv8_27[0]), "r"(_raw_stv8_27[1]), "r"(_raw_stv8_27[2]), "r"(_raw_stv8_27[3]), "r"(_raw_stv8_27[4]), "r"(_raw_stv8_27[5]), "r"(_raw_stv8_27[6]), "r"(_raw_stv8_27[7]) : "memory");
            }
            float _wrg_0 = ssq8[0];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_0 += __shfl_xor_sync(0xFFFFFFFF, _wrg_0, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8] = _wrg_0;
            }
            float _wrg_1 = ssq8[1];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_1 += __shfl_xor_sync(0xFFFFFFFF, _wrg_1, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 1] = _wrg_1;
            }
            uint32_t _sysv_poll_group_1[32];
            uint32_t _sysv_min_28;
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]), "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 32768))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]), "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 40960))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]), "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 49152))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]), "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 57344))) : "memory");
                uint32_t _sysv_min_29_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_0) : "r"(_sysv_poll_group_1[0]), "r"(_sysv_poll_group_1[1]));
                uint32_t _sysv_min_29_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_1) : "r"(_sysv_poll_group_1[2]), "r"(_sysv_poll_group_1[3]));
                uint32_t _sysv_min_29_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_2) : "r"(_sysv_min_29_0), "r"(_sysv_min_29_1));
                uint32_t _sysv_min_29_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_3) : "r"(_sysv_poll_group_1[4]), "r"(_sysv_poll_group_1[5]));
                uint32_t _sysv_min_29_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_4) : "r"(_sysv_poll_group_1[6]), "r"(_sysv_poll_group_1[7]));
                uint32_t _sysv_min_29_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_5) : "r"(_sysv_min_29_3), "r"(_sysv_min_29_4));
                uint32_t _sysv_min_29_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_6) : "r"(_sysv_min_29_2), "r"(_sysv_min_29_5));
                uint32_t _sysv_min_29_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_7) : "r"(_sysv_poll_group_1[8]), "r"(_sysv_poll_group_1[9]));
                uint32_t _sysv_min_29_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_8) : "r"(_sysv_poll_group_1[10]), "r"(_sysv_poll_group_1[11]));
                uint32_t _sysv_min_29_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_9) : "r"(_sysv_min_29_7), "r"(_sysv_min_29_8));
                uint32_t _sysv_min_29_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_10) : "r"(_sysv_poll_group_1[12]), "r"(_sysv_poll_group_1[13]));
                uint32_t _sysv_min_29_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_11) : "r"(_sysv_poll_group_1[14]), "r"(_sysv_poll_group_1[15]));
                uint32_t _sysv_min_29_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_12) : "r"(_sysv_min_29_10), "r"(_sysv_min_29_11));
                uint32_t _sysv_min_29_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_13) : "r"(_sysv_min_29_9), "r"(_sysv_min_29_12));
                uint32_t _sysv_min_29_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_14) : "r"(_sysv_min_29_6), "r"(_sysv_min_29_13));
                uint32_t _sysv_min_29_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_15) : "r"(_sysv_poll_group_1[16]), "r"(_sysv_poll_group_1[17]));
                uint32_t _sysv_min_29_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_16) : "r"(_sysv_poll_group_1[18]), "r"(_sysv_poll_group_1[19]));
                uint32_t _sysv_min_29_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_17) : "r"(_sysv_min_29_15), "r"(_sysv_min_29_16));
                uint32_t _sysv_min_29_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_18) : "r"(_sysv_poll_group_1[20]), "r"(_sysv_poll_group_1[21]));
                uint32_t _sysv_min_29_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_19) : "r"(_sysv_poll_group_1[22]), "r"(_sysv_poll_group_1[23]));
                uint32_t _sysv_min_29_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_20) : "r"(_sysv_min_29_18), "r"(_sysv_min_29_19));
                uint32_t _sysv_min_29_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_21) : "r"(_sysv_min_29_17), "r"(_sysv_min_29_20));
                uint32_t _sysv_min_29_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_22) : "r"(_sysv_poll_group_1[24]), "r"(_sysv_poll_group_1[25]));
                uint32_t _sysv_min_29_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_23) : "r"(_sysv_poll_group_1[26]), "r"(_sysv_poll_group_1[27]));
                uint32_t _sysv_min_29_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_24) : "r"(_sysv_min_29_22), "r"(_sysv_min_29_23));
                uint32_t _sysv_min_29_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_25) : "r"(_sysv_poll_group_1[28]), "r"(_sysv_poll_group_1[29]));
                uint32_t _sysv_min_29_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_26) : "r"(_sysv_poll_group_1[30]), "r"(_sysv_poll_group_1[31]));
                uint32_t _sysv_min_29_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_27) : "r"(_sysv_min_29_25), "r"(_sysv_min_29_26));
                uint32_t _sysv_min_29_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_28) : "r"(_sysv_min_29_24), "r"(_sysv_min_29_27));
                uint32_t _sysv_min_29_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_29) : "r"(_sysv_min_29_21), "r"(_sysv_min_29_28));
                uint32_t _sysv_min_29_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_29_30) : "r"(_sysv_min_29_14), "r"(_sysv_min_29_29));
                _sysv_min_28 = _sysv_min_29_30;
            } while (
                ((_sysv_min_28 & 0xffffu) == 0x8000u) || ((_sysv_min_28 >> 16) == 0x8000u)
            );
            float s8_2[8];
            float part_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_2[_pair * 2])[0]), "=f"((&part_2[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[_pair]));
            }
            s8_2[0] = part_2[0];
            s8_2[1] = part_2[1];
            s8_2[2] = part_2[2];
            s8_2[3] = part_2[3];
            s8_2[4] = part_2[4];
            s8_2[5] = part_2[5];
            s8_2[6] = part_2[6];
            s8_2[7] = part_2[7];
            float sum2_24[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_24[0]), "=&f"(sum2_24[1]) : "r"(_sysv_poll_group_1[8]), "f"(s8_2[0]), "f"(s8_2[1]));
            s8_2[0] = sum2_24[0];
            s8_2[1] = sum2_24[1];
            float sum2_25[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_25[0]), "=&f"(sum2_25[1]) : "r"(_sysv_poll_group_1[9]), "f"(s8_2[2]), "f"(s8_2[3]));
            s8_2[2] = sum2_25[0];
            s8_2[3] = sum2_25[1];
            float sum2_26[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_26[0]), "=&f"(sum2_26[1]) : "r"(_sysv_poll_group_1[10]), "f"(s8_2[4]), "f"(s8_2[5]));
            s8_2[4] = sum2_26[0];
            s8_2[5] = sum2_26[1];
            float sum2_27[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_27[0]), "=&f"(sum2_27[1]) : "r"(_sysv_poll_group_1[11]), "f"(s8_2[6]), "f"(s8_2[7]));
            s8_2[6] = sum2_27[0];
            s8_2[7] = sum2_27[1];
            float sum2_28[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_28[0]), "=&f"(sum2_28[1]) : "r"(_sysv_poll_group_1[16]), "f"(s8_2[0]), "f"(s8_2[1]));
            s8_2[0] = sum2_28[0];
            s8_2[1] = sum2_28[1];
            float sum2_29[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_29[0]), "=&f"(sum2_29[1]) : "r"(_sysv_poll_group_1[17]), "f"(s8_2[2]), "f"(s8_2[3]));
            s8_2[2] = sum2_29[0];
            s8_2[3] = sum2_29[1];
            float sum2_30[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_30[0]), "=&f"(sum2_30[1]) : "r"(_sysv_poll_group_1[18]), "f"(s8_2[4]), "f"(s8_2[5]));
            s8_2[4] = sum2_30[0];
            s8_2[5] = sum2_30[1];
            float sum2_31[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_31[0]), "=&f"(sum2_31[1]) : "r"(_sysv_poll_group_1[19]), "f"(s8_2[6]), "f"(s8_2[7]));
            s8_2[6] = sum2_31[0];
            s8_2[7] = sum2_31[1];
            float sum2_32[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_32[0]), "=&f"(sum2_32[1]) : "r"(_sysv_poll_group_1[24]), "f"(s8_2[0]), "f"(s8_2[1]));
            s8_2[0] = sum2_32[0];
            s8_2[1] = sum2_32[1];
            float sum2_33[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_33[0]), "=&f"(sum2_33[1]) : "r"(_sysv_poll_group_1[25]), "f"(s8_2[2]), "f"(s8_2[3]));
            s8_2[2] = sum2_33[0];
            s8_2[3] = sum2_33[1];
            float sum2_34[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_34[0]), "=&f"(sum2_34[1]) : "r"(_sysv_poll_group_1[26]), "f"(s8_2[4]), "f"(s8_2[5]));
            s8_2[4] = sum2_34[0];
            s8_2[5] = sum2_34[1];
            float sum2_35[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_35[0]), "=&f"(sum2_35[1]) : "r"(_sysv_poll_group_1[27]), "f"(s8_2[6]), "f"(s8_2[7]));
            s8_2[6] = sum2_35[0];
            s8_2[7] = sum2_35[1];
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
            float s8_3[8];
            float part_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_3[_pair * 2])[0]), "=f"((&part_3[_pair * 2])[1])
                    : "r"(_sysv_poll_group_1[4 + _pair]));
            }
            s8_3[0] = part_3[0];
            s8_3[1] = part_3[1];
            s8_3[2] = part_3[2];
            s8_3[3] = part_3[3];
            s8_3[4] = part_3[4];
            s8_3[5] = part_3[5];
            s8_3[6] = part_3[6];
            s8_3[7] = part_3[7];
            float sum2_36[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_36[0]), "=&f"(sum2_36[1]) : "r"(_sysv_poll_group_1[12]), "f"(s8_3[0]), "f"(s8_3[1]));
            s8_3[0] = sum2_36[0];
            s8_3[1] = sum2_36[1];
            float sum2_37[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_37[0]), "=&f"(sum2_37[1]) : "r"(_sysv_poll_group_1[13]), "f"(s8_3[2]), "f"(s8_3[3]));
            s8_3[2] = sum2_37[0];
            s8_3[3] = sum2_37[1];
            float sum2_38[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_38[0]), "=&f"(sum2_38[1]) : "r"(_sysv_poll_group_1[14]), "f"(s8_3[4]), "f"(s8_3[5]));
            s8_3[4] = sum2_38[0];
            s8_3[5] = sum2_38[1];
            float sum2_39[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_39[0]), "=&f"(sum2_39[1]) : "r"(_sysv_poll_group_1[15]), "f"(s8_3[6]), "f"(s8_3[7]));
            s8_3[6] = sum2_39[0];
            s8_3[7] = sum2_39[1];
            float sum2_40[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_40[0]), "=&f"(sum2_40[1]) : "r"(_sysv_poll_group_1[20]), "f"(s8_3[0]), "f"(s8_3[1]));
            s8_3[0] = sum2_40[0];
            s8_3[1] = sum2_40[1];
            float sum2_41[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_41[0]), "=&f"(sum2_41[1]) : "r"(_sysv_poll_group_1[21]), "f"(s8_3[2]), "f"(s8_3[3]));
            s8_3[2] = sum2_41[0];
            s8_3[3] = sum2_41[1];
            float sum2_42[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_42[0]), "=&f"(sum2_42[1]) : "r"(_sysv_poll_group_1[22]), "f"(s8_3[4]), "f"(s8_3[5]));
            s8_3[4] = sum2_42[0];
            s8_3[5] = sum2_42[1];
            float sum2_43[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_43[0]), "=&f"(sum2_43[1]) : "r"(_sysv_poll_group_1[23]), "f"(s8_3[6]), "f"(s8_3[7]));
            s8_3[6] = sum2_43[0];
            s8_3[7] = sum2_43[1];
            float sum2_44[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_44[0]), "=&f"(sum2_44[1]) : "r"(_sysv_poll_group_1[28]), "f"(s8_3[0]), "f"(s8_3[1]));
            s8_3[0] = sum2_44[0];
            s8_3[1] = sum2_44[1];
            float sum2_45[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_45[0]), "=&f"(sum2_45[1]) : "r"(_sysv_poll_group_1[29]), "f"(s8_3[2]), "f"(s8_3[3]));
            s8_3[2] = sum2_45[0];
            s8_3[3] = sum2_45[1];
            float sum2_46[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_46[0]), "=&f"(sum2_46[1]) : "r"(_sysv_poll_group_1[30]), "f"(s8_3[4]), "f"(s8_3[5]));
            s8_3[4] = sum2_46[0];
            s8_3[5] = sum2_46[1];
            float sum2_47[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_47[0]), "=&f"(sum2_47[1]) : "r"(_sysv_poll_group_1[31]), "f"(s8_3[6]), "f"(s8_3[7]));
            s8_3[6] = sum2_47[0];
            s8_3[7] = sum2_47[1];
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
                const unsigned* _raw_stv8_30 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 32768))), "r"(_raw_stv8_30[0]), "r"(_raw_stv8_30[1]), "r"(_raw_stv8_30[2]), "r"(_raw_stv8_30[3]), "r"(_raw_stv8_30[4]), "r"(_raw_stv8_30[5]), "r"(_raw_stv8_30[6]), "r"(_raw_stv8_30[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_31 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 40960))), "r"(_raw_stv8_31[0]), "r"(_raw_stv8_31[1]), "r"(_raw_stv8_31[2]), "r"(_raw_stv8_31[3]), "r"(_raw_stv8_31[4]), "r"(_raw_stv8_31[5]), "r"(_raw_stv8_31[6]), "r"(_raw_stv8_31[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_32 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 49152))), "r"(_raw_stv8_32[0]), "r"(_raw_stv8_32[1]), "r"(_raw_stv8_32[2]), "r"(_raw_stv8_32[3]), "r"(_raw_stv8_32[4]), "r"(_raw_stv8_32[5]), "r"(_raw_stv8_32[6]), "r"(_raw_stv8_32[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_33 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 57344))), "r"(_raw_stv8_33[0]), "r"(_raw_stv8_33[1]), "r"(_raw_stv8_33[2]), "r"(_raw_stv8_33[3]), "r"(_raw_stv8_33[4]), "r"(_raw_stv8_33[5]), "r"(_raw_stv8_33[6]), "r"(_raw_stv8_33[7]) : "memory");
            }
            float _wrg_2 = ssq8[2];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_2 += __shfl_xor_sync(0xFFFFFFFF, _wrg_2, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 2] = _wrg_2;
            }
            float _wrg_3 = ssq8[3];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_3 += __shfl_xor_sync(0xFFFFFFFF, _wrg_3, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 3] = _wrg_3;
            }
            uint32_t _sysv_poll_group_2[32];
            uint32_t _sysv_min_34;
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_2[0]), "=r"(_sysv_poll_group_2[1]), "=r"(_sysv_poll_group_2[2]), "=r"(_sysv_poll_group_2[3]), "=r"(_sysv_poll_group_2[4]), "=r"(_sysv_poll_group_2[5]), "=r"(_sysv_poll_group_2[6]), "=r"(_sysv_poll_group_2[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 65536))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_2[8]), "=r"(_sysv_poll_group_2[9]), "=r"(_sysv_poll_group_2[10]), "=r"(_sysv_poll_group_2[11]), "=r"(_sysv_poll_group_2[12]), "=r"(_sysv_poll_group_2[13]), "=r"(_sysv_poll_group_2[14]), "=r"(_sysv_poll_group_2[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 73728))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_2[16]), "=r"(_sysv_poll_group_2[17]), "=r"(_sysv_poll_group_2[18]), "=r"(_sysv_poll_group_2[19]), "=r"(_sysv_poll_group_2[20]), "=r"(_sysv_poll_group_2[21]), "=r"(_sysv_poll_group_2[22]), "=r"(_sysv_poll_group_2[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 81920))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_2[24]), "=r"(_sysv_poll_group_2[25]), "=r"(_sysv_poll_group_2[26]), "=r"(_sysv_poll_group_2[27]), "=r"(_sysv_poll_group_2[28]), "=r"(_sysv_poll_group_2[29]), "=r"(_sysv_poll_group_2[30]), "=r"(_sysv_poll_group_2[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 90112))) : "memory");
                uint32_t _sysv_min_35_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_0) : "r"(_sysv_poll_group_2[0]), "r"(_sysv_poll_group_2[1]));
                uint32_t _sysv_min_35_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_1) : "r"(_sysv_poll_group_2[2]), "r"(_sysv_poll_group_2[3]));
                uint32_t _sysv_min_35_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_2) : "r"(_sysv_min_35_0), "r"(_sysv_min_35_1));
                uint32_t _sysv_min_35_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_3) : "r"(_sysv_poll_group_2[4]), "r"(_sysv_poll_group_2[5]));
                uint32_t _sysv_min_35_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_4) : "r"(_sysv_poll_group_2[6]), "r"(_sysv_poll_group_2[7]));
                uint32_t _sysv_min_35_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_5) : "r"(_sysv_min_35_3), "r"(_sysv_min_35_4));
                uint32_t _sysv_min_35_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_6) : "r"(_sysv_min_35_2), "r"(_sysv_min_35_5));
                uint32_t _sysv_min_35_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_7) : "r"(_sysv_poll_group_2[8]), "r"(_sysv_poll_group_2[9]));
                uint32_t _sysv_min_35_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_8) : "r"(_sysv_poll_group_2[10]), "r"(_sysv_poll_group_2[11]));
                uint32_t _sysv_min_35_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_9) : "r"(_sysv_min_35_7), "r"(_sysv_min_35_8));
                uint32_t _sysv_min_35_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_10) : "r"(_sysv_poll_group_2[12]), "r"(_sysv_poll_group_2[13]));
                uint32_t _sysv_min_35_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_11) : "r"(_sysv_poll_group_2[14]), "r"(_sysv_poll_group_2[15]));
                uint32_t _sysv_min_35_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_12) : "r"(_sysv_min_35_10), "r"(_sysv_min_35_11));
                uint32_t _sysv_min_35_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_13) : "r"(_sysv_min_35_9), "r"(_sysv_min_35_12));
                uint32_t _sysv_min_35_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_14) : "r"(_sysv_min_35_6), "r"(_sysv_min_35_13));
                uint32_t _sysv_min_35_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_15) : "r"(_sysv_poll_group_2[16]), "r"(_sysv_poll_group_2[17]));
                uint32_t _sysv_min_35_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_16) : "r"(_sysv_poll_group_2[18]), "r"(_sysv_poll_group_2[19]));
                uint32_t _sysv_min_35_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_17) : "r"(_sysv_min_35_15), "r"(_sysv_min_35_16));
                uint32_t _sysv_min_35_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_18) : "r"(_sysv_poll_group_2[20]), "r"(_sysv_poll_group_2[21]));
                uint32_t _sysv_min_35_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_19) : "r"(_sysv_poll_group_2[22]), "r"(_sysv_poll_group_2[23]));
                uint32_t _sysv_min_35_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_20) : "r"(_sysv_min_35_18), "r"(_sysv_min_35_19));
                uint32_t _sysv_min_35_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_21) : "r"(_sysv_min_35_17), "r"(_sysv_min_35_20));
                uint32_t _sysv_min_35_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_22) : "r"(_sysv_poll_group_2[24]), "r"(_sysv_poll_group_2[25]));
                uint32_t _sysv_min_35_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_23) : "r"(_sysv_poll_group_2[26]), "r"(_sysv_poll_group_2[27]));
                uint32_t _sysv_min_35_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_24) : "r"(_sysv_min_35_22), "r"(_sysv_min_35_23));
                uint32_t _sysv_min_35_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_25) : "r"(_sysv_poll_group_2[28]), "r"(_sysv_poll_group_2[29]));
                uint32_t _sysv_min_35_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_26) : "r"(_sysv_poll_group_2[30]), "r"(_sysv_poll_group_2[31]));
                uint32_t _sysv_min_35_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_27) : "r"(_sysv_min_35_25), "r"(_sysv_min_35_26));
                uint32_t _sysv_min_35_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_28) : "r"(_sysv_min_35_24), "r"(_sysv_min_35_27));
                uint32_t _sysv_min_35_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_29) : "r"(_sysv_min_35_21), "r"(_sysv_min_35_28));
                uint32_t _sysv_min_35_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_35_30) : "r"(_sysv_min_35_14), "r"(_sysv_min_35_29));
                _sysv_min_34 = _sysv_min_35_30;
            } while (
                ((_sysv_min_34 & 0xffffu) == 0x8000u) || ((_sysv_min_34 >> 16) == 0x8000u)
            );
            float s8_4[8];
            float part_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_4[_pair * 2])[0]), "=f"((&part_4[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[_pair]));
            }
            s8_4[0] = part_4[0];
            s8_4[1] = part_4[1];
            s8_4[2] = part_4[2];
            s8_4[3] = part_4[3];
            s8_4[4] = part_4[4];
            s8_4[5] = part_4[5];
            s8_4[6] = part_4[6];
            s8_4[7] = part_4[7];
            float sum2_48[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_48[0]), "=&f"(sum2_48[1]) : "r"(_sysv_poll_group_2[8]), "f"(s8_4[0]), "f"(s8_4[1]));
            s8_4[0] = sum2_48[0];
            s8_4[1] = sum2_48[1];
            float sum2_49[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_49[0]), "=&f"(sum2_49[1]) : "r"(_sysv_poll_group_2[9]), "f"(s8_4[2]), "f"(s8_4[3]));
            s8_4[2] = sum2_49[0];
            s8_4[3] = sum2_49[1];
            float sum2_50[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_50[0]), "=&f"(sum2_50[1]) : "r"(_sysv_poll_group_2[10]), "f"(s8_4[4]), "f"(s8_4[5]));
            s8_4[4] = sum2_50[0];
            s8_4[5] = sum2_50[1];
            float sum2_51[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_51[0]), "=&f"(sum2_51[1]) : "r"(_sysv_poll_group_2[11]), "f"(s8_4[6]), "f"(s8_4[7]));
            s8_4[6] = sum2_51[0];
            s8_4[7] = sum2_51[1];
            float sum2_52[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_52[0]), "=&f"(sum2_52[1]) : "r"(_sysv_poll_group_2[16]), "f"(s8_4[0]), "f"(s8_4[1]));
            s8_4[0] = sum2_52[0];
            s8_4[1] = sum2_52[1];
            float sum2_53[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_53[0]), "=&f"(sum2_53[1]) : "r"(_sysv_poll_group_2[17]), "f"(s8_4[2]), "f"(s8_4[3]));
            s8_4[2] = sum2_53[0];
            s8_4[3] = sum2_53[1];
            float sum2_54[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_54[0]), "=&f"(sum2_54[1]) : "r"(_sysv_poll_group_2[18]), "f"(s8_4[4]), "f"(s8_4[5]));
            s8_4[4] = sum2_54[0];
            s8_4[5] = sum2_54[1];
            float sum2_55[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_55[0]), "=&f"(sum2_55[1]) : "r"(_sysv_poll_group_2[19]), "f"(s8_4[6]), "f"(s8_4[7]));
            s8_4[6] = sum2_55[0];
            s8_4[7] = sum2_55[1];
            float sum2_56[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_56[0]), "=&f"(sum2_56[1]) : "r"(_sysv_poll_group_2[24]), "f"(s8_4[0]), "f"(s8_4[1]));
            s8_4[0] = sum2_56[0];
            s8_4[1] = sum2_56[1];
            float sum2_57[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_57[0]), "=&f"(sum2_57[1]) : "r"(_sysv_poll_group_2[25]), "f"(s8_4[2]), "f"(s8_4[3]));
            s8_4[2] = sum2_57[0];
            s8_4[3] = sum2_57[1];
            float sum2_58[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_58[0]), "=&f"(sum2_58[1]) : "r"(_sysv_poll_group_2[26]), "f"(s8_4[4]), "f"(s8_4[5]));
            s8_4[4] = sum2_58[0];
            s8_4[5] = sum2_58[1];
            float sum2_59[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_59[0]), "=&f"(sum2_59[1]) : "r"(_sysv_poll_group_2[27]), "f"(s8_4[6]), "f"(s8_4[7]));
            s8_4[6] = sum2_59[0];
            s8_4[7] = sum2_59[1];
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
            float s8_5[8];
            float part_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_5[_pair * 2])[0]), "=f"((&part_5[_pair * 2])[1])
                    : "r"(_sysv_poll_group_2[4 + _pair]));
            }
            s8_5[0] = part_5[0];
            s8_5[1] = part_5[1];
            s8_5[2] = part_5[2];
            s8_5[3] = part_5[3];
            s8_5[4] = part_5[4];
            s8_5[5] = part_5[5];
            s8_5[6] = part_5[6];
            s8_5[7] = part_5[7];
            float sum2_60[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_60[0]), "=&f"(sum2_60[1]) : "r"(_sysv_poll_group_2[12]), "f"(s8_5[0]), "f"(s8_5[1]));
            s8_5[0] = sum2_60[0];
            s8_5[1] = sum2_60[1];
            float sum2_61[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_61[0]), "=&f"(sum2_61[1]) : "r"(_sysv_poll_group_2[13]), "f"(s8_5[2]), "f"(s8_5[3]));
            s8_5[2] = sum2_61[0];
            s8_5[3] = sum2_61[1];
            float sum2_62[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_62[0]), "=&f"(sum2_62[1]) : "r"(_sysv_poll_group_2[14]), "f"(s8_5[4]), "f"(s8_5[5]));
            s8_5[4] = sum2_62[0];
            s8_5[5] = sum2_62[1];
            float sum2_63[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_63[0]), "=&f"(sum2_63[1]) : "r"(_sysv_poll_group_2[15]), "f"(s8_5[6]), "f"(s8_5[7]));
            s8_5[6] = sum2_63[0];
            s8_5[7] = sum2_63[1];
            float sum2_64[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_64[0]), "=&f"(sum2_64[1]) : "r"(_sysv_poll_group_2[20]), "f"(s8_5[0]), "f"(s8_5[1]));
            s8_5[0] = sum2_64[0];
            s8_5[1] = sum2_64[1];
            float sum2_65[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_65[0]), "=&f"(sum2_65[1]) : "r"(_sysv_poll_group_2[21]), "f"(s8_5[2]), "f"(s8_5[3]));
            s8_5[2] = sum2_65[0];
            s8_5[3] = sum2_65[1];
            float sum2_66[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_66[0]), "=&f"(sum2_66[1]) : "r"(_sysv_poll_group_2[22]), "f"(s8_5[4]), "f"(s8_5[5]));
            s8_5[4] = sum2_66[0];
            s8_5[5] = sum2_66[1];
            float sum2_67[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_67[0]), "=&f"(sum2_67[1]) : "r"(_sysv_poll_group_2[23]), "f"(s8_5[6]), "f"(s8_5[7]));
            s8_5[6] = sum2_67[0];
            s8_5[7] = sum2_67[1];
            float sum2_68[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_68[0]), "=&f"(sum2_68[1]) : "r"(_sysv_poll_group_2[28]), "f"(s8_5[0]), "f"(s8_5[1]));
            s8_5[0] = sum2_68[0];
            s8_5[1] = sum2_68[1];
            float sum2_69[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_69[0]), "=&f"(sum2_69[1]) : "r"(_sysv_poll_group_2[29]), "f"(s8_5[2]), "f"(s8_5[3]));
            s8_5[2] = sum2_69[0];
            s8_5[3] = sum2_69[1];
            float sum2_70[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_70[0]), "=&f"(sum2_70[1]) : "r"(_sysv_poll_group_2[30]), "f"(s8_5[4]), "f"(s8_5[5]));
            s8_5[4] = sum2_70[0];
            s8_5[5] = sum2_70[1];
            float sum2_71[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_71[0]), "=&f"(sum2_71[1]) : "r"(_sysv_poll_group_2[31]), "f"(s8_5[6]), "f"(s8_5[7]));
            s8_5[6] = sum2_71[0];
            s8_5[7] = sum2_71[1];
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
                const unsigned* _raw_stv8_36 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 65536))), "r"(_raw_stv8_36[0]), "r"(_raw_stv8_36[1]), "r"(_raw_stv8_36[2]), "r"(_raw_stv8_36[3]), "r"(_raw_stv8_36[4]), "r"(_raw_stv8_36[5]), "r"(_raw_stv8_36[6]), "r"(_raw_stv8_36[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_37 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 73728))), "r"(_raw_stv8_37[0]), "r"(_raw_stv8_37[1]), "r"(_raw_stv8_37[2]), "r"(_raw_stv8_37[3]), "r"(_raw_stv8_37[4]), "r"(_raw_stv8_37[5]), "r"(_raw_stv8_37[6]), "r"(_raw_stv8_37[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_38 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 81920))), "r"(_raw_stv8_38[0]), "r"(_raw_stv8_38[1]), "r"(_raw_stv8_38[2]), "r"(_raw_stv8_38[3]), "r"(_raw_stv8_38[4]), "r"(_raw_stv8_38[5]), "r"(_raw_stv8_38[6]), "r"(_raw_stv8_38[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_39 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 90112))), "r"(_raw_stv8_39[0]), "r"(_raw_stv8_39[1]), "r"(_raw_stv8_39[2]), "r"(_raw_stv8_39[3]), "r"(_raw_stv8_39[4]), "r"(_raw_stv8_39[5]), "r"(_raw_stv8_39[6]), "r"(_raw_stv8_39[7]) : "memory");
            }
            float _wrg_4 = ssq8[4];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_4 += __shfl_xor_sync(0xFFFFFFFF, _wrg_4, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 4] = _wrg_4;
            }
            float _wrg_5 = ssq8[5];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_5 += __shfl_xor_sync(0xFFFFFFFF, _wrg_5, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 5] = _wrg_5;
            }
            uint32_t _sysv_poll_group_3[32];
            uint32_t _sysv_min_40;
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_3[0]), "=r"(_sysv_poll_group_3[1]), "=r"(_sysv_poll_group_3[2]), "=r"(_sysv_poll_group_3[3]), "=r"(_sysv_poll_group_3[4]), "=r"(_sysv_poll_group_3[5]), "=r"(_sysv_poll_group_3[6]), "=r"(_sysv_poll_group_3[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 98304))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_3[8]), "=r"(_sysv_poll_group_3[9]), "=r"(_sysv_poll_group_3[10]), "=r"(_sysv_poll_group_3[11]), "=r"(_sysv_poll_group_3[12]), "=r"(_sysv_poll_group_3[13]), "=r"(_sysv_poll_group_3[14]), "=r"(_sysv_poll_group_3[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 106496))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_3[16]), "=r"(_sysv_poll_group_3[17]), "=r"(_sysv_poll_group_3[18]), "=r"(_sysv_poll_group_3[19]), "=r"(_sysv_poll_group_3[20]), "=r"(_sysv_poll_group_3[21]), "=r"(_sysv_poll_group_3[22]), "=r"(_sysv_poll_group_3[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 114688))) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll_group_3[24]), "=r"(_sysv_poll_group_3[25]), "=r"(_sysv_poll_group_3[26]), "=r"(_sysv_poll_group_3[27]), "=r"(_sysv_poll_group_3[28]), "=r"(_sysv_poll_group_3[29]), "=r"(_sysv_poll_group_3[30]), "=r"(_sysv_poll_group_3[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + col64 + 122880))) : "memory");
                uint32_t _sysv_min_41_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_0) : "r"(_sysv_poll_group_3[0]), "r"(_sysv_poll_group_3[1]));
                uint32_t _sysv_min_41_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_1) : "r"(_sysv_poll_group_3[2]), "r"(_sysv_poll_group_3[3]));
                uint32_t _sysv_min_41_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_2) : "r"(_sysv_min_41_0), "r"(_sysv_min_41_1));
                uint32_t _sysv_min_41_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_3) : "r"(_sysv_poll_group_3[4]), "r"(_sysv_poll_group_3[5]));
                uint32_t _sysv_min_41_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_4) : "r"(_sysv_poll_group_3[6]), "r"(_sysv_poll_group_3[7]));
                uint32_t _sysv_min_41_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_5) : "r"(_sysv_min_41_3), "r"(_sysv_min_41_4));
                uint32_t _sysv_min_41_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_6) : "r"(_sysv_min_41_2), "r"(_sysv_min_41_5));
                uint32_t _sysv_min_41_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_7) : "r"(_sysv_poll_group_3[8]), "r"(_sysv_poll_group_3[9]));
                uint32_t _sysv_min_41_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_8) : "r"(_sysv_poll_group_3[10]), "r"(_sysv_poll_group_3[11]));
                uint32_t _sysv_min_41_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_9) : "r"(_sysv_min_41_7), "r"(_sysv_min_41_8));
                uint32_t _sysv_min_41_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_10) : "r"(_sysv_poll_group_3[12]), "r"(_sysv_poll_group_3[13]));
                uint32_t _sysv_min_41_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_11) : "r"(_sysv_poll_group_3[14]), "r"(_sysv_poll_group_3[15]));
                uint32_t _sysv_min_41_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_12) : "r"(_sysv_min_41_10), "r"(_sysv_min_41_11));
                uint32_t _sysv_min_41_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_13) : "r"(_sysv_min_41_9), "r"(_sysv_min_41_12));
                uint32_t _sysv_min_41_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_14) : "r"(_sysv_min_41_6), "r"(_sysv_min_41_13));
                uint32_t _sysv_min_41_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_15) : "r"(_sysv_poll_group_3[16]), "r"(_sysv_poll_group_3[17]));
                uint32_t _sysv_min_41_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_16) : "r"(_sysv_poll_group_3[18]), "r"(_sysv_poll_group_3[19]));
                uint32_t _sysv_min_41_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_17) : "r"(_sysv_min_41_15), "r"(_sysv_min_41_16));
                uint32_t _sysv_min_41_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_18) : "r"(_sysv_poll_group_3[20]), "r"(_sysv_poll_group_3[21]));
                uint32_t _sysv_min_41_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_19) : "r"(_sysv_poll_group_3[22]), "r"(_sysv_poll_group_3[23]));
                uint32_t _sysv_min_41_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_20) : "r"(_sysv_min_41_18), "r"(_sysv_min_41_19));
                uint32_t _sysv_min_41_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_21) : "r"(_sysv_min_41_17), "r"(_sysv_min_41_20));
                uint32_t _sysv_min_41_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_22) : "r"(_sysv_poll_group_3[24]), "r"(_sysv_poll_group_3[25]));
                uint32_t _sysv_min_41_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_23) : "r"(_sysv_poll_group_3[26]), "r"(_sysv_poll_group_3[27]));
                uint32_t _sysv_min_41_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_24) : "r"(_sysv_min_41_22), "r"(_sysv_min_41_23));
                uint32_t _sysv_min_41_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_25) : "r"(_sysv_poll_group_3[28]), "r"(_sysv_poll_group_3[29]));
                uint32_t _sysv_min_41_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_26) : "r"(_sysv_poll_group_3[30]), "r"(_sysv_poll_group_3[31]));
                uint32_t _sysv_min_41_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_27) : "r"(_sysv_min_41_25), "r"(_sysv_min_41_26));
                uint32_t _sysv_min_41_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_28) : "r"(_sysv_min_41_24), "r"(_sysv_min_41_27));
                uint32_t _sysv_min_41_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_29) : "r"(_sysv_min_41_21), "r"(_sysv_min_41_28));
                uint32_t _sysv_min_41_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_41_30) : "r"(_sysv_min_41_14), "r"(_sysv_min_41_29));
                _sysv_min_40 = _sysv_min_41_30;
            } while (
                ((_sysv_min_40 & 0xffffu) == 0x8000u) || ((_sysv_min_40 >> 16) == 0x8000u)
            );
            float s8_6[8];
            float part_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_6[_pair * 2])[0]), "=f"((&part_6[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[_pair]));
            }
            s8_6[0] = part_6[0];
            s8_6[1] = part_6[1];
            s8_6[2] = part_6[2];
            s8_6[3] = part_6[3];
            s8_6[4] = part_6[4];
            s8_6[5] = part_6[5];
            s8_6[6] = part_6[6];
            s8_6[7] = part_6[7];
            float sum2_72[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_72[0]), "=&f"(sum2_72[1]) : "r"(_sysv_poll_group_3[8]), "f"(s8_6[0]), "f"(s8_6[1]));
            s8_6[0] = sum2_72[0];
            s8_6[1] = sum2_72[1];
            float sum2_73[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_73[0]), "=&f"(sum2_73[1]) : "r"(_sysv_poll_group_3[9]), "f"(s8_6[2]), "f"(s8_6[3]));
            s8_6[2] = sum2_73[0];
            s8_6[3] = sum2_73[1];
            float sum2_74[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_74[0]), "=&f"(sum2_74[1]) : "r"(_sysv_poll_group_3[10]), "f"(s8_6[4]), "f"(s8_6[5]));
            s8_6[4] = sum2_74[0];
            s8_6[5] = sum2_74[1];
            float sum2_75[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_75[0]), "=&f"(sum2_75[1]) : "r"(_sysv_poll_group_3[11]), "f"(s8_6[6]), "f"(s8_6[7]));
            s8_6[6] = sum2_75[0];
            s8_6[7] = sum2_75[1];
            float sum2_76[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_76[0]), "=&f"(sum2_76[1]) : "r"(_sysv_poll_group_3[16]), "f"(s8_6[0]), "f"(s8_6[1]));
            s8_6[0] = sum2_76[0];
            s8_6[1] = sum2_76[1];
            float sum2_77[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_77[0]), "=&f"(sum2_77[1]) : "r"(_sysv_poll_group_3[17]), "f"(s8_6[2]), "f"(s8_6[3]));
            s8_6[2] = sum2_77[0];
            s8_6[3] = sum2_77[1];
            float sum2_78[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_78[0]), "=&f"(sum2_78[1]) : "r"(_sysv_poll_group_3[18]), "f"(s8_6[4]), "f"(s8_6[5]));
            s8_6[4] = sum2_78[0];
            s8_6[5] = sum2_78[1];
            float sum2_79[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_79[0]), "=&f"(sum2_79[1]) : "r"(_sysv_poll_group_3[19]), "f"(s8_6[6]), "f"(s8_6[7]));
            s8_6[6] = sum2_79[0];
            s8_6[7] = sum2_79[1];
            float sum2_80[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_80[0]), "=&f"(sum2_80[1]) : "r"(_sysv_poll_group_3[24]), "f"(s8_6[0]), "f"(s8_6[1]));
            s8_6[0] = sum2_80[0];
            s8_6[1] = sum2_80[1];
            float sum2_81[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_81[0]), "=&f"(sum2_81[1]) : "r"(_sysv_poll_group_3[25]), "f"(s8_6[2]), "f"(s8_6[3]));
            s8_6[2] = sum2_81[0];
            s8_6[3] = sum2_81[1];
            float sum2_82[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_82[0]), "=&f"(sum2_82[1]) : "r"(_sysv_poll_group_3[26]), "f"(s8_6[4]), "f"(s8_6[5]));
            s8_6[4] = sum2_82[0];
            s8_6[5] = sum2_82[1];
            float sum2_83[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_83[0]), "=&f"(sum2_83[1]) : "r"(_sysv_poll_group_3[27]), "f"(s8_6[6]), "f"(s8_6[7]));
            s8_6[6] = sum2_83[0];
            s8_6[7] = sum2_83[1];
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
            float s8_7[8];
            float part_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&part_7[_pair * 2])[0]), "=f"((&part_7[_pair * 2])[1])
                    : "r"(_sysv_poll_group_3[4 + _pair]));
            }
            s8_7[0] = part_7[0];
            s8_7[1] = part_7[1];
            s8_7[2] = part_7[2];
            s8_7[3] = part_7[3];
            s8_7[4] = part_7[4];
            s8_7[5] = part_7[5];
            s8_7[6] = part_7[6];
            s8_7[7] = part_7[7];
            float sum2_84[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_84[0]), "=&f"(sum2_84[1]) : "r"(_sysv_poll_group_3[12]), "f"(s8_7[0]), "f"(s8_7[1]));
            s8_7[0] = sum2_84[0];
            s8_7[1] = sum2_84[1];
            float sum2_85[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_85[0]), "=&f"(sum2_85[1]) : "r"(_sysv_poll_group_3[13]), "f"(s8_7[2]), "f"(s8_7[3]));
            s8_7[2] = sum2_85[0];
            s8_7[3] = sum2_85[1];
            float sum2_86[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_86[0]), "=&f"(sum2_86[1]) : "r"(_sysv_poll_group_3[14]), "f"(s8_7[4]), "f"(s8_7[5]));
            s8_7[4] = sum2_86[0];
            s8_7[5] = sum2_86[1];
            float sum2_87[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_87[0]), "=&f"(sum2_87[1]) : "r"(_sysv_poll_group_3[15]), "f"(s8_7[6]), "f"(s8_7[7]));
            s8_7[6] = sum2_87[0];
            s8_7[7] = sum2_87[1];
            float sum2_88[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_88[0]), "=&f"(sum2_88[1]) : "r"(_sysv_poll_group_3[20]), "f"(s8_7[0]), "f"(s8_7[1]));
            s8_7[0] = sum2_88[0];
            s8_7[1] = sum2_88[1];
            float sum2_89[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_89[0]), "=&f"(sum2_89[1]) : "r"(_sysv_poll_group_3[21]), "f"(s8_7[2]), "f"(s8_7[3]));
            s8_7[2] = sum2_89[0];
            s8_7[3] = sum2_89[1];
            float sum2_90[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_90[0]), "=&f"(sum2_90[1]) : "r"(_sysv_poll_group_3[22]), "f"(s8_7[4]), "f"(s8_7[5]));
            s8_7[4] = sum2_90[0];
            s8_7[5] = sum2_90[1];
            float sum2_91[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_91[0]), "=&f"(sum2_91[1]) : "r"(_sysv_poll_group_3[23]), "f"(s8_7[6]), "f"(s8_7[7]));
            s8_7[6] = sum2_91[0];
            s8_7[7] = sum2_91[1];
            float sum2_92[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_92[0]), "=&f"(sum2_92[1]) : "r"(_sysv_poll_group_3[28]), "f"(s8_7[0]), "f"(s8_7[1]));
            s8_7[0] = sum2_92[0];
            s8_7[1] = sum2_92[1];
            float sum2_93[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_93[0]), "=&f"(sum2_93[1]) : "r"(_sysv_poll_group_3[29]), "f"(s8_7[2]), "f"(s8_7[3]));
            s8_7[2] = sum2_93[0];
            s8_7[3] = sum2_93[1];
            float sum2_94[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_94[0]), "=&f"(sum2_94[1]) : "r"(_sysv_poll_group_3[30]), "f"(s8_7[4]), "f"(s8_7[5]));
            s8_7[4] = sum2_94[0];
            s8_7[5] = sum2_94[1];
            float sum2_95[2];
            asm(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(sum2_95[0]), "=&f"(sum2_95[1]) : "r"(_sysv_poll_group_3[31]), "f"(s8_7[6]), "f"(s8_7[7]));
            s8_7[6] = sum2_95[0];
            s8_7[7] = sum2_95[1];
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
                const unsigned* _raw_stv8_42 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 98304))), "r"(_raw_stv8_42[0]), "r"(_raw_stv8_42[1]), "r"(_raw_stv8_42[2]), "r"(_raw_stv8_42[3]), "r"(_raw_stv8_42[4]), "r"(_raw_stv8_42[5]), "r"(_raw_stv8_42[6]), "r"(_raw_stv8_42[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_43 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 106496))), "r"(_raw_stv8_43[0]), "r"(_raw_stv8_43[1]), "r"(_raw_stv8_43[2]), "r"(_raw_stv8_43[3]), "r"(_raw_stv8_43[4]), "r"(_raw_stv8_43[5]), "r"(_raw_stv8_43[6]), "r"(_raw_stv8_43[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_44 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 114688))), "r"(_raw_stv8_44[0]), "r"(_raw_stv8_44[1]), "r"(_raw_stv8_44[2]), "r"(_raw_stv8_44[3]), "r"(_raw_stv8_44[4]), "r"(_raw_stv8_44[5]), "r"(_raw_stv8_44[6]), "r"(_raw_stv8_44[7]) : "memory");
            }
            {
                const unsigned* _raw_stv8_45 = reinterpret_cast<const unsigned*>(sent8);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(push_local + (src + col64 + 122880))), "r"(_raw_stv8_45[0]), "r"(_raw_stv8_45[1]), "r"(_raw_stv8_45[2]), "r"(_raw_stv8_45[3]), "r"(_raw_stv8_45[4]), "r"(_raw_stv8_45[5]), "r"(_raw_stv8_45[6]), "r"(_raw_stv8_45[7]) : "memory");
            }
            float _wrg_6 = ssq8[6];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _wrg_6 += __shfl_xor_sync(0xFFFFFFFF, _wrg_6, offset);
            if (lane == 0) {
                warp_tab[parity + warp * 8 + 6] = _wrg_6;
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
            float _sqrt_46;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_46) : "f"(__fmaf_rn(tl, 0.00024414062f, eps)));
            float rsl = __frcp_rn(_sqrt_46);
            float rs8[8];
            float _shfl_sync_47 = __shfl_sync(0xFFFFFFFF, rsl, 0);
            rs8[0] = _shfl_sync_47;
            float _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, rsl, 1);
            rs8[1] = _shfl_sync_48;
            float _shfl_sync_49 = __shfl_sync(0xFFFFFFFF, rsl, 2);
            rs8[2] = _shfl_sync_49;
            float _shfl_sync_50 = __shfl_sync(0xFFFFFFFF, rsl, 3);
            rs8[3] = _shfl_sync_50;
            float _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, rsl, 4);
            rs8[4] = _shfl_sync_51;
            float _shfl_sync_52 = __shfl_sync(0xFFFFFFFF, rsl, 5);
            rs8[5] = _shfl_sync_52;
            float _shfl_sync_53 = __shfl_sync(0xFFFFFFFF, rsl, 6);
            rs8[6] = _shfl_sync_53;
            float _shfl_sync_54 = __shfl_sync(0xFFFFFFFF, rsl, 7);
            rs8[7] = _shfl_sync_54;
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
            float2 rs2_0 = make_float2(rs_0, rs_0);
            float2 we2_0 = make_float2(w8[0], w8[0]);
            float yna_0 = ((mode != 0) ? yn8_0[0] * gw8[0] : yn8_0[0]);
            float ynb_0 = ((mode != 0) ? yn8_0[1] * gw8[1] : yn8_0[1]);
            float2 _f32x2_mul_a_55 = make_float2(yna_0, ynb_0);
            float2 ov_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&rs2_0));
            float oa_0 = ((mode != 0) ? ov_0.x : ov_0.x * gw8[0]);
            float ob_0 = ((mode != 0) ? ov_0.y : ov_0.y * gw8[1]);
            float2 o2v_0 = make_float2(oa_0, ob_0);
            float2 acc2_0 = fma_f32x2_rn_ftz(o2v_0, we2_0, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_0.x;
            acc8[1] = acc2_0.y;
            float yna_1 = ((mode != 0) ? yn8_0[2] * gw8[2] : yn8_0[2]);
            float ynb_1 = ((mode != 0) ? yn8_0[3] * gw8[3] : yn8_0[3]);
            float2 _f32x2_mul_a_56 = make_float2(yna_1, ynb_1);
            float2 ov_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&rs2_0));
            float oa_1 = ((mode != 0) ? ov_1.x : ov_1.x * gw8[2]);
            float ob_1 = ((mode != 0) ? ov_1.y : ov_1.y * gw8[3]);
            float2 o2v_1 = make_float2(oa_1, ob_1);
            float2 acc2_1 = fma_f32x2_rn_ftz(o2v_1, we2_0, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_1.x;
            acc8[3] = acc2_1.y;
            float yna_2 = ((mode != 0) ? yn8_0[4] * gw8[4] : yn8_0[4]);
            float ynb_2 = ((mode != 0) ? yn8_0[5] * gw8[5] : yn8_0[5]);
            float2 _f32x2_mul_a_57 = make_float2(yna_2, ynb_2);
            float2 ov_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&rs2_0));
            float oa_2 = ((mode != 0) ? ov_2.x : ov_2.x * gw8[4]);
            float ob_2 = ((mode != 0) ? ov_2.y : ov_2.y * gw8[5]);
            float2 o2v_2 = make_float2(oa_2, ob_2);
            float2 acc2_2 = fma_f32x2_rn_ftz(o2v_2, we2_0, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_2.x;
            acc8[5] = acc2_2.y;
            float yna_3 = ((mode != 0) ? yn8_0[6] * gw8[6] : yn8_0[6]);
            float ynb_3 = ((mode != 0) ? yn8_0[7] * gw8[7] : yn8_0[7]);
            float2 _f32x2_mul_a_58 = make_float2(yna_3, ynb_3);
            float2 ov_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&rs2_0));
            float oa_3 = ((mode != 0) ? ov_3.x : ov_3.x * gw8[6]);
            float ob_3 = ((mode != 0) ? ov_3.y : ov_3.y * gw8[7]);
            float2 o2v_3 = make_float2(oa_3, ob_3);
            float2 acc2_3 = fma_f32x2_rn_ftz(o2v_3, we2_0, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_3.x;
            acc8[7] = acc2_3.y;
            float rs_1 = rs8[1];
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
            float2 rs2_1 = make_float2(rs_1, rs_1);
            float2 we2_1 = make_float2(w8[1], w8[1]);
            float yna_4 = ((mode != 0) ? yn8_1[0] * gw8[0] : yn8_1[0]);
            float ynb_4 = ((mode != 0) ? yn8_1[1] * gw8[1] : yn8_1[1]);
            float2 _f32x2_mul_a_59 = make_float2(yna_4, ynb_4);
            float2 ov_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&rs2_1));
            float oa_4 = ((mode != 0) ? ov_4.x : ov_4.x * gw8[0]);
            float ob_4 = ((mode != 0) ? ov_4.y : ov_4.y * gw8[1]);
            float2 o2v_4 = make_float2(oa_4, ob_4);
            float2 acc2_4 = fma_f32x2_rn_ftz(o2v_4, we2_1, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_4.x;
            acc8[1] = acc2_4.y;
            float yna_5 = ((mode != 0) ? yn8_1[2] * gw8[2] : yn8_1[2]);
            float ynb_5 = ((mode != 0) ? yn8_1[3] * gw8[3] : yn8_1[3]);
            float2 _f32x2_mul_a_60 = make_float2(yna_5, ynb_5);
            float2 ov_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&rs2_1));
            float oa_5 = ((mode != 0) ? ov_5.x : ov_5.x * gw8[2]);
            float ob_5 = ((mode != 0) ? ov_5.y : ov_5.y * gw8[3]);
            float2 o2v_5 = make_float2(oa_5, ob_5);
            float2 acc2_5 = fma_f32x2_rn_ftz(o2v_5, we2_1, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_5.x;
            acc8[3] = acc2_5.y;
            float yna_6 = ((mode != 0) ? yn8_1[4] * gw8[4] : yn8_1[4]);
            float ynb_6 = ((mode != 0) ? yn8_1[5] * gw8[5] : yn8_1[5]);
            float2 _f32x2_mul_a_61 = make_float2(yna_6, ynb_6);
            float2 ov_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&rs2_1));
            float oa_6 = ((mode != 0) ? ov_6.x : ov_6.x * gw8[4]);
            float ob_6 = ((mode != 0) ? ov_6.y : ov_6.y * gw8[5]);
            float2 o2v_6 = make_float2(oa_6, ob_6);
            float2 acc2_6 = fma_f32x2_rn_ftz(o2v_6, we2_1, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_6.x;
            acc8[5] = acc2_6.y;
            float yna_7 = ((mode != 0) ? yn8_1[6] * gw8[6] : yn8_1[6]);
            float ynb_7 = ((mode != 0) ? yn8_1[7] * gw8[7] : yn8_1[7]);
            float2 _f32x2_mul_a_62 = make_float2(yna_7, ynb_7);
            float2 ov_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&rs2_1));
            float oa_7 = ((mode != 0) ? ov_7.x : ov_7.x * gw8[6]);
            float ob_7 = ((mode != 0) ? ov_7.y : ov_7.y * gw8[7]);
            float2 o2v_7 = make_float2(oa_7, ob_7);
            float2 acc2_7 = fma_f32x2_rn_ftz(o2v_7, we2_1, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_7.x;
            acc8[7] = acc2_7.y;
            float rs_2 = rs8[2];
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
            float2 rs2_2 = make_float2(rs_2, rs_2);
            float2 we2_2 = make_float2(w8[2], w8[2]);
            float yna_8 = ((mode != 0) ? yn8_2[0] * gw8[0] : yn8_2[0]);
            float ynb_8 = ((mode != 0) ? yn8_2[1] * gw8[1] : yn8_2[1]);
            float2 _f32x2_mul_a_63 = make_float2(yna_8, ynb_8);
            float2 ov_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&rs2_2));
            float oa_8 = ((mode != 0) ? ov_8.x : ov_8.x * gw8[0]);
            float ob_8 = ((mode != 0) ? ov_8.y : ov_8.y * gw8[1]);
            float2 o2v_8 = make_float2(oa_8, ob_8);
            float2 acc2_8 = fma_f32x2_rn_ftz(o2v_8, we2_2, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_8.x;
            acc8[1] = acc2_8.y;
            float yna_9 = ((mode != 0) ? yn8_2[2] * gw8[2] : yn8_2[2]);
            float ynb_9 = ((mode != 0) ? yn8_2[3] * gw8[3] : yn8_2[3]);
            float2 _f32x2_mul_a_64 = make_float2(yna_9, ynb_9);
            float2 ov_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&rs2_2));
            float oa_9 = ((mode != 0) ? ov_9.x : ov_9.x * gw8[2]);
            float ob_9 = ((mode != 0) ? ov_9.y : ov_9.y * gw8[3]);
            float2 o2v_9 = make_float2(oa_9, ob_9);
            float2 acc2_9 = fma_f32x2_rn_ftz(o2v_9, we2_2, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_9.x;
            acc8[3] = acc2_9.y;
            float yna_10 = ((mode != 0) ? yn8_2[4] * gw8[4] : yn8_2[4]);
            float ynb_10 = ((mode != 0) ? yn8_2[5] * gw8[5] : yn8_2[5]);
            float2 _f32x2_mul_a_65 = make_float2(yna_10, ynb_10);
            float2 ov_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&rs2_2));
            float oa_10 = ((mode != 0) ? ov_10.x : ov_10.x * gw8[4]);
            float ob_10 = ((mode != 0) ? ov_10.y : ov_10.y * gw8[5]);
            float2 o2v_10 = make_float2(oa_10, ob_10);
            float2 acc2_10 = fma_f32x2_rn_ftz(o2v_10, we2_2, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_10.x;
            acc8[5] = acc2_10.y;
            float yna_11 = ((mode != 0) ? yn8_2[6] * gw8[6] : yn8_2[6]);
            float ynb_11 = ((mode != 0) ? yn8_2[7] * gw8[7] : yn8_2[7]);
            float2 _f32x2_mul_a_66 = make_float2(yna_11, ynb_11);
            float2 ov_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&rs2_2));
            float oa_11 = ((mode != 0) ? ov_11.x : ov_11.x * gw8[6]);
            float ob_11 = ((mode != 0) ? ov_11.y : ov_11.y * gw8[7]);
            float2 o2v_11 = make_float2(oa_11, ob_11);
            float2 acc2_11 = fma_f32x2_rn_ftz(o2v_11, we2_2, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_11.x;
            acc8[7] = acc2_11.y;
            float rs_3 = rs8[3];
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
            float2 rs2_3 = make_float2(rs_3, rs_3);
            float2 we2_3 = make_float2(w8[3], w8[3]);
            float yna_12 = ((mode != 0) ? yn8_3[0] * gw8[0] : yn8_3[0]);
            float ynb_12 = ((mode != 0) ? yn8_3[1] * gw8[1] : yn8_3[1]);
            float2 _f32x2_mul_a_67 = make_float2(yna_12, ynb_12);
            float2 ov_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&rs2_3));
            float oa_12 = ((mode != 0) ? ov_12.x : ov_12.x * gw8[0]);
            float ob_12 = ((mode != 0) ? ov_12.y : ov_12.y * gw8[1]);
            float2 o2v_12 = make_float2(oa_12, ob_12);
            float2 acc2_12 = fma_f32x2_rn_ftz(o2v_12, we2_3, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_12.x;
            acc8[1] = acc2_12.y;
            float yna_13 = ((mode != 0) ? yn8_3[2] * gw8[2] : yn8_3[2]);
            float ynb_13 = ((mode != 0) ? yn8_3[3] * gw8[3] : yn8_3[3]);
            float2 _f32x2_mul_a_68 = make_float2(yna_13, ynb_13);
            float2 ov_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&rs2_3));
            float oa_13 = ((mode != 0) ? ov_13.x : ov_13.x * gw8[2]);
            float ob_13 = ((mode != 0) ? ov_13.y : ov_13.y * gw8[3]);
            float2 o2v_13 = make_float2(oa_13, ob_13);
            float2 acc2_13 = fma_f32x2_rn_ftz(o2v_13, we2_3, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_13.x;
            acc8[3] = acc2_13.y;
            float yna_14 = ((mode != 0) ? yn8_3[4] * gw8[4] : yn8_3[4]);
            float ynb_14 = ((mode != 0) ? yn8_3[5] * gw8[5] : yn8_3[5]);
            float2 _f32x2_mul_a_69 = make_float2(yna_14, ynb_14);
            float2 ov_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&rs2_3));
            float oa_14 = ((mode != 0) ? ov_14.x : ov_14.x * gw8[4]);
            float ob_14 = ((mode != 0) ? ov_14.y : ov_14.y * gw8[5]);
            float2 o2v_14 = make_float2(oa_14, ob_14);
            float2 acc2_14 = fma_f32x2_rn_ftz(o2v_14, we2_3, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_14.x;
            acc8[5] = acc2_14.y;
            float yna_15 = ((mode != 0) ? yn8_3[6] * gw8[6] : yn8_3[6]);
            float ynb_15 = ((mode != 0) ? yn8_3[7] * gw8[7] : yn8_3[7]);
            float2 _f32x2_mul_a_70 = make_float2(yna_15, ynb_15);
            float2 ov_15;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&rs2_3));
            float oa_15 = ((mode != 0) ? ov_15.x : ov_15.x * gw8[6]);
            float ob_15 = ((mode != 0) ? ov_15.y : ov_15.y * gw8[7]);
            float2 o2v_15 = make_float2(oa_15, ob_15);
            float2 acc2_15 = fma_f32x2_rn_ftz(o2v_15, we2_3, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_15.x;
            acc8[7] = acc2_15.y;
            float rs_4 = rs8[4];
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
            float2 rs2_4 = make_float2(rs_4, rs_4);
            float2 we2_4 = make_float2(w8[4], w8[4]);
            float yna_16 = ((mode != 0) ? yn8_4[0] * gw8[0] : yn8_4[0]);
            float ynb_16 = ((mode != 0) ? yn8_4[1] * gw8[1] : yn8_4[1]);
            float2 _f32x2_mul_a_71 = make_float2(yna_16, ynb_16);
            float2 ov_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&rs2_4));
            float oa_16 = ((mode != 0) ? ov_16.x : ov_16.x * gw8[0]);
            float ob_16 = ((mode != 0) ? ov_16.y : ov_16.y * gw8[1]);
            float2 o2v_16 = make_float2(oa_16, ob_16);
            float2 acc2_16 = fma_f32x2_rn_ftz(o2v_16, we2_4, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_16.x;
            acc8[1] = acc2_16.y;
            float yna_17 = ((mode != 0) ? yn8_4[2] * gw8[2] : yn8_4[2]);
            float ynb_17 = ((mode != 0) ? yn8_4[3] * gw8[3] : yn8_4[3]);
            float2 _f32x2_mul_a_72 = make_float2(yna_17, ynb_17);
            float2 ov_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&rs2_4));
            float oa_17 = ((mode != 0) ? ov_17.x : ov_17.x * gw8[2]);
            float ob_17 = ((mode != 0) ? ov_17.y : ov_17.y * gw8[3]);
            float2 o2v_17 = make_float2(oa_17, ob_17);
            float2 acc2_17 = fma_f32x2_rn_ftz(o2v_17, we2_4, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_17.x;
            acc8[3] = acc2_17.y;
            float yna_18 = ((mode != 0) ? yn8_4[4] * gw8[4] : yn8_4[4]);
            float ynb_18 = ((mode != 0) ? yn8_4[5] * gw8[5] : yn8_4[5]);
            float2 _f32x2_mul_a_73 = make_float2(yna_18, ynb_18);
            float2 ov_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&rs2_4));
            float oa_18 = ((mode != 0) ? ov_18.x : ov_18.x * gw8[4]);
            float ob_18 = ((mode != 0) ? ov_18.y : ov_18.y * gw8[5]);
            float2 o2v_18 = make_float2(oa_18, ob_18);
            float2 acc2_18 = fma_f32x2_rn_ftz(o2v_18, we2_4, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_18.x;
            acc8[5] = acc2_18.y;
            float yna_19 = ((mode != 0) ? yn8_4[6] * gw8[6] : yn8_4[6]);
            float ynb_19 = ((mode != 0) ? yn8_4[7] * gw8[7] : yn8_4[7]);
            float2 _f32x2_mul_a_74 = make_float2(yna_19, ynb_19);
            float2 ov_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&rs2_4));
            float oa_19 = ((mode != 0) ? ov_19.x : ov_19.x * gw8[6]);
            float ob_19 = ((mode != 0) ? ov_19.y : ov_19.y * gw8[7]);
            float2 o2v_19 = make_float2(oa_19, ob_19);
            float2 acc2_19 = fma_f32x2_rn_ftz(o2v_19, we2_4, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_19.x;
            acc8[7] = acc2_19.y;
            float rs_5 = rs8[5];
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
            float2 rs2_5 = make_float2(rs_5, rs_5);
            float2 we2_5 = make_float2(w8[5], w8[5]);
            float yna_20 = ((mode != 0) ? yn8_5[0] * gw8[0] : yn8_5[0]);
            float ynb_20 = ((mode != 0) ? yn8_5[1] * gw8[1] : yn8_5[1]);
            float2 _f32x2_mul_a_75 = make_float2(yna_20, ynb_20);
            float2 ov_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&rs2_5));
            float oa_20 = ((mode != 0) ? ov_20.x : ov_20.x * gw8[0]);
            float ob_20 = ((mode != 0) ? ov_20.y : ov_20.y * gw8[1]);
            float2 o2v_20 = make_float2(oa_20, ob_20);
            float2 acc2_20 = fma_f32x2_rn_ftz(o2v_20, we2_5, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_20.x;
            acc8[1] = acc2_20.y;
            float yna_21 = ((mode != 0) ? yn8_5[2] * gw8[2] : yn8_5[2]);
            float ynb_21 = ((mode != 0) ? yn8_5[3] * gw8[3] : yn8_5[3]);
            float2 _f32x2_mul_a_76 = make_float2(yna_21, ynb_21);
            float2 ov_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&rs2_5));
            float oa_21 = ((mode != 0) ? ov_21.x : ov_21.x * gw8[2]);
            float ob_21 = ((mode != 0) ? ov_21.y : ov_21.y * gw8[3]);
            float2 o2v_21 = make_float2(oa_21, ob_21);
            float2 acc2_21 = fma_f32x2_rn_ftz(o2v_21, we2_5, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_21.x;
            acc8[3] = acc2_21.y;
            float yna_22 = ((mode != 0) ? yn8_5[4] * gw8[4] : yn8_5[4]);
            float ynb_22 = ((mode != 0) ? yn8_5[5] * gw8[5] : yn8_5[5]);
            float2 _f32x2_mul_a_77 = make_float2(yna_22, ynb_22);
            float2 ov_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&rs2_5));
            float oa_22 = ((mode != 0) ? ov_22.x : ov_22.x * gw8[4]);
            float ob_22 = ((mode != 0) ? ov_22.y : ov_22.y * gw8[5]);
            float2 o2v_22 = make_float2(oa_22, ob_22);
            float2 acc2_22 = fma_f32x2_rn_ftz(o2v_22, we2_5, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_22.x;
            acc8[5] = acc2_22.y;
            float yna_23 = ((mode != 0) ? yn8_5[6] * gw8[6] : yn8_5[6]);
            float ynb_23 = ((mode != 0) ? yn8_5[7] * gw8[7] : yn8_5[7]);
            float2 _f32x2_mul_a_78 = make_float2(yna_23, ynb_23);
            float2 ov_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&rs2_5));
            float oa_23 = ((mode != 0) ? ov_23.x : ov_23.x * gw8[6]);
            float ob_23 = ((mode != 0) ? ov_23.y : ov_23.y * gw8[7]);
            float2 o2v_23 = make_float2(oa_23, ob_23);
            float2 acc2_23 = fma_f32x2_rn_ftz(o2v_23, we2_5, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_23.x;
            acc8[7] = acc2_23.y;
            float rs_6 = rs8[6];
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
            float2 rs2_6 = make_float2(rs_6, rs_6);
            float2 we2_6 = make_float2(w8[6], w8[6]);
            float yna_24 = ((mode != 0) ? yn8_6[0] * gw8[0] : yn8_6[0]);
            float ynb_24 = ((mode != 0) ? yn8_6[1] * gw8[1] : yn8_6[1]);
            float2 _f32x2_mul_a_79 = make_float2(yna_24, ynb_24);
            float2 ov_24;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&rs2_6));
            float oa_24 = ((mode != 0) ? ov_24.x : ov_24.x * gw8[0]);
            float ob_24 = ((mode != 0) ? ov_24.y : ov_24.y * gw8[1]);
            float2 o2v_24 = make_float2(oa_24, ob_24);
            float2 acc2_24 = fma_f32x2_rn_ftz(o2v_24, we2_6, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_24.x;
            acc8[1] = acc2_24.y;
            float yna_25 = ((mode != 0) ? yn8_6[2] * gw8[2] : yn8_6[2]);
            float ynb_25 = ((mode != 0) ? yn8_6[3] * gw8[3] : yn8_6[3]);
            float2 _f32x2_mul_a_80 = make_float2(yna_25, ynb_25);
            float2 ov_25;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&rs2_6));
            float oa_25 = ((mode != 0) ? ov_25.x : ov_25.x * gw8[2]);
            float ob_25 = ((mode != 0) ? ov_25.y : ov_25.y * gw8[3]);
            float2 o2v_25 = make_float2(oa_25, ob_25);
            float2 acc2_25 = fma_f32x2_rn_ftz(o2v_25, we2_6, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_25.x;
            acc8[3] = acc2_25.y;
            float yna_26 = ((mode != 0) ? yn8_6[4] * gw8[4] : yn8_6[4]);
            float ynb_26 = ((mode != 0) ? yn8_6[5] * gw8[5] : yn8_6[5]);
            float2 _f32x2_mul_a_81 = make_float2(yna_26, ynb_26);
            float2 ov_26;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&rs2_6));
            float oa_26 = ((mode != 0) ? ov_26.x : ov_26.x * gw8[4]);
            float ob_26 = ((mode != 0) ? ov_26.y : ov_26.y * gw8[5]);
            float2 o2v_26 = make_float2(oa_26, ob_26);
            float2 acc2_26 = fma_f32x2_rn_ftz(o2v_26, we2_6, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_26.x;
            acc8[5] = acc2_26.y;
            float yna_27 = ((mode != 0) ? yn8_6[6] * gw8[6] : yn8_6[6]);
            float ynb_27 = ((mode != 0) ? yn8_6[7] * gw8[7] : yn8_6[7]);
            float2 _f32x2_mul_a_82 = make_float2(yna_27, ynb_27);
            float2 ov_27;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&rs2_6));
            float oa_27 = ((mode != 0) ? ov_27.x : ov_27.x * gw8[6]);
            float ob_27 = ((mode != 0) ? ov_27.y : ov_27.y * gw8[7]);
            float2 o2v_27 = make_float2(oa_27, ob_27);
            float2 acc2_27 = fma_f32x2_rn_ftz(o2v_27, we2_6, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_27.x;
            acc8[7] = acc2_27.y;
            float rs_7 = rs8[7];
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
            float2 rs2_7 = make_float2(rs_7, rs_7);
            float2 we2_7 = make_float2(w8[7], w8[7]);
            float yna_28 = ((mode != 0) ? yn8_7[0] * gw8[0] : yn8_7[0]);
            float ynb_28 = ((mode != 0) ? yn8_7[1] * gw8[1] : yn8_7[1]);
            float2 _f32x2_mul_a_83 = make_float2(yna_28, ynb_28);
            float2 ov_28;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&rs2_7));
            float oa_28 = ((mode != 0) ? ov_28.x : ov_28.x * gw8[0]);
            float ob_28 = ((mode != 0) ? ov_28.y : ov_28.y * gw8[1]);
            float2 o2v_28 = make_float2(oa_28, ob_28);
            float2 acc2_28 = fma_f32x2_rn_ftz(o2v_28, we2_7, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_28.x;
            acc8[1] = acc2_28.y;
            float yna_29 = ((mode != 0) ? yn8_7[2] * gw8[2] : yn8_7[2]);
            float ynb_29 = ((mode != 0) ? yn8_7[3] * gw8[3] : yn8_7[3]);
            float2 _f32x2_mul_a_84 = make_float2(yna_29, ynb_29);
            float2 ov_29;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&rs2_7));
            float oa_29 = ((mode != 0) ? ov_29.x : ov_29.x * gw8[2]);
            float ob_29 = ((mode != 0) ? ov_29.y : ov_29.y * gw8[3]);
            float2 o2v_29 = make_float2(oa_29, ob_29);
            float2 acc2_29 = fma_f32x2_rn_ftz(o2v_29, we2_7, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_29.x;
            acc8[3] = acc2_29.y;
            float yna_30 = ((mode != 0) ? yn8_7[4] * gw8[4] : yn8_7[4]);
            float ynb_30 = ((mode != 0) ? yn8_7[5] * gw8[5] : yn8_7[5]);
            float2 _f32x2_mul_a_85 = make_float2(yna_30, ynb_30);
            float2 ov_30;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&rs2_7));
            float oa_30 = ((mode != 0) ? ov_30.x : ov_30.x * gw8[4]);
            float ob_30 = ((mode != 0) ? ov_30.y : ov_30.y * gw8[5]);
            float2 o2v_30 = make_float2(oa_30, ob_30);
            float2 acc2_30 = fma_f32x2_rn_ftz(o2v_30, we2_7, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_30.x;
            acc8[5] = acc2_30.y;
            float yna_31 = ((mode != 0) ? yn8_7[6] * gw8[6] : yn8_7[6]);
            float ynb_31 = ((mode != 0) ? yn8_7[7] * gw8[7] : yn8_7[7]);
            float2 _f32x2_mul_a_86 = make_float2(yna_31, ynb_31);
            float2 ov_31;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&rs2_7));
            float oa_31 = ((mode != 0) ? ov_31.x : ov_31.x * gw8[6]);
            float ob_31 = ((mode != 0) ? ov_31.y : ov_31.y * gw8[7]);
            float2 o2v_31 = make_float2(oa_31, ob_31);
            float2 acc2_31 = fma_f32x2_rn_ftz(o2v_31, we2_7, make_float2(acc8[6], acc8[7]));
            acc8[6] = acc2_31.x;
            acc8[7] = acc2_31.y;
            unsigned int acc8_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 0], acc8[_lp*2+1 + 0]));
                acc8_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _packed16_sanitize_8[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_87 = static_cast<uint32_t>(acc8_bf16[_p16s_word]);
                uint32_t _p16s_lo_87 = _p16s_carrier_87 & 0x0000ffffu;
                uint32_t _p16s_hi_87 = (_p16s_carrier_87 >> 16) & 0x0000ffffu;
                if (_p16s_lo_87 == 0x00008000u) _p16s_lo_87 = 0u;
                if (_p16s_hi_87 == 0x00008000u) _p16s_hi_87 = 0u;
                _packed16_sanitize_8[_p16s_word] = _p16s_lo_87 | (_p16s_hi_87 << 16);
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
                auto _gst_88 = bcast_local + land;
                __builtin_assume(__isGlobal(_gst_88));
                reinterpret_cast<int4*>(_gst_88)[0] = reinterpret_cast<int4*>(sent4)[0];
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
