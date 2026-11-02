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
#define SMEM_STG_OFF 2048
#define SMEM_STG_STAGE_BYTES 65536
#define SMEM_STG_STRIDE 65536
#define SMEM_TOTAL 67584
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


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
kernel_cake_tp_moe_finalize_rmsnorm_75007e91ada8af5c1687(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int max_ctas, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, unsigned long long c_mcast, int* __restrict__ ctl)
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
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 2048);
    const int stg_addr = smem + 2048;

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
    long long push_slot = push_cta - col64;
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
    int idxn[8];
    if (n_mine > 0) {
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(idx + (b * 8) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            idxn[0 + 0] = _ivld_1.x;
            idxn[0 + 1] = _ivld_1.y;
            idxn[0 + 2] = _ivld_1.z;
            idxn[0 + 3] = _ivld_1.w;
        }
        {
            const int4* _ivptr_2 = reinterpret_cast<const int4*>(idx + (b * 8 + 4) + 0);
            int4 _ivld_2;
            _ivld_2 = *_ivptr_2;
            idxn[4 + 0] = _ivld_2.x;
            idxn[4 + 1] = _ivld_2.y;
            idxn[4 + 2] = _ivld_2.z;
            idxn[4 + 3] = _ivld_2.w;
        }
    }
    #pragma unroll 1
    for (int step = 0; step < steps; step++) {
        int s = s0 + step;
        if (n_mine > step) {
            int tok = b + step * grid;
            int owner = b + step & 3;
            int gb = s >> 2 & 3;
            long long dst = push_gen_elems * (long long)gb + push_cta + rank64 * 4096;
            int idx8[8];
            idx8[0] = idxn[0];
            idx8[1] = idxn[1];
            idx8[2] = idxn[2];
            idx8[3] = idxn[3];
            idx8[4] = idxn[4];
            idx8[5] = idxn[5];
            idx8[6] = idxn[6];
            idx8[7] = idxn[7];
            if (n_mine > step + 1) {
                {
                    const int4* _ivptr_3 = reinterpret_cast<const int4*>(idx + ((tok + grid) * 8) + 0);
                    int4 _ivld_3;
                    _ivld_3 = *_ivptr_3;
                    idxn[0 + 0] = _ivld_3.x;
                    idxn[0 + 1] = _ivld_3.y;
                    idxn[0 + 2] = _ivld_3.z;
                    idxn[0 + 3] = _ivld_3.w;
                }
                {
                    const int4* _ivptr_4 = reinterpret_cast<const int4*>(idx + ((tok + grid) * 8 + 4) + 0);
                    int4 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    idxn[4 + 0] = _ivld_4.x;
                    idxn[4 + 1] = _ivld_4.y;
                    idxn[4 + 2] = _ivld_4.z;
                    idxn[4 + 3] = _ivld_4.w;
                }
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
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_0 * (long long)row_stride + col64) + 0);
                    packets[0 + 0] = _uv4_5.x;
                    packets[0 + 1] = _uv4_5.y;
                    packets[0 + 2] = _uv4_5.z;
                    packets[0 + 3] = _uv4_5.w;
                }
            }
            int ridx_1 = idx8[1];
            if (ridx_1 >= 0 && ridx_1 < rows) {
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_1 * (long long)row_stride + col64) + 0);
                    packets[4 + 0] = _uv4_6.x;
                    packets[4 + 1] = _uv4_6.y;
                    packets[4 + 2] = _uv4_6.z;
                    packets[4 + 3] = _uv4_6.w;
                }
            }
            int ridx_2 = idx8[2];
            if (ridx_2 >= 0 && ridx_2 < rows) {
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_2 * (long long)row_stride + col64) + 0);
                    packets[8 + 0] = _uv4_7.x;
                    packets[8 + 1] = _uv4_7.y;
                    packets[8 + 2] = _uv4_7.z;
                    packets[8 + 3] = _uv4_7.w;
                }
            }
            int ridx_3 = idx8[3];
            if (ridx_3 >= 0 && ridx_3 < rows) {
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_3 * (long long)row_stride + col64) + 0);
                    packets[12 + 0] = _uv4_8.x;
                    packets[12 + 1] = _uv4_8.y;
                    packets[12 + 2] = _uv4_8.z;
                    packets[12 + 3] = _uv4_8.w;
                }
            }
            int ridx_4 = idx8[4];
            if (ridx_4 >= 0 && ridx_4 < rows) {
                {
                    uint4 _uv4_9 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_4 * (long long)row_stride + col64) + 0);
                    packets[16 + 0] = _uv4_9.x;
                    packets[16 + 1] = _uv4_9.y;
                    packets[16 + 2] = _uv4_9.z;
                    packets[16 + 3] = _uv4_9.w;
                }
            }
            int ridx_5 = idx8[5];
            if (ridx_5 >= 0 && ridx_5 < rows) {
                {
                    uint4 _uv4_10 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_5 * (long long)row_stride + col64) + 0);
                    packets[20 + 0] = _uv4_10.x;
                    packets[20 + 1] = _uv4_10.y;
                    packets[20 + 2] = _uv4_10.z;
                    packets[20 + 3] = _uv4_10.w;
                }
            }
            int ridx_6 = idx8[6];
            if (ridx_6 >= 0 && ridx_6 < rows) {
                {
                    uint4 _uv4_11 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_6 * (long long)row_stride + col64) + 0);
                    packets[24 + 0] = _uv4_11.x;
                    packets[24 + 1] = _uv4_11.y;
                    packets[24 + 2] = _uv4_11.z;
                    packets[24 + 3] = _uv4_11.w;
                }
            }
            int ridx_7 = idx8[7];
            if (ridx_7 >= 0 && ridx_7 < rows) {
                {
                    uint4 _uv4_12 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_7 * (long long)row_stride + col64) + 0);
                    packets[28 + 0] = _uv4_12.x;
                    packets[28 + 1] = _uv4_12.y;
                    packets[28 + 2] = _uv4_12.z;
                    packets[28 + 3] = _uv4_12.w;
                }
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            __syncthreads();
            unsigned int rawk_0[4];
            rawk_0[0] = packets[0];
            rawk_0[1] = packets[1];
            rawk_0[2] = packets[2];
            rawk_0[3] = packets[3];
            uint32_t _sank_0[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_13 = static_cast<uint32_t>(rawk_0[_p16s_word]);
                uint32_t _p16s_lo_13 = _p16s_carrier_13 & 0x0000ffffu;
                uint32_t _p16s_hi_13 = (_p16s_carrier_13 >> 16) & 0x0000ffffu;
                if (_p16s_lo_13 == 0x00008000u) _p16s_lo_13 = 0u;
                if (_p16s_hi_13 == 0x00008000u) _p16s_hi_13 = 0u;
                _sank_0[_p16s_word] = _p16s_lo_13 | (_p16s_hi_13 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(tid * 16)), "r"(_sank_0[0]), "r"(_sank_0[1]), "r"(_sank_0[2]), "r"(_sank_0[3]) : "memory");
            unsigned int rawk_1[4];
            rawk_1[0] = packets[4];
            rawk_1[1] = packets[5];
            rawk_1[2] = packets[6];
            rawk_1[3] = packets[7];
            uint32_t _sank_1[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_14 = static_cast<uint32_t>(rawk_1[_p16s_word]);
                uint32_t _p16s_lo_14 = _p16s_carrier_14 & 0x0000ffffu;
                uint32_t _p16s_hi_14 = (_p16s_carrier_14 >> 16) & 0x0000ffffu;
                if (_p16s_lo_14 == 0x00008000u) _p16s_lo_14 = 0u;
                if (_p16s_hi_14 == 0x00008000u) _p16s_hi_14 = 0u;
                _sank_1[_p16s_word] = _p16s_lo_14 | (_p16s_hi_14 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(8192 + tid * 16)), "r"(_sank_1[0]), "r"(_sank_1[1]), "r"(_sank_1[2]), "r"(_sank_1[3]) : "memory");
            unsigned int rawk_2[4];
            rawk_2[0] = packets[8];
            rawk_2[1] = packets[9];
            rawk_2[2] = packets[10];
            rawk_2[3] = packets[11];
            uint32_t _sank_2[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_15 = static_cast<uint32_t>(rawk_2[_p16s_word]);
                uint32_t _p16s_lo_15 = _p16s_carrier_15 & 0x0000ffffu;
                uint32_t _p16s_hi_15 = (_p16s_carrier_15 >> 16) & 0x0000ffffu;
                if (_p16s_lo_15 == 0x00008000u) _p16s_lo_15 = 0u;
                if (_p16s_hi_15 == 0x00008000u) _p16s_hi_15 = 0u;
                _sank_2[_p16s_word] = _p16s_lo_15 | (_p16s_hi_15 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(16384 + tid * 16)), "r"(_sank_2[0]), "r"(_sank_2[1]), "r"(_sank_2[2]), "r"(_sank_2[3]) : "memory");
            unsigned int rawk_3[4];
            rawk_3[0] = packets[12];
            rawk_3[1] = packets[13];
            rawk_3[2] = packets[14];
            rawk_3[3] = packets[15];
            uint32_t _sank_3[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_16 = static_cast<uint32_t>(rawk_3[_p16s_word]);
                uint32_t _p16s_lo_16 = _p16s_carrier_16 & 0x0000ffffu;
                uint32_t _p16s_hi_16 = (_p16s_carrier_16 >> 16) & 0x0000ffffu;
                if (_p16s_lo_16 == 0x00008000u) _p16s_lo_16 = 0u;
                if (_p16s_hi_16 == 0x00008000u) _p16s_hi_16 = 0u;
                _sank_3[_p16s_word] = _p16s_lo_16 | (_p16s_hi_16 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(24576 + tid * 16)), "r"(_sank_3[0]), "r"(_sank_3[1]), "r"(_sank_3[2]), "r"(_sank_3[3]) : "memory");
            unsigned int rawk_4[4];
            rawk_4[0] = packets[16];
            rawk_4[1] = packets[17];
            rawk_4[2] = packets[18];
            rawk_4[3] = packets[19];
            uint32_t _sank_4[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_17 = static_cast<uint32_t>(rawk_4[_p16s_word]);
                uint32_t _p16s_lo_17 = _p16s_carrier_17 & 0x0000ffffu;
                uint32_t _p16s_hi_17 = (_p16s_carrier_17 >> 16) & 0x0000ffffu;
                if (_p16s_lo_17 == 0x00008000u) _p16s_lo_17 = 0u;
                if (_p16s_hi_17 == 0x00008000u) _p16s_hi_17 = 0u;
                _sank_4[_p16s_word] = _p16s_lo_17 | (_p16s_hi_17 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(32768 + tid * 16)), "r"(_sank_4[0]), "r"(_sank_4[1]), "r"(_sank_4[2]), "r"(_sank_4[3]) : "memory");
            unsigned int rawk_5[4];
            rawk_5[0] = packets[20];
            rawk_5[1] = packets[21];
            rawk_5[2] = packets[22];
            rawk_5[3] = packets[23];
            uint32_t _sank_5[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_18 = static_cast<uint32_t>(rawk_5[_p16s_word]);
                uint32_t _p16s_lo_18 = _p16s_carrier_18 & 0x0000ffffu;
                uint32_t _p16s_hi_18 = (_p16s_carrier_18 >> 16) & 0x0000ffffu;
                if (_p16s_lo_18 == 0x00008000u) _p16s_lo_18 = 0u;
                if (_p16s_hi_18 == 0x00008000u) _p16s_hi_18 = 0u;
                _sank_5[_p16s_word] = _p16s_lo_18 | (_p16s_hi_18 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(40960 + tid * 16)), "r"(_sank_5[0]), "r"(_sank_5[1]), "r"(_sank_5[2]), "r"(_sank_5[3]) : "memory");
            unsigned int rawk_6[4];
            rawk_6[0] = packets[24];
            rawk_6[1] = packets[25];
            rawk_6[2] = packets[26];
            rawk_6[3] = packets[27];
            uint32_t _sank_6[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_19 = static_cast<uint32_t>(rawk_6[_p16s_word]);
                uint32_t _p16s_lo_19 = _p16s_carrier_19 & 0x0000ffffu;
                uint32_t _p16s_hi_19 = (_p16s_carrier_19 >> 16) & 0x0000ffffu;
                if (_p16s_lo_19 == 0x00008000u) _p16s_lo_19 = 0u;
                if (_p16s_hi_19 == 0x00008000u) _p16s_hi_19 = 0u;
                _sank_6[_p16s_word] = _p16s_lo_19 | (_p16s_hi_19 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(49152 + tid * 16)), "r"(_sank_6[0]), "r"(_sank_6[1]), "r"(_sank_6[2]), "r"(_sank_6[3]) : "memory");
            unsigned int rawk_7[4];
            rawk_7[0] = packets[28];
            rawk_7[1] = packets[29];
            rawk_7[2] = packets[30];
            rawk_7[3] = packets[31];
            uint32_t _sank_7[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_20 = static_cast<uint32_t>(rawk_7[_p16s_word]);
                uint32_t _p16s_lo_20 = _p16s_carrier_20 & 0x0000ffffu;
                uint32_t _p16s_hi_20 = (_p16s_carrier_20 >> 16) & 0x0000ffffu;
                if (_p16s_lo_20 == 0x00008000u) _p16s_lo_20 = 0u;
                if (_p16s_hi_20 == 0x00008000u) _p16s_hi_20 = 0u;
                _sank_7[_p16s_word] = _p16s_lo_20 | (_p16s_hi_20 << 16);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(57344 + tid * 16)), "r"(_sank_7[0]), "r"(_sank_7[1]), "r"(_sank_7[2]), "r"(_sank_7[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid == 0) {
                asm volatile("fence.proxy.async;" ::: "memory");
                long long dstb = push_gen_elems * (long long)gb + push_slot + rank64 * 4096;
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_21 = reinterpret_cast<void*>(push_peers_0 + dstb);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_21), "r"(stg_addr + 0), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_22 = reinterpret_cast<void*>(push_peers_1 + dstb);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_22), "r"(stg_addr + 0), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_23 = reinterpret_cast<void*>(push_peers_2 + dstb);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_23), "r"(stg_addr + 0), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_24 = reinterpret_cast<void*>(push_peers_3 + dstb);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_24), "r"(stg_addr + 0), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_25 = reinterpret_cast<void*>(push_peers_0 + (dstb + 16384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_25), "r"(stg_addr + 8192), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_26 = reinterpret_cast<void*>(push_peers_1 + (dstb + 16384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_26), "r"(stg_addr + 8192), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_27 = reinterpret_cast<void*>(push_peers_2 + (dstb + 16384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_27), "r"(stg_addr + 8192), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_28 = reinterpret_cast<void*>(push_peers_3 + (dstb + 16384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_28), "r"(stg_addr + 8192), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_29 = reinterpret_cast<void*>(push_peers_0 + (dstb + 32768));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_29), "r"(stg_addr + 16384), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_30 = reinterpret_cast<void*>(push_peers_1 + (dstb + 32768));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_30), "r"(stg_addr + 16384), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_31 = reinterpret_cast<void*>(push_peers_2 + (dstb + 32768));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_31), "r"(stg_addr + 16384), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_32 = reinterpret_cast<void*>(push_peers_3 + (dstb + 32768));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_32), "r"(stg_addr + 16384), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_33 = reinterpret_cast<void*>(push_peers_0 + (dstb + 49152));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_33), "r"(stg_addr + 24576), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_34 = reinterpret_cast<void*>(push_peers_1 + (dstb + 49152));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_34), "r"(stg_addr + 24576), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_35 = reinterpret_cast<void*>(push_peers_2 + (dstb + 49152));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_35), "r"(stg_addr + 24576), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_36 = reinterpret_cast<void*>(push_peers_3 + (dstb + 49152));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_36), "r"(stg_addr + 24576), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_37 = reinterpret_cast<void*>(push_peers_0 + (dstb + 65536));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_37), "r"(stg_addr + 32768), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_38 = reinterpret_cast<void*>(push_peers_1 + (dstb + 65536));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_38), "r"(stg_addr + 32768), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_39 = reinterpret_cast<void*>(push_peers_2 + (dstb + 65536));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_39), "r"(stg_addr + 32768), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_40 = reinterpret_cast<void*>(push_peers_3 + (dstb + 65536));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_40), "r"(stg_addr + 32768), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_41 = reinterpret_cast<void*>(push_peers_0 + (dstb + 81920));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_41), "r"(stg_addr + 40960), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_42 = reinterpret_cast<void*>(push_peers_1 + (dstb + 81920));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_42), "r"(stg_addr + 40960), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_43 = reinterpret_cast<void*>(push_peers_2 + (dstb + 81920));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_43), "r"(stg_addr + 40960), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_44 = reinterpret_cast<void*>(push_peers_3 + (dstb + 81920));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_44), "r"(stg_addr + 40960), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_45 = reinterpret_cast<void*>(push_peers_0 + (dstb + 98304));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_45), "r"(stg_addr + 49152), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_46 = reinterpret_cast<void*>(push_peers_1 + (dstb + 98304));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_46), "r"(stg_addr + 49152), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_47 = reinterpret_cast<void*>(push_peers_2 + (dstb + 98304));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_47), "r"(stg_addr + 49152), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_48 = reinterpret_cast<void*>(push_peers_3 + (dstb + 98304));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_48), "r"(stg_addr + 49152), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 0) {
                    {
                        void* _cpbulk_dst_49 = reinterpret_cast<void*>(push_peers_0 + (dstb + 114688));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_49), "r"(stg_addr + 57344), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 1) {
                    {
                        void* _cpbulk_dst_50 = reinterpret_cast<void*>(push_peers_1 + (dstb + 114688));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_50), "r"(stg_addr + 57344), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 2) {
                    {
                        void* _cpbulk_dst_51 = reinterpret_cast<void*>(push_peers_2 + (dstb + 114688));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_51), "r"(stg_addr + 57344), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                if (owner == 3) {
                    {
                        void* _cpbulk_dst_52 = reinterpret_cast<void*>(push_peers_3 + (dstb + 114688));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_52), "r"(stg_addr + 57344), "r"((uint32_t)(8192))
                            : "memory");
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
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
                    uint4 _uv4_55 = *reinterpret_cast<const uint4*>(expert_weights + (tok_r * 4) + 0);
                    _vec_load_3[0 + 0] = _uv4_55.x;
                    _vec_load_3[0 + 1] = _uv4_55.y;
                    _vec_load_3[0 + 2] = _uv4_55.z;
                    _vec_load_3[0 + 3] = _uv4_55.w;
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
            uint32_t _sysv_min_56;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + src)) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 4096))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 8192))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 12288))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 16384))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 20480))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 24576))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 28672))) : "memory");
                uint32_t _sysv_min_57_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_0) : "r"(_sysv_poll_group_0[0]), "r"(_sysv_poll_group_0[1]));
                uint32_t _sysv_min_57_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_1) : "r"(_sysv_poll_group_0[2]), "r"(_sysv_poll_group_0[3]));
                uint32_t _sysv_min_57_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_2) : "r"(_sysv_min_57_0), "r"(_sysv_min_57_1));
                uint32_t _sysv_min_57_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_3) : "r"(_sysv_poll_group_0[4]), "r"(_sysv_poll_group_0[5]));
                uint32_t _sysv_min_57_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_4) : "r"(_sysv_poll_group_0[6]), "r"(_sysv_poll_group_0[7]));
                uint32_t _sysv_min_57_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_5) : "r"(_sysv_min_57_3), "r"(_sysv_min_57_4));
                uint32_t _sysv_min_57_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_6) : "r"(_sysv_min_57_2), "r"(_sysv_min_57_5));
                uint32_t _sysv_min_57_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_7) : "r"(_sysv_poll_group_0[8]), "r"(_sysv_poll_group_0[9]));
                uint32_t _sysv_min_57_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_8) : "r"(_sysv_poll_group_0[10]), "r"(_sysv_poll_group_0[11]));
                uint32_t _sysv_min_57_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_9) : "r"(_sysv_min_57_7), "r"(_sysv_min_57_8));
                uint32_t _sysv_min_57_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_10) : "r"(_sysv_poll_group_0[12]), "r"(_sysv_poll_group_0[13]));
                uint32_t _sysv_min_57_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_11) : "r"(_sysv_poll_group_0[14]), "r"(_sysv_poll_group_0[15]));
                uint32_t _sysv_min_57_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_12) : "r"(_sysv_min_57_10), "r"(_sysv_min_57_11));
                uint32_t _sysv_min_57_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_13) : "r"(_sysv_min_57_9), "r"(_sysv_min_57_12));
                uint32_t _sysv_min_57_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_14) : "r"(_sysv_min_57_6), "r"(_sysv_min_57_13));
                uint32_t _sysv_min_57_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_15) : "r"(_sysv_poll_group_0[16]), "r"(_sysv_poll_group_0[17]));
                uint32_t _sysv_min_57_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_16) : "r"(_sysv_poll_group_0[18]), "r"(_sysv_poll_group_0[19]));
                uint32_t _sysv_min_57_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_17) : "r"(_sysv_min_57_15), "r"(_sysv_min_57_16));
                uint32_t _sysv_min_57_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_18) : "r"(_sysv_poll_group_0[20]), "r"(_sysv_poll_group_0[21]));
                uint32_t _sysv_min_57_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_19) : "r"(_sysv_poll_group_0[22]), "r"(_sysv_poll_group_0[23]));
                uint32_t _sysv_min_57_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_20) : "r"(_sysv_min_57_18), "r"(_sysv_min_57_19));
                uint32_t _sysv_min_57_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_21) : "r"(_sysv_min_57_17), "r"(_sysv_min_57_20));
                uint32_t _sysv_min_57_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_22) : "r"(_sysv_poll_group_0[24]), "r"(_sysv_poll_group_0[25]));
                uint32_t _sysv_min_57_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_23) : "r"(_sysv_poll_group_0[26]), "r"(_sysv_poll_group_0[27]));
                uint32_t _sysv_min_57_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_24) : "r"(_sysv_min_57_22), "r"(_sysv_min_57_23));
                uint32_t _sysv_min_57_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_25) : "r"(_sysv_poll_group_0[28]), "r"(_sysv_poll_group_0[29]));
                uint32_t _sysv_min_57_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_26) : "r"(_sysv_poll_group_0[30]), "r"(_sysv_poll_group_0[31]));
                uint32_t _sysv_min_57_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_27) : "r"(_sysv_min_57_25), "r"(_sysv_min_57_26));
                uint32_t _sysv_min_57_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_28) : "r"(_sysv_min_57_24), "r"(_sysv_min_57_27));
                uint32_t _sysv_min_57_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_29) : "r"(_sysv_min_57_21), "r"(_sysv_min_57_28));
                uint32_t _sysv_min_57_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_57_30) : "r"(_sysv_min_57_14), "r"(_sysv_min_57_29));
                _sysv_min_56 = _sysv_min_57_30;
            } while (
                ((_sysv_min_56 & 0xffffu) == 0x8000u) || ((_sysv_min_56 >> 16) == 0x8000u)
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
            float2 sum2_0 = add_f32x2_noftz(make_float2(s8_0[0], s8_0[1]), make_float2(part_1[0], part_1[1]));
            s8_0[0] = sum2_0.x;
            s8_0[1] = sum2_0.y;
            float2 sum2_1 = add_f32x2_noftz(make_float2(s8_0[2], s8_0[3]), make_float2(part_1[2], part_1[3]));
            s8_0[2] = sum2_1.x;
            s8_0[3] = sum2_1.y;
            float2 sum2_2 = add_f32x2_noftz(make_float2(s8_0[4], s8_0[5]), make_float2(part_1[4], part_1[5]));
            s8_0[4] = sum2_2.x;
            s8_0[5] = sum2_2.y;
            float2 sum2_3 = add_f32x2_noftz(make_float2(s8_0[6], s8_0[7]), make_float2(part_1[6], part_1[7]));
            s8_0[6] = sum2_3.x;
            s8_0[7] = sum2_3.y;
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
            float2 sum2_4 = add_f32x2_noftz(make_float2(s8_0[0], s8_0[1]), make_float2(part_2[0], part_2[1]));
            s8_0[0] = sum2_4.x;
            s8_0[1] = sum2_4.y;
            float2 sum2_5 = add_f32x2_noftz(make_float2(s8_0[2], s8_0[3]), make_float2(part_2[2], part_2[3]));
            s8_0[2] = sum2_5.x;
            s8_0[3] = sum2_5.y;
            float2 sum2_6 = add_f32x2_noftz(make_float2(s8_0[4], s8_0[5]), make_float2(part_2[4], part_2[5]));
            s8_0[4] = sum2_6.x;
            s8_0[5] = sum2_6.y;
            float2 sum2_7 = add_f32x2_noftz(make_float2(s8_0[6], s8_0[7]), make_float2(part_2[6], part_2[7]));
            s8_0[6] = sum2_7.x;
            s8_0[7] = sum2_7.y;
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
            float2 sum2_8 = add_f32x2_noftz(make_float2(s8_0[0], s8_0[1]), make_float2(part_3[0], part_3[1]));
            s8_0[0] = sum2_8.x;
            s8_0[1] = sum2_8.y;
            float2 sum2_9 = add_f32x2_noftz(make_float2(s8_0[2], s8_0[3]), make_float2(part_3[2], part_3[3]));
            s8_0[2] = sum2_9.x;
            s8_0[3] = sum2_9.y;
            float2 sum2_10 = add_f32x2_noftz(make_float2(s8_0[4], s8_0[5]), make_float2(part_3[4], part_3[5]));
            s8_0[4] = sum2_10.x;
            s8_0[5] = sum2_10.y;
            float2 sum2_11 = add_f32x2_noftz(make_float2(s8_0[6], s8_0[7]), make_float2(part_3[6], part_3[7]));
            s8_0[6] = sum2_11.x;
            s8_0[7] = sum2_11.y;
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
                auto _gst_58 = push_local + src;
                __builtin_assume(__isGlobal(_gst_58));
                reinterpret_cast<int4*>(_gst_58)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_59 = push_local + (src + 4096);
                __builtin_assume(__isGlobal(_gst_59));
                reinterpret_cast<int4*>(_gst_59)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_60 = push_local + (src + 8192);
                __builtin_assume(__isGlobal(_gst_60));
                reinterpret_cast<int4*>(_gst_60)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_61 = push_local + (src + 12288);
                __builtin_assume(__isGlobal(_gst_61));
                reinterpret_cast<int4*>(_gst_61)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            float2 sum2_12 = add_f32x2_noftz(make_float2(s8_1[0], s8_1[1]), make_float2(part_5[0], part_5[1]));
            s8_1[0] = sum2_12.x;
            s8_1[1] = sum2_12.y;
            float2 sum2_13 = add_f32x2_noftz(make_float2(s8_1[2], s8_1[3]), make_float2(part_5[2], part_5[3]));
            s8_1[2] = sum2_13.x;
            s8_1[3] = sum2_13.y;
            float2 sum2_14 = add_f32x2_noftz(make_float2(s8_1[4], s8_1[5]), make_float2(part_5[4], part_5[5]));
            s8_1[4] = sum2_14.x;
            s8_1[5] = sum2_14.y;
            float2 sum2_15 = add_f32x2_noftz(make_float2(s8_1[6], s8_1[7]), make_float2(part_5[6], part_5[7]));
            s8_1[6] = sum2_15.x;
            s8_1[7] = sum2_15.y;
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
            float2 sum2_16 = add_f32x2_noftz(make_float2(s8_1[0], s8_1[1]), make_float2(part_6[0], part_6[1]));
            s8_1[0] = sum2_16.x;
            s8_1[1] = sum2_16.y;
            float2 sum2_17 = add_f32x2_noftz(make_float2(s8_1[2], s8_1[3]), make_float2(part_6[2], part_6[3]));
            s8_1[2] = sum2_17.x;
            s8_1[3] = sum2_17.y;
            float2 sum2_18 = add_f32x2_noftz(make_float2(s8_1[4], s8_1[5]), make_float2(part_6[4], part_6[5]));
            s8_1[4] = sum2_18.x;
            s8_1[5] = sum2_18.y;
            float2 sum2_19 = add_f32x2_noftz(make_float2(s8_1[6], s8_1[7]), make_float2(part_6[6], part_6[7]));
            s8_1[6] = sum2_19.x;
            s8_1[7] = sum2_19.y;
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
            float2 sum2_20 = add_f32x2_noftz(make_float2(s8_1[0], s8_1[1]), make_float2(part_7[0], part_7[1]));
            s8_1[0] = sum2_20.x;
            s8_1[1] = sum2_20.y;
            float2 sum2_21 = add_f32x2_noftz(make_float2(s8_1[2], s8_1[3]), make_float2(part_7[2], part_7[3]));
            s8_1[2] = sum2_21.x;
            s8_1[3] = sum2_21.y;
            float2 sum2_22 = add_f32x2_noftz(make_float2(s8_1[4], s8_1[5]), make_float2(part_7[4], part_7[5]));
            s8_1[4] = sum2_22.x;
            s8_1[5] = sum2_22.y;
            float2 sum2_23 = add_f32x2_noftz(make_float2(s8_1[6], s8_1[7]), make_float2(part_7[6], part_7[7]));
            s8_1[6] = sum2_23.x;
            s8_1[7] = sum2_23.y;
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
                auto _gst_62 = push_local + (src + 16384);
                __builtin_assume(__isGlobal(_gst_62));
                reinterpret_cast<int4*>(_gst_62)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_63 = push_local + (src + 20480);
                __builtin_assume(__isGlobal(_gst_63));
                reinterpret_cast<int4*>(_gst_63)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_64 = push_local + (src + 24576);
                __builtin_assume(__isGlobal(_gst_64));
                reinterpret_cast<int4*>(_gst_64)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_65 = push_local + (src + 28672);
                __builtin_assume(__isGlobal(_gst_65));
                reinterpret_cast<int4*>(_gst_65)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            uint32_t _sysv_min_66;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 32768))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 36864))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 40960))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 45056))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 49152))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 53248))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 57344))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 61440))) : "memory");
                uint32_t _sysv_min_67_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_0) : "r"(_sysv_poll_group_1[0]), "r"(_sysv_poll_group_1[1]));
                uint32_t _sysv_min_67_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_1) : "r"(_sysv_poll_group_1[2]), "r"(_sysv_poll_group_1[3]));
                uint32_t _sysv_min_67_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_2) : "r"(_sysv_min_67_0), "r"(_sysv_min_67_1));
                uint32_t _sysv_min_67_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_3) : "r"(_sysv_poll_group_1[4]), "r"(_sysv_poll_group_1[5]));
                uint32_t _sysv_min_67_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_4) : "r"(_sysv_poll_group_1[6]), "r"(_sysv_poll_group_1[7]));
                uint32_t _sysv_min_67_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_5) : "r"(_sysv_min_67_3), "r"(_sysv_min_67_4));
                uint32_t _sysv_min_67_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_6) : "r"(_sysv_min_67_2), "r"(_sysv_min_67_5));
                uint32_t _sysv_min_67_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_7) : "r"(_sysv_poll_group_1[8]), "r"(_sysv_poll_group_1[9]));
                uint32_t _sysv_min_67_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_8) : "r"(_sysv_poll_group_1[10]), "r"(_sysv_poll_group_1[11]));
                uint32_t _sysv_min_67_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_9) : "r"(_sysv_min_67_7), "r"(_sysv_min_67_8));
                uint32_t _sysv_min_67_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_10) : "r"(_sysv_poll_group_1[12]), "r"(_sysv_poll_group_1[13]));
                uint32_t _sysv_min_67_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_11) : "r"(_sysv_poll_group_1[14]), "r"(_sysv_poll_group_1[15]));
                uint32_t _sysv_min_67_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_12) : "r"(_sysv_min_67_10), "r"(_sysv_min_67_11));
                uint32_t _sysv_min_67_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_13) : "r"(_sysv_min_67_9), "r"(_sysv_min_67_12));
                uint32_t _sysv_min_67_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_14) : "r"(_sysv_min_67_6), "r"(_sysv_min_67_13));
                uint32_t _sysv_min_67_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_15) : "r"(_sysv_poll_group_1[16]), "r"(_sysv_poll_group_1[17]));
                uint32_t _sysv_min_67_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_16) : "r"(_sysv_poll_group_1[18]), "r"(_sysv_poll_group_1[19]));
                uint32_t _sysv_min_67_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_17) : "r"(_sysv_min_67_15), "r"(_sysv_min_67_16));
                uint32_t _sysv_min_67_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_18) : "r"(_sysv_poll_group_1[20]), "r"(_sysv_poll_group_1[21]));
                uint32_t _sysv_min_67_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_19) : "r"(_sysv_poll_group_1[22]), "r"(_sysv_poll_group_1[23]));
                uint32_t _sysv_min_67_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_20) : "r"(_sysv_min_67_18), "r"(_sysv_min_67_19));
                uint32_t _sysv_min_67_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_21) : "r"(_sysv_min_67_17), "r"(_sysv_min_67_20));
                uint32_t _sysv_min_67_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_22) : "r"(_sysv_poll_group_1[24]), "r"(_sysv_poll_group_1[25]));
                uint32_t _sysv_min_67_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_23) : "r"(_sysv_poll_group_1[26]), "r"(_sysv_poll_group_1[27]));
                uint32_t _sysv_min_67_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_24) : "r"(_sysv_min_67_22), "r"(_sysv_min_67_23));
                uint32_t _sysv_min_67_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_25) : "r"(_sysv_poll_group_1[28]), "r"(_sysv_poll_group_1[29]));
                uint32_t _sysv_min_67_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_26) : "r"(_sysv_poll_group_1[30]), "r"(_sysv_poll_group_1[31]));
                uint32_t _sysv_min_67_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_27) : "r"(_sysv_min_67_25), "r"(_sysv_min_67_26));
                uint32_t _sysv_min_67_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_28) : "r"(_sysv_min_67_24), "r"(_sysv_min_67_27));
                uint32_t _sysv_min_67_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_29) : "r"(_sysv_min_67_21), "r"(_sysv_min_67_28));
                uint32_t _sysv_min_67_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_67_30) : "r"(_sysv_min_67_14), "r"(_sysv_min_67_29));
                _sysv_min_66 = _sysv_min_67_30;
            } while (
                ((_sysv_min_66 & 0xffffu) == 0x8000u) || ((_sysv_min_66 >> 16) == 0x8000u)
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
            float2 sum2_24 = add_f32x2_noftz(make_float2(s8_2[0], s8_2[1]), make_float2(part_9[0], part_9[1]));
            s8_2[0] = sum2_24.x;
            s8_2[1] = sum2_24.y;
            float2 sum2_25 = add_f32x2_noftz(make_float2(s8_2[2], s8_2[3]), make_float2(part_9[2], part_9[3]));
            s8_2[2] = sum2_25.x;
            s8_2[3] = sum2_25.y;
            float2 sum2_26 = add_f32x2_noftz(make_float2(s8_2[4], s8_2[5]), make_float2(part_9[4], part_9[5]));
            s8_2[4] = sum2_26.x;
            s8_2[5] = sum2_26.y;
            float2 sum2_27 = add_f32x2_noftz(make_float2(s8_2[6], s8_2[7]), make_float2(part_9[6], part_9[7]));
            s8_2[6] = sum2_27.x;
            s8_2[7] = sum2_27.y;
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
            float2 sum2_28 = add_f32x2_noftz(make_float2(s8_2[0], s8_2[1]), make_float2(part_10[0], part_10[1]));
            s8_2[0] = sum2_28.x;
            s8_2[1] = sum2_28.y;
            float2 sum2_29 = add_f32x2_noftz(make_float2(s8_2[2], s8_2[3]), make_float2(part_10[2], part_10[3]));
            s8_2[2] = sum2_29.x;
            s8_2[3] = sum2_29.y;
            float2 sum2_30 = add_f32x2_noftz(make_float2(s8_2[4], s8_2[5]), make_float2(part_10[4], part_10[5]));
            s8_2[4] = sum2_30.x;
            s8_2[5] = sum2_30.y;
            float2 sum2_31 = add_f32x2_noftz(make_float2(s8_2[6], s8_2[7]), make_float2(part_10[6], part_10[7]));
            s8_2[6] = sum2_31.x;
            s8_2[7] = sum2_31.y;
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
            float2 sum2_32 = add_f32x2_noftz(make_float2(s8_2[0], s8_2[1]), make_float2(part_11[0], part_11[1]));
            s8_2[0] = sum2_32.x;
            s8_2[1] = sum2_32.y;
            float2 sum2_33 = add_f32x2_noftz(make_float2(s8_2[2], s8_2[3]), make_float2(part_11[2], part_11[3]));
            s8_2[2] = sum2_33.x;
            s8_2[3] = sum2_33.y;
            float2 sum2_34 = add_f32x2_noftz(make_float2(s8_2[4], s8_2[5]), make_float2(part_11[4], part_11[5]));
            s8_2[4] = sum2_34.x;
            s8_2[5] = sum2_34.y;
            float2 sum2_35 = add_f32x2_noftz(make_float2(s8_2[6], s8_2[7]), make_float2(part_11[6], part_11[7]));
            s8_2[6] = sum2_35.x;
            s8_2[7] = sum2_35.y;
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
                auto _gst_68 = push_local + (src + 32768);
                __builtin_assume(__isGlobal(_gst_68));
                reinterpret_cast<int4*>(_gst_68)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_69 = push_local + (src + 36864);
                __builtin_assume(__isGlobal(_gst_69));
                reinterpret_cast<int4*>(_gst_69)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_70 = push_local + (src + 40960);
                __builtin_assume(__isGlobal(_gst_70));
                reinterpret_cast<int4*>(_gst_70)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_71 = push_local + (src + 45056);
                __builtin_assume(__isGlobal(_gst_71));
                reinterpret_cast<int4*>(_gst_71)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            float2 sum2_36 = add_f32x2_noftz(make_float2(s8_3[0], s8_3[1]), make_float2(part_13[0], part_13[1]));
            s8_3[0] = sum2_36.x;
            s8_3[1] = sum2_36.y;
            float2 sum2_37 = add_f32x2_noftz(make_float2(s8_3[2], s8_3[3]), make_float2(part_13[2], part_13[3]));
            s8_3[2] = sum2_37.x;
            s8_3[3] = sum2_37.y;
            float2 sum2_38 = add_f32x2_noftz(make_float2(s8_3[4], s8_3[5]), make_float2(part_13[4], part_13[5]));
            s8_3[4] = sum2_38.x;
            s8_3[5] = sum2_38.y;
            float2 sum2_39 = add_f32x2_noftz(make_float2(s8_3[6], s8_3[7]), make_float2(part_13[6], part_13[7]));
            s8_3[6] = sum2_39.x;
            s8_3[7] = sum2_39.y;
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
            float2 sum2_40 = add_f32x2_noftz(make_float2(s8_3[0], s8_3[1]), make_float2(part_14[0], part_14[1]));
            s8_3[0] = sum2_40.x;
            s8_3[1] = sum2_40.y;
            float2 sum2_41 = add_f32x2_noftz(make_float2(s8_3[2], s8_3[3]), make_float2(part_14[2], part_14[3]));
            s8_3[2] = sum2_41.x;
            s8_3[3] = sum2_41.y;
            float2 sum2_42 = add_f32x2_noftz(make_float2(s8_3[4], s8_3[5]), make_float2(part_14[4], part_14[5]));
            s8_3[4] = sum2_42.x;
            s8_3[5] = sum2_42.y;
            float2 sum2_43 = add_f32x2_noftz(make_float2(s8_3[6], s8_3[7]), make_float2(part_14[6], part_14[7]));
            s8_3[6] = sum2_43.x;
            s8_3[7] = sum2_43.y;
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
            float2 sum2_44 = add_f32x2_noftz(make_float2(s8_3[0], s8_3[1]), make_float2(part_15[0], part_15[1]));
            s8_3[0] = sum2_44.x;
            s8_3[1] = sum2_44.y;
            float2 sum2_45 = add_f32x2_noftz(make_float2(s8_3[2], s8_3[3]), make_float2(part_15[2], part_15[3]));
            s8_3[2] = sum2_45.x;
            s8_3[3] = sum2_45.y;
            float2 sum2_46 = add_f32x2_noftz(make_float2(s8_3[4], s8_3[5]), make_float2(part_15[4], part_15[5]));
            s8_3[4] = sum2_46.x;
            s8_3[5] = sum2_46.y;
            float2 sum2_47 = add_f32x2_noftz(make_float2(s8_3[6], s8_3[7]), make_float2(part_15[6], part_15[7]));
            s8_3[6] = sum2_47.x;
            s8_3[7] = sum2_47.y;
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
                auto _gst_72 = push_local + (src + 49152);
                __builtin_assume(__isGlobal(_gst_72));
                reinterpret_cast<int4*>(_gst_72)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_73 = push_local + (src + 53248);
                __builtin_assume(__isGlobal(_gst_73));
                reinterpret_cast<int4*>(_gst_73)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_74 = push_local + (src + 57344);
                __builtin_assume(__isGlobal(_gst_74));
                reinterpret_cast<int4*>(_gst_74)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_75 = push_local + (src + 61440);
                __builtin_assume(__isGlobal(_gst_75));
                reinterpret_cast<int4*>(_gst_75)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            uint32_t _sysv_min_76;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[0]), "=r"(_sysv_poll_group_2[1]), "=r"(_sysv_poll_group_2[2]), "=r"(_sysv_poll_group_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 65536))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[4]), "=r"(_sysv_poll_group_2[5]), "=r"(_sysv_poll_group_2[6]), "=r"(_sysv_poll_group_2[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 69632))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[8]), "=r"(_sysv_poll_group_2[9]), "=r"(_sysv_poll_group_2[10]), "=r"(_sysv_poll_group_2[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 73728))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[12]), "=r"(_sysv_poll_group_2[13]), "=r"(_sysv_poll_group_2[14]), "=r"(_sysv_poll_group_2[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 77824))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[16]), "=r"(_sysv_poll_group_2[17]), "=r"(_sysv_poll_group_2[18]), "=r"(_sysv_poll_group_2[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 81920))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[20]), "=r"(_sysv_poll_group_2[21]), "=r"(_sysv_poll_group_2[22]), "=r"(_sysv_poll_group_2[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 86016))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[24]), "=r"(_sysv_poll_group_2[25]), "=r"(_sysv_poll_group_2[26]), "=r"(_sysv_poll_group_2[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 90112))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_2[28]), "=r"(_sysv_poll_group_2[29]), "=r"(_sysv_poll_group_2[30]), "=r"(_sysv_poll_group_2[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 94208))) : "memory");
                uint32_t _sysv_min_77_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_0) : "r"(_sysv_poll_group_2[0]), "r"(_sysv_poll_group_2[1]));
                uint32_t _sysv_min_77_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_1) : "r"(_sysv_poll_group_2[2]), "r"(_sysv_poll_group_2[3]));
                uint32_t _sysv_min_77_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_2) : "r"(_sysv_min_77_0), "r"(_sysv_min_77_1));
                uint32_t _sysv_min_77_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_3) : "r"(_sysv_poll_group_2[4]), "r"(_sysv_poll_group_2[5]));
                uint32_t _sysv_min_77_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_4) : "r"(_sysv_poll_group_2[6]), "r"(_sysv_poll_group_2[7]));
                uint32_t _sysv_min_77_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_5) : "r"(_sysv_min_77_3), "r"(_sysv_min_77_4));
                uint32_t _sysv_min_77_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_6) : "r"(_sysv_min_77_2), "r"(_sysv_min_77_5));
                uint32_t _sysv_min_77_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_7) : "r"(_sysv_poll_group_2[8]), "r"(_sysv_poll_group_2[9]));
                uint32_t _sysv_min_77_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_8) : "r"(_sysv_poll_group_2[10]), "r"(_sysv_poll_group_2[11]));
                uint32_t _sysv_min_77_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_9) : "r"(_sysv_min_77_7), "r"(_sysv_min_77_8));
                uint32_t _sysv_min_77_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_10) : "r"(_sysv_poll_group_2[12]), "r"(_sysv_poll_group_2[13]));
                uint32_t _sysv_min_77_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_11) : "r"(_sysv_poll_group_2[14]), "r"(_sysv_poll_group_2[15]));
                uint32_t _sysv_min_77_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_12) : "r"(_sysv_min_77_10), "r"(_sysv_min_77_11));
                uint32_t _sysv_min_77_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_13) : "r"(_sysv_min_77_9), "r"(_sysv_min_77_12));
                uint32_t _sysv_min_77_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_14) : "r"(_sysv_min_77_6), "r"(_sysv_min_77_13));
                uint32_t _sysv_min_77_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_15) : "r"(_sysv_poll_group_2[16]), "r"(_sysv_poll_group_2[17]));
                uint32_t _sysv_min_77_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_16) : "r"(_sysv_poll_group_2[18]), "r"(_sysv_poll_group_2[19]));
                uint32_t _sysv_min_77_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_17) : "r"(_sysv_min_77_15), "r"(_sysv_min_77_16));
                uint32_t _sysv_min_77_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_18) : "r"(_sysv_poll_group_2[20]), "r"(_sysv_poll_group_2[21]));
                uint32_t _sysv_min_77_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_19) : "r"(_sysv_poll_group_2[22]), "r"(_sysv_poll_group_2[23]));
                uint32_t _sysv_min_77_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_20) : "r"(_sysv_min_77_18), "r"(_sysv_min_77_19));
                uint32_t _sysv_min_77_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_21) : "r"(_sysv_min_77_17), "r"(_sysv_min_77_20));
                uint32_t _sysv_min_77_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_22) : "r"(_sysv_poll_group_2[24]), "r"(_sysv_poll_group_2[25]));
                uint32_t _sysv_min_77_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_23) : "r"(_sysv_poll_group_2[26]), "r"(_sysv_poll_group_2[27]));
                uint32_t _sysv_min_77_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_24) : "r"(_sysv_min_77_22), "r"(_sysv_min_77_23));
                uint32_t _sysv_min_77_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_25) : "r"(_sysv_poll_group_2[28]), "r"(_sysv_poll_group_2[29]));
                uint32_t _sysv_min_77_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_26) : "r"(_sysv_poll_group_2[30]), "r"(_sysv_poll_group_2[31]));
                uint32_t _sysv_min_77_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_27) : "r"(_sysv_min_77_25), "r"(_sysv_min_77_26));
                uint32_t _sysv_min_77_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_28) : "r"(_sysv_min_77_24), "r"(_sysv_min_77_27));
                uint32_t _sysv_min_77_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_29) : "r"(_sysv_min_77_21), "r"(_sysv_min_77_28));
                uint32_t _sysv_min_77_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_77_30) : "r"(_sysv_min_77_14), "r"(_sysv_min_77_29));
                _sysv_min_76 = _sysv_min_77_30;
            } while (
                ((_sysv_min_76 & 0xffffu) == 0x8000u) || ((_sysv_min_76 >> 16) == 0x8000u)
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
            float2 sum2_48 = add_f32x2_noftz(make_float2(s8_4[0], s8_4[1]), make_float2(part_17[0], part_17[1]));
            s8_4[0] = sum2_48.x;
            s8_4[1] = sum2_48.y;
            float2 sum2_49 = add_f32x2_noftz(make_float2(s8_4[2], s8_4[3]), make_float2(part_17[2], part_17[3]));
            s8_4[2] = sum2_49.x;
            s8_4[3] = sum2_49.y;
            float2 sum2_50 = add_f32x2_noftz(make_float2(s8_4[4], s8_4[5]), make_float2(part_17[4], part_17[5]));
            s8_4[4] = sum2_50.x;
            s8_4[5] = sum2_50.y;
            float2 sum2_51 = add_f32x2_noftz(make_float2(s8_4[6], s8_4[7]), make_float2(part_17[6], part_17[7]));
            s8_4[6] = sum2_51.x;
            s8_4[7] = sum2_51.y;
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
            float2 sum2_52 = add_f32x2_noftz(make_float2(s8_4[0], s8_4[1]), make_float2(part_18[0], part_18[1]));
            s8_4[0] = sum2_52.x;
            s8_4[1] = sum2_52.y;
            float2 sum2_53 = add_f32x2_noftz(make_float2(s8_4[2], s8_4[3]), make_float2(part_18[2], part_18[3]));
            s8_4[2] = sum2_53.x;
            s8_4[3] = sum2_53.y;
            float2 sum2_54 = add_f32x2_noftz(make_float2(s8_4[4], s8_4[5]), make_float2(part_18[4], part_18[5]));
            s8_4[4] = sum2_54.x;
            s8_4[5] = sum2_54.y;
            float2 sum2_55 = add_f32x2_noftz(make_float2(s8_4[6], s8_4[7]), make_float2(part_18[6], part_18[7]));
            s8_4[6] = sum2_55.x;
            s8_4[7] = sum2_55.y;
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
            float2 sum2_56 = add_f32x2_noftz(make_float2(s8_4[0], s8_4[1]), make_float2(part_19[0], part_19[1]));
            s8_4[0] = sum2_56.x;
            s8_4[1] = sum2_56.y;
            float2 sum2_57 = add_f32x2_noftz(make_float2(s8_4[2], s8_4[3]), make_float2(part_19[2], part_19[3]));
            s8_4[2] = sum2_57.x;
            s8_4[3] = sum2_57.y;
            float2 sum2_58 = add_f32x2_noftz(make_float2(s8_4[4], s8_4[5]), make_float2(part_19[4], part_19[5]));
            s8_4[4] = sum2_58.x;
            s8_4[5] = sum2_58.y;
            float2 sum2_59 = add_f32x2_noftz(make_float2(s8_4[6], s8_4[7]), make_float2(part_19[6], part_19[7]));
            s8_4[6] = sum2_59.x;
            s8_4[7] = sum2_59.y;
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
                auto _gst_78 = push_local + (src + 65536);
                __builtin_assume(__isGlobal(_gst_78));
                reinterpret_cast<int4*>(_gst_78)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_79 = push_local + (src + 69632);
                __builtin_assume(__isGlobal(_gst_79));
                reinterpret_cast<int4*>(_gst_79)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_80 = push_local + (src + 73728);
                __builtin_assume(__isGlobal(_gst_80));
                reinterpret_cast<int4*>(_gst_80)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_81 = push_local + (src + 77824);
                __builtin_assume(__isGlobal(_gst_81));
                reinterpret_cast<int4*>(_gst_81)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            float2 sum2_60 = add_f32x2_noftz(make_float2(s8_5[0], s8_5[1]), make_float2(part_21[0], part_21[1]));
            s8_5[0] = sum2_60.x;
            s8_5[1] = sum2_60.y;
            float2 sum2_61 = add_f32x2_noftz(make_float2(s8_5[2], s8_5[3]), make_float2(part_21[2], part_21[3]));
            s8_5[2] = sum2_61.x;
            s8_5[3] = sum2_61.y;
            float2 sum2_62 = add_f32x2_noftz(make_float2(s8_5[4], s8_5[5]), make_float2(part_21[4], part_21[5]));
            s8_5[4] = sum2_62.x;
            s8_5[5] = sum2_62.y;
            float2 sum2_63 = add_f32x2_noftz(make_float2(s8_5[6], s8_5[7]), make_float2(part_21[6], part_21[7]));
            s8_5[6] = sum2_63.x;
            s8_5[7] = sum2_63.y;
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
            float2 sum2_64 = add_f32x2_noftz(make_float2(s8_5[0], s8_5[1]), make_float2(part_22[0], part_22[1]));
            s8_5[0] = sum2_64.x;
            s8_5[1] = sum2_64.y;
            float2 sum2_65 = add_f32x2_noftz(make_float2(s8_5[2], s8_5[3]), make_float2(part_22[2], part_22[3]));
            s8_5[2] = sum2_65.x;
            s8_5[3] = sum2_65.y;
            float2 sum2_66 = add_f32x2_noftz(make_float2(s8_5[4], s8_5[5]), make_float2(part_22[4], part_22[5]));
            s8_5[4] = sum2_66.x;
            s8_5[5] = sum2_66.y;
            float2 sum2_67 = add_f32x2_noftz(make_float2(s8_5[6], s8_5[7]), make_float2(part_22[6], part_22[7]));
            s8_5[6] = sum2_67.x;
            s8_5[7] = sum2_67.y;
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
            float2 sum2_68 = add_f32x2_noftz(make_float2(s8_5[0], s8_5[1]), make_float2(part_23[0], part_23[1]));
            s8_5[0] = sum2_68.x;
            s8_5[1] = sum2_68.y;
            float2 sum2_69 = add_f32x2_noftz(make_float2(s8_5[2], s8_5[3]), make_float2(part_23[2], part_23[3]));
            s8_5[2] = sum2_69.x;
            s8_5[3] = sum2_69.y;
            float2 sum2_70 = add_f32x2_noftz(make_float2(s8_5[4], s8_5[5]), make_float2(part_23[4], part_23[5]));
            s8_5[4] = sum2_70.x;
            s8_5[5] = sum2_70.y;
            float2 sum2_71 = add_f32x2_noftz(make_float2(s8_5[6], s8_5[7]), make_float2(part_23[6], part_23[7]));
            s8_5[6] = sum2_71.x;
            s8_5[7] = sum2_71.y;
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
                auto _gst_82 = push_local + (src + 81920);
                __builtin_assume(__isGlobal(_gst_82));
                reinterpret_cast<int4*>(_gst_82)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_83 = push_local + (src + 86016);
                __builtin_assume(__isGlobal(_gst_83));
                reinterpret_cast<int4*>(_gst_83)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_84 = push_local + (src + 90112);
                __builtin_assume(__isGlobal(_gst_84));
                reinterpret_cast<int4*>(_gst_84)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_85 = push_local + (src + 94208);
                __builtin_assume(__isGlobal(_gst_85));
                reinterpret_cast<int4*>(_gst_85)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            uint32_t _sysv_min_86;
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[0]), "=r"(_sysv_poll_group_3[1]), "=r"(_sysv_poll_group_3[2]), "=r"(_sysv_poll_group_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 98304))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[4]), "=r"(_sysv_poll_group_3[5]), "=r"(_sysv_poll_group_3[6]), "=r"(_sysv_poll_group_3[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 102400))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[8]), "=r"(_sysv_poll_group_3[9]), "=r"(_sysv_poll_group_3[10]), "=r"(_sysv_poll_group_3[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 106496))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[12]), "=r"(_sysv_poll_group_3[13]), "=r"(_sysv_poll_group_3[14]), "=r"(_sysv_poll_group_3[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 110592))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[16]), "=r"(_sysv_poll_group_3[17]), "=r"(_sysv_poll_group_3[18]), "=r"(_sysv_poll_group_3[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 114688))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[20]), "=r"(_sysv_poll_group_3[21]), "=r"(_sysv_poll_group_3[22]), "=r"(_sysv_poll_group_3[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 118784))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[24]), "=r"(_sysv_poll_group_3[25]), "=r"(_sysv_poll_group_3[26]), "=r"(_sysv_poll_group_3[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 122880))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_3[28]), "=r"(_sysv_poll_group_3[29]), "=r"(_sysv_poll_group_3[30]), "=r"(_sysv_poll_group_3[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(push_local + (src + 126976))) : "memory");
                uint32_t _sysv_min_87_0;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_0) : "r"(_sysv_poll_group_3[0]), "r"(_sysv_poll_group_3[1]));
                uint32_t _sysv_min_87_1;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_1) : "r"(_sysv_poll_group_3[2]), "r"(_sysv_poll_group_3[3]));
                uint32_t _sysv_min_87_2;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_2) : "r"(_sysv_min_87_0), "r"(_sysv_min_87_1));
                uint32_t _sysv_min_87_3;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_3) : "r"(_sysv_poll_group_3[4]), "r"(_sysv_poll_group_3[5]));
                uint32_t _sysv_min_87_4;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_4) : "r"(_sysv_poll_group_3[6]), "r"(_sysv_poll_group_3[7]));
                uint32_t _sysv_min_87_5;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_5) : "r"(_sysv_min_87_3), "r"(_sysv_min_87_4));
                uint32_t _sysv_min_87_6;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_6) : "r"(_sysv_min_87_2), "r"(_sysv_min_87_5));
                uint32_t _sysv_min_87_7;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_7) : "r"(_sysv_poll_group_3[8]), "r"(_sysv_poll_group_3[9]));
                uint32_t _sysv_min_87_8;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_8) : "r"(_sysv_poll_group_3[10]), "r"(_sysv_poll_group_3[11]));
                uint32_t _sysv_min_87_9;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_9) : "r"(_sysv_min_87_7), "r"(_sysv_min_87_8));
                uint32_t _sysv_min_87_10;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_10) : "r"(_sysv_poll_group_3[12]), "r"(_sysv_poll_group_3[13]));
                uint32_t _sysv_min_87_11;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_11) : "r"(_sysv_poll_group_3[14]), "r"(_sysv_poll_group_3[15]));
                uint32_t _sysv_min_87_12;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_12) : "r"(_sysv_min_87_10), "r"(_sysv_min_87_11));
                uint32_t _sysv_min_87_13;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_13) : "r"(_sysv_min_87_9), "r"(_sysv_min_87_12));
                uint32_t _sysv_min_87_14;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_14) : "r"(_sysv_min_87_6), "r"(_sysv_min_87_13));
                uint32_t _sysv_min_87_15;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_15) : "r"(_sysv_poll_group_3[16]), "r"(_sysv_poll_group_3[17]));
                uint32_t _sysv_min_87_16;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_16) : "r"(_sysv_poll_group_3[18]), "r"(_sysv_poll_group_3[19]));
                uint32_t _sysv_min_87_17;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_17) : "r"(_sysv_min_87_15), "r"(_sysv_min_87_16));
                uint32_t _sysv_min_87_18;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_18) : "r"(_sysv_poll_group_3[20]), "r"(_sysv_poll_group_3[21]));
                uint32_t _sysv_min_87_19;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_19) : "r"(_sysv_poll_group_3[22]), "r"(_sysv_poll_group_3[23]));
                uint32_t _sysv_min_87_20;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_20) : "r"(_sysv_min_87_18), "r"(_sysv_min_87_19));
                uint32_t _sysv_min_87_21;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_21) : "r"(_sysv_min_87_17), "r"(_sysv_min_87_20));
                uint32_t _sysv_min_87_22;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_22) : "r"(_sysv_poll_group_3[24]), "r"(_sysv_poll_group_3[25]));
                uint32_t _sysv_min_87_23;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_23) : "r"(_sysv_poll_group_3[26]), "r"(_sysv_poll_group_3[27]));
                uint32_t _sysv_min_87_24;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_24) : "r"(_sysv_min_87_22), "r"(_sysv_min_87_23));
                uint32_t _sysv_min_87_25;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_25) : "r"(_sysv_poll_group_3[28]), "r"(_sysv_poll_group_3[29]));
                uint32_t _sysv_min_87_26;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_26) : "r"(_sysv_poll_group_3[30]), "r"(_sysv_poll_group_3[31]));
                uint32_t _sysv_min_87_27;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_27) : "r"(_sysv_min_87_25), "r"(_sysv_min_87_26));
                uint32_t _sysv_min_87_28;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_28) : "r"(_sysv_min_87_24), "r"(_sysv_min_87_27));
                uint32_t _sysv_min_87_29;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_29) : "r"(_sysv_min_87_21), "r"(_sysv_min_87_28));
                uint32_t _sysv_min_87_30;
                asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_87_30) : "r"(_sysv_min_87_14), "r"(_sysv_min_87_29));
                _sysv_min_86 = _sysv_min_87_30;
            } while (
                ((_sysv_min_86 & 0xffffu) == 0x8000u) || ((_sysv_min_86 >> 16) == 0x8000u)
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
            float2 sum2_72 = add_f32x2_noftz(make_float2(s8_6[0], s8_6[1]), make_float2(part_25[0], part_25[1]));
            s8_6[0] = sum2_72.x;
            s8_6[1] = sum2_72.y;
            float2 sum2_73 = add_f32x2_noftz(make_float2(s8_6[2], s8_6[3]), make_float2(part_25[2], part_25[3]));
            s8_6[2] = sum2_73.x;
            s8_6[3] = sum2_73.y;
            float2 sum2_74 = add_f32x2_noftz(make_float2(s8_6[4], s8_6[5]), make_float2(part_25[4], part_25[5]));
            s8_6[4] = sum2_74.x;
            s8_6[5] = sum2_74.y;
            float2 sum2_75 = add_f32x2_noftz(make_float2(s8_6[6], s8_6[7]), make_float2(part_25[6], part_25[7]));
            s8_6[6] = sum2_75.x;
            s8_6[7] = sum2_75.y;
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
            float2 sum2_76 = add_f32x2_noftz(make_float2(s8_6[0], s8_6[1]), make_float2(part_26[0], part_26[1]));
            s8_6[0] = sum2_76.x;
            s8_6[1] = sum2_76.y;
            float2 sum2_77 = add_f32x2_noftz(make_float2(s8_6[2], s8_6[3]), make_float2(part_26[2], part_26[3]));
            s8_6[2] = sum2_77.x;
            s8_6[3] = sum2_77.y;
            float2 sum2_78 = add_f32x2_noftz(make_float2(s8_6[4], s8_6[5]), make_float2(part_26[4], part_26[5]));
            s8_6[4] = sum2_78.x;
            s8_6[5] = sum2_78.y;
            float2 sum2_79 = add_f32x2_noftz(make_float2(s8_6[6], s8_6[7]), make_float2(part_26[6], part_26[7]));
            s8_6[6] = sum2_79.x;
            s8_6[7] = sum2_79.y;
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
            float2 sum2_80 = add_f32x2_noftz(make_float2(s8_6[0], s8_6[1]), make_float2(part_27[0], part_27[1]));
            s8_6[0] = sum2_80.x;
            s8_6[1] = sum2_80.y;
            float2 sum2_81 = add_f32x2_noftz(make_float2(s8_6[2], s8_6[3]), make_float2(part_27[2], part_27[3]));
            s8_6[2] = sum2_81.x;
            s8_6[3] = sum2_81.y;
            float2 sum2_82 = add_f32x2_noftz(make_float2(s8_6[4], s8_6[5]), make_float2(part_27[4], part_27[5]));
            s8_6[4] = sum2_82.x;
            s8_6[5] = sum2_82.y;
            float2 sum2_83 = add_f32x2_noftz(make_float2(s8_6[6], s8_6[7]), make_float2(part_27[6], part_27[7]));
            s8_6[6] = sum2_83.x;
            s8_6[7] = sum2_83.y;
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
                auto _gst_88 = push_local + (src + 98304);
                __builtin_assume(__isGlobal(_gst_88));
                reinterpret_cast<int4*>(_gst_88)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_89 = push_local + (src + 102400);
                __builtin_assume(__isGlobal(_gst_89));
                reinterpret_cast<int4*>(_gst_89)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_90 = push_local + (src + 106496);
                __builtin_assume(__isGlobal(_gst_90));
                reinterpret_cast<int4*>(_gst_90)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_91 = push_local + (src + 110592);
                __builtin_assume(__isGlobal(_gst_91));
                reinterpret_cast<int4*>(_gst_91)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            float2 sum2_84 = add_f32x2_noftz(make_float2(s8_7[0], s8_7[1]), make_float2(part_29[0], part_29[1]));
            s8_7[0] = sum2_84.x;
            s8_7[1] = sum2_84.y;
            float2 sum2_85 = add_f32x2_noftz(make_float2(s8_7[2], s8_7[3]), make_float2(part_29[2], part_29[3]));
            s8_7[2] = sum2_85.x;
            s8_7[3] = sum2_85.y;
            float2 sum2_86 = add_f32x2_noftz(make_float2(s8_7[4], s8_7[5]), make_float2(part_29[4], part_29[5]));
            s8_7[4] = sum2_86.x;
            s8_7[5] = sum2_86.y;
            float2 sum2_87 = add_f32x2_noftz(make_float2(s8_7[6], s8_7[7]), make_float2(part_29[6], part_29[7]));
            s8_7[6] = sum2_87.x;
            s8_7[7] = sum2_87.y;
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
            float2 sum2_88 = add_f32x2_noftz(make_float2(s8_7[0], s8_7[1]), make_float2(part_30[0], part_30[1]));
            s8_7[0] = sum2_88.x;
            s8_7[1] = sum2_88.y;
            float2 sum2_89 = add_f32x2_noftz(make_float2(s8_7[2], s8_7[3]), make_float2(part_30[2], part_30[3]));
            s8_7[2] = sum2_89.x;
            s8_7[3] = sum2_89.y;
            float2 sum2_90 = add_f32x2_noftz(make_float2(s8_7[4], s8_7[5]), make_float2(part_30[4], part_30[5]));
            s8_7[4] = sum2_90.x;
            s8_7[5] = sum2_90.y;
            float2 sum2_91 = add_f32x2_noftz(make_float2(s8_7[6], s8_7[7]), make_float2(part_30[6], part_30[7]));
            s8_7[6] = sum2_91.x;
            s8_7[7] = sum2_91.y;
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
            float2 sum2_92 = add_f32x2_noftz(make_float2(s8_7[0], s8_7[1]), make_float2(part_31[0], part_31[1]));
            s8_7[0] = sum2_92.x;
            s8_7[1] = sum2_92.y;
            float2 sum2_93 = add_f32x2_noftz(make_float2(s8_7[2], s8_7[3]), make_float2(part_31[2], part_31[3]));
            s8_7[2] = sum2_93.x;
            s8_7[3] = sum2_93.y;
            float2 sum2_94 = add_f32x2_noftz(make_float2(s8_7[4], s8_7[5]), make_float2(part_31[4], part_31[5]));
            s8_7[4] = sum2_94.x;
            s8_7[5] = sum2_94.y;
            float2 sum2_95 = add_f32x2_noftz(make_float2(s8_7[6], s8_7[7]), make_float2(part_31[6], part_31[7]));
            s8_7[6] = sum2_95.x;
            s8_7[7] = sum2_95.y;
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
                auto _gst_92 = push_local + (src + 114688);
                __builtin_assume(__isGlobal(_gst_92));
                reinterpret_cast<int4*>(_gst_92)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_93 = push_local + (src + 118784);
                __builtin_assume(__isGlobal(_gst_93));
                reinterpret_cast<int4*>(_gst_93)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_94 = push_local + (src + 122880);
                __builtin_assume(__isGlobal(_gst_94));
                reinterpret_cast<int4*>(_gst_94)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
            {
                auto _gst_95 = push_local + (src + 126976);
                __builtin_assume(__isGlobal(_gst_95));
                reinterpret_cast<int4*>(_gst_95)[0] = reinterpret_cast<int4*>(sent4)[0];
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
            float _sqrt_96;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_96) : "f"(__fmaf_rn(tl, 0.00024414062f, eps)));
            float rsl = __frcp_rn(_sqrt_96);
            float rs8[8];
            float _shfl_sync_97 = __shfl_sync(0xFFFFFFFF, rsl, 0);
            rs8[0] = _shfl_sync_97;
            float _shfl_sync_98 = __shfl_sync(0xFFFFFFFF, rsl, 1);
            rs8[1] = _shfl_sync_98;
            float _shfl_sync_99 = __shfl_sync(0xFFFFFFFF, rsl, 2);
            rs8[2] = _shfl_sync_99;
            float _shfl_sync_100 = __shfl_sync(0xFFFFFFFF, rsl, 3);
            rs8[3] = _shfl_sync_100;
            float _shfl_sync_101 = __shfl_sync(0xFFFFFFFF, rsl, 4);
            rs8[4] = _shfl_sync_101;
            float _shfl_sync_102 = __shfl_sync(0xFFFFFFFF, rsl, 5);
            rs8[5] = _shfl_sync_102;
            float _shfl_sync_103 = __shfl_sync(0xFFFFFFFF, rsl, 6);
            rs8[6] = _shfl_sync_103;
            float _shfl_sync_104 = __shfl_sync(0xFFFFFFFF, rsl, 7);
            rs8[7] = _shfl_sync_104;
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
            float2 _f32x2_mul_a_105 = make_float2(yna_0, ynb_0);
            float2 ov_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&rs2_0));
            float oa_0 = ((mode != 0) ? ov_0.x : ov_0.x * gw8[0]);
            float ob_0 = ((mode != 0) ? ov_0.y : ov_0.y * gw8[1]);
            float2 o2v_0 = make_float2(oa_0, ob_0);
            float2 acc2_0 = fma_f32x2_rn_ftz(o2v_0, we2_0, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_0.x;
            acc8[1] = acc2_0.y;
            float yna_1 = ((mode != 0) ? yn8_0[2] * gw8[2] : yn8_0[2]);
            float ynb_1 = ((mode != 0) ? yn8_0[3] * gw8[3] : yn8_0[3]);
            float2 _f32x2_mul_a_106 = make_float2(yna_1, ynb_1);
            float2 ov_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&rs2_0));
            float oa_1 = ((mode != 0) ? ov_1.x : ov_1.x * gw8[2]);
            float ob_1 = ((mode != 0) ? ov_1.y : ov_1.y * gw8[3]);
            float2 o2v_1 = make_float2(oa_1, ob_1);
            float2 acc2_1 = fma_f32x2_rn_ftz(o2v_1, we2_0, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_1.x;
            acc8[3] = acc2_1.y;
            float yna_2 = ((mode != 0) ? yn8_0[4] * gw8[4] : yn8_0[4]);
            float ynb_2 = ((mode != 0) ? yn8_0[5] * gw8[5] : yn8_0[5]);
            float2 _f32x2_mul_a_107 = make_float2(yna_2, ynb_2);
            float2 ov_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&rs2_0));
            float oa_2 = ((mode != 0) ? ov_2.x : ov_2.x * gw8[4]);
            float ob_2 = ((mode != 0) ? ov_2.y : ov_2.y * gw8[5]);
            float2 o2v_2 = make_float2(oa_2, ob_2);
            float2 acc2_2 = fma_f32x2_rn_ftz(o2v_2, we2_0, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_2.x;
            acc8[5] = acc2_2.y;
            float yna_3 = ((mode != 0) ? yn8_0[6] * gw8[6] : yn8_0[6]);
            float ynb_3 = ((mode != 0) ? yn8_0[7] * gw8[7] : yn8_0[7]);
            float2 _f32x2_mul_a_108 = make_float2(yna_3, ynb_3);
            float2 ov_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&rs2_0));
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
            float2 _f32x2_mul_a_109 = make_float2(yna_4, ynb_4);
            float2 ov_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&rs2_1));
            float oa_4 = ((mode != 0) ? ov_4.x : ov_4.x * gw8[0]);
            float ob_4 = ((mode != 0) ? ov_4.y : ov_4.y * gw8[1]);
            float2 o2v_4 = make_float2(oa_4, ob_4);
            float2 acc2_4 = fma_f32x2_rn_ftz(o2v_4, we2_1, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_4.x;
            acc8[1] = acc2_4.y;
            float yna_5 = ((mode != 0) ? yn8_1[2] * gw8[2] : yn8_1[2]);
            float ynb_5 = ((mode != 0) ? yn8_1[3] * gw8[3] : yn8_1[3]);
            float2 _f32x2_mul_a_110 = make_float2(yna_5, ynb_5);
            float2 ov_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&rs2_1));
            float oa_5 = ((mode != 0) ? ov_5.x : ov_5.x * gw8[2]);
            float ob_5 = ((mode != 0) ? ov_5.y : ov_5.y * gw8[3]);
            float2 o2v_5 = make_float2(oa_5, ob_5);
            float2 acc2_5 = fma_f32x2_rn_ftz(o2v_5, we2_1, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_5.x;
            acc8[3] = acc2_5.y;
            float yna_6 = ((mode != 0) ? yn8_1[4] * gw8[4] : yn8_1[4]);
            float ynb_6 = ((mode != 0) ? yn8_1[5] * gw8[5] : yn8_1[5]);
            float2 _f32x2_mul_a_111 = make_float2(yna_6, ynb_6);
            float2 ov_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&rs2_1));
            float oa_6 = ((mode != 0) ? ov_6.x : ov_6.x * gw8[4]);
            float ob_6 = ((mode != 0) ? ov_6.y : ov_6.y * gw8[5]);
            float2 o2v_6 = make_float2(oa_6, ob_6);
            float2 acc2_6 = fma_f32x2_rn_ftz(o2v_6, we2_1, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_6.x;
            acc8[5] = acc2_6.y;
            float yna_7 = ((mode != 0) ? yn8_1[6] * gw8[6] : yn8_1[6]);
            float ynb_7 = ((mode != 0) ? yn8_1[7] * gw8[7] : yn8_1[7]);
            float2 _f32x2_mul_a_112 = make_float2(yna_7, ynb_7);
            float2 ov_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&rs2_1));
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
            float2 _f32x2_mul_a_113 = make_float2(yna_8, ynb_8);
            float2 ov_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&rs2_2));
            float oa_8 = ((mode != 0) ? ov_8.x : ov_8.x * gw8[0]);
            float ob_8 = ((mode != 0) ? ov_8.y : ov_8.y * gw8[1]);
            float2 o2v_8 = make_float2(oa_8, ob_8);
            float2 acc2_8 = fma_f32x2_rn_ftz(o2v_8, we2_2, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_8.x;
            acc8[1] = acc2_8.y;
            float yna_9 = ((mode != 0) ? yn8_2[2] * gw8[2] : yn8_2[2]);
            float ynb_9 = ((mode != 0) ? yn8_2[3] * gw8[3] : yn8_2[3]);
            float2 _f32x2_mul_a_114 = make_float2(yna_9, ynb_9);
            float2 ov_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&rs2_2));
            float oa_9 = ((mode != 0) ? ov_9.x : ov_9.x * gw8[2]);
            float ob_9 = ((mode != 0) ? ov_9.y : ov_9.y * gw8[3]);
            float2 o2v_9 = make_float2(oa_9, ob_9);
            float2 acc2_9 = fma_f32x2_rn_ftz(o2v_9, we2_2, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_9.x;
            acc8[3] = acc2_9.y;
            float yna_10 = ((mode != 0) ? yn8_2[4] * gw8[4] : yn8_2[4]);
            float ynb_10 = ((mode != 0) ? yn8_2[5] * gw8[5] : yn8_2[5]);
            float2 _f32x2_mul_a_115 = make_float2(yna_10, ynb_10);
            float2 ov_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&rs2_2));
            float oa_10 = ((mode != 0) ? ov_10.x : ov_10.x * gw8[4]);
            float ob_10 = ((mode != 0) ? ov_10.y : ov_10.y * gw8[5]);
            float2 o2v_10 = make_float2(oa_10, ob_10);
            float2 acc2_10 = fma_f32x2_rn_ftz(o2v_10, we2_2, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_10.x;
            acc8[5] = acc2_10.y;
            float yna_11 = ((mode != 0) ? yn8_2[6] * gw8[6] : yn8_2[6]);
            float ynb_11 = ((mode != 0) ? yn8_2[7] * gw8[7] : yn8_2[7]);
            float2 _f32x2_mul_a_116 = make_float2(yna_11, ynb_11);
            float2 ov_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&rs2_2));
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
            float2 _f32x2_mul_a_117 = make_float2(yna_12, ynb_12);
            float2 ov_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&rs2_3));
            float oa_12 = ((mode != 0) ? ov_12.x : ov_12.x * gw8[0]);
            float ob_12 = ((mode != 0) ? ov_12.y : ov_12.y * gw8[1]);
            float2 o2v_12 = make_float2(oa_12, ob_12);
            float2 acc2_12 = fma_f32x2_rn_ftz(o2v_12, we2_3, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_12.x;
            acc8[1] = acc2_12.y;
            float yna_13 = ((mode != 0) ? yn8_3[2] * gw8[2] : yn8_3[2]);
            float ynb_13 = ((mode != 0) ? yn8_3[3] * gw8[3] : yn8_3[3]);
            float2 _f32x2_mul_a_118 = make_float2(yna_13, ynb_13);
            float2 ov_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&rs2_3));
            float oa_13 = ((mode != 0) ? ov_13.x : ov_13.x * gw8[2]);
            float ob_13 = ((mode != 0) ? ov_13.y : ov_13.y * gw8[3]);
            float2 o2v_13 = make_float2(oa_13, ob_13);
            float2 acc2_13 = fma_f32x2_rn_ftz(o2v_13, we2_3, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_13.x;
            acc8[3] = acc2_13.y;
            float yna_14 = ((mode != 0) ? yn8_3[4] * gw8[4] : yn8_3[4]);
            float ynb_14 = ((mode != 0) ? yn8_3[5] * gw8[5] : yn8_3[5]);
            float2 _f32x2_mul_a_119 = make_float2(yna_14, ynb_14);
            float2 ov_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&rs2_3));
            float oa_14 = ((mode != 0) ? ov_14.x : ov_14.x * gw8[4]);
            float ob_14 = ((mode != 0) ? ov_14.y : ov_14.y * gw8[5]);
            float2 o2v_14 = make_float2(oa_14, ob_14);
            float2 acc2_14 = fma_f32x2_rn_ftz(o2v_14, we2_3, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_14.x;
            acc8[5] = acc2_14.y;
            float yna_15 = ((mode != 0) ? yn8_3[6] * gw8[6] : yn8_3[6]);
            float ynb_15 = ((mode != 0) ? yn8_3[7] * gw8[7] : yn8_3[7]);
            float2 _f32x2_mul_a_120 = make_float2(yna_15, ynb_15);
            float2 ov_15;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&rs2_3));
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
            float2 _f32x2_mul_a_121 = make_float2(yna_16, ynb_16);
            float2 ov_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&rs2_4));
            float oa_16 = ((mode != 0) ? ov_16.x : ov_16.x * gw8[0]);
            float ob_16 = ((mode != 0) ? ov_16.y : ov_16.y * gw8[1]);
            float2 o2v_16 = make_float2(oa_16, ob_16);
            float2 acc2_16 = fma_f32x2_rn_ftz(o2v_16, we2_4, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_16.x;
            acc8[1] = acc2_16.y;
            float yna_17 = ((mode != 0) ? yn8_4[2] * gw8[2] : yn8_4[2]);
            float ynb_17 = ((mode != 0) ? yn8_4[3] * gw8[3] : yn8_4[3]);
            float2 _f32x2_mul_a_122 = make_float2(yna_17, ynb_17);
            float2 ov_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&rs2_4));
            float oa_17 = ((mode != 0) ? ov_17.x : ov_17.x * gw8[2]);
            float ob_17 = ((mode != 0) ? ov_17.y : ov_17.y * gw8[3]);
            float2 o2v_17 = make_float2(oa_17, ob_17);
            float2 acc2_17 = fma_f32x2_rn_ftz(o2v_17, we2_4, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_17.x;
            acc8[3] = acc2_17.y;
            float yna_18 = ((mode != 0) ? yn8_4[4] * gw8[4] : yn8_4[4]);
            float ynb_18 = ((mode != 0) ? yn8_4[5] * gw8[5] : yn8_4[5]);
            float2 _f32x2_mul_a_123 = make_float2(yna_18, ynb_18);
            float2 ov_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&rs2_4));
            float oa_18 = ((mode != 0) ? ov_18.x : ov_18.x * gw8[4]);
            float ob_18 = ((mode != 0) ? ov_18.y : ov_18.y * gw8[5]);
            float2 o2v_18 = make_float2(oa_18, ob_18);
            float2 acc2_18 = fma_f32x2_rn_ftz(o2v_18, we2_4, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_18.x;
            acc8[5] = acc2_18.y;
            float yna_19 = ((mode != 0) ? yn8_4[6] * gw8[6] : yn8_4[6]);
            float ynb_19 = ((mode != 0) ? yn8_4[7] * gw8[7] : yn8_4[7]);
            float2 _f32x2_mul_a_124 = make_float2(yna_19, ynb_19);
            float2 ov_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&rs2_4));
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
            float2 _f32x2_mul_a_125 = make_float2(yna_20, ynb_20);
            float2 ov_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&rs2_5));
            float oa_20 = ((mode != 0) ? ov_20.x : ov_20.x * gw8[0]);
            float ob_20 = ((mode != 0) ? ov_20.y : ov_20.y * gw8[1]);
            float2 o2v_20 = make_float2(oa_20, ob_20);
            float2 acc2_20 = fma_f32x2_rn_ftz(o2v_20, we2_5, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_20.x;
            acc8[1] = acc2_20.y;
            float yna_21 = ((mode != 0) ? yn8_5[2] * gw8[2] : yn8_5[2]);
            float ynb_21 = ((mode != 0) ? yn8_5[3] * gw8[3] : yn8_5[3]);
            float2 _f32x2_mul_a_126 = make_float2(yna_21, ynb_21);
            float2 ov_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&rs2_5));
            float oa_21 = ((mode != 0) ? ov_21.x : ov_21.x * gw8[2]);
            float ob_21 = ((mode != 0) ? ov_21.y : ov_21.y * gw8[3]);
            float2 o2v_21 = make_float2(oa_21, ob_21);
            float2 acc2_21 = fma_f32x2_rn_ftz(o2v_21, we2_5, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_21.x;
            acc8[3] = acc2_21.y;
            float yna_22 = ((mode != 0) ? yn8_5[4] * gw8[4] : yn8_5[4]);
            float ynb_22 = ((mode != 0) ? yn8_5[5] * gw8[5] : yn8_5[5]);
            float2 _f32x2_mul_a_127 = make_float2(yna_22, ynb_22);
            float2 ov_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&rs2_5));
            float oa_22 = ((mode != 0) ? ov_22.x : ov_22.x * gw8[4]);
            float ob_22 = ((mode != 0) ? ov_22.y : ov_22.y * gw8[5]);
            float2 o2v_22 = make_float2(oa_22, ob_22);
            float2 acc2_22 = fma_f32x2_rn_ftz(o2v_22, we2_5, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_22.x;
            acc8[5] = acc2_22.y;
            float yna_23 = ((mode != 0) ? yn8_5[6] * gw8[6] : yn8_5[6]);
            float ynb_23 = ((mode != 0) ? yn8_5[7] * gw8[7] : yn8_5[7]);
            float2 _f32x2_mul_a_128 = make_float2(yna_23, ynb_23);
            float2 ov_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&rs2_5));
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
            float2 _f32x2_mul_a_129 = make_float2(yna_24, ynb_24);
            float2 ov_24;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&rs2_6));
            float oa_24 = ((mode != 0) ? ov_24.x : ov_24.x * gw8[0]);
            float ob_24 = ((mode != 0) ? ov_24.y : ov_24.y * gw8[1]);
            float2 o2v_24 = make_float2(oa_24, ob_24);
            float2 acc2_24 = fma_f32x2_rn_ftz(o2v_24, we2_6, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_24.x;
            acc8[1] = acc2_24.y;
            float yna_25 = ((mode != 0) ? yn8_6[2] * gw8[2] : yn8_6[2]);
            float ynb_25 = ((mode != 0) ? yn8_6[3] * gw8[3] : yn8_6[3]);
            float2 _f32x2_mul_a_130 = make_float2(yna_25, ynb_25);
            float2 ov_25;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&rs2_6));
            float oa_25 = ((mode != 0) ? ov_25.x : ov_25.x * gw8[2]);
            float ob_25 = ((mode != 0) ? ov_25.y : ov_25.y * gw8[3]);
            float2 o2v_25 = make_float2(oa_25, ob_25);
            float2 acc2_25 = fma_f32x2_rn_ftz(o2v_25, we2_6, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_25.x;
            acc8[3] = acc2_25.y;
            float yna_26 = ((mode != 0) ? yn8_6[4] * gw8[4] : yn8_6[4]);
            float ynb_26 = ((mode != 0) ? yn8_6[5] * gw8[5] : yn8_6[5]);
            float2 _f32x2_mul_a_131 = make_float2(yna_26, ynb_26);
            float2 ov_26;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_131), "l"(*(const unsigned long long*)&rs2_6));
            float oa_26 = ((mode != 0) ? ov_26.x : ov_26.x * gw8[4]);
            float ob_26 = ((mode != 0) ? ov_26.y : ov_26.y * gw8[5]);
            float2 o2v_26 = make_float2(oa_26, ob_26);
            float2 acc2_26 = fma_f32x2_rn_ftz(o2v_26, we2_6, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_26.x;
            acc8[5] = acc2_26.y;
            float yna_27 = ((mode != 0) ? yn8_6[6] * gw8[6] : yn8_6[6]);
            float ynb_27 = ((mode != 0) ? yn8_6[7] * gw8[7] : yn8_6[7]);
            float2 _f32x2_mul_a_132 = make_float2(yna_27, ynb_27);
            float2 ov_27;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&rs2_6));
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
            float2 _f32x2_mul_a_133 = make_float2(yna_28, ynb_28);
            float2 ov_28;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&rs2_7));
            float oa_28 = ((mode != 0) ? ov_28.x : ov_28.x * gw8[0]);
            float ob_28 = ((mode != 0) ? ov_28.y : ov_28.y * gw8[1]);
            float2 o2v_28 = make_float2(oa_28, ob_28);
            float2 acc2_28 = fma_f32x2_rn_ftz(o2v_28, we2_7, make_float2(acc8[0], acc8[1]));
            acc8[0] = acc2_28.x;
            acc8[1] = acc2_28.y;
            float yna_29 = ((mode != 0) ? yn8_7[2] * gw8[2] : yn8_7[2]);
            float ynb_29 = ((mode != 0) ? yn8_7[3] * gw8[3] : yn8_7[3]);
            float2 _f32x2_mul_a_134 = make_float2(yna_29, ynb_29);
            float2 ov_29;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&rs2_7));
            float oa_29 = ((mode != 0) ? ov_29.x : ov_29.x * gw8[2]);
            float ob_29 = ((mode != 0) ? ov_29.y : ov_29.y * gw8[3]);
            float2 o2v_29 = make_float2(oa_29, ob_29);
            float2 acc2_29 = fma_f32x2_rn_ftz(o2v_29, we2_7, make_float2(acc8[2], acc8[3]));
            acc8[2] = acc2_29.x;
            acc8[3] = acc2_29.y;
            float yna_30 = ((mode != 0) ? yn8_7[4] * gw8[4] : yn8_7[4]);
            float ynb_30 = ((mode != 0) ? yn8_7[5] * gw8[5] : yn8_7[5]);
            float2 _f32x2_mul_a_135 = make_float2(yna_30, ynb_30);
            float2 ov_30;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&rs2_7));
            float oa_30 = ((mode != 0) ? ov_30.x : ov_30.x * gw8[4]);
            float ob_30 = ((mode != 0) ? ov_30.y : ov_30.y * gw8[5]);
            float2 o2v_30 = make_float2(oa_30, ob_30);
            float2 acc2_30 = fma_f32x2_rn_ftz(o2v_30, we2_7, make_float2(acc8[4], acc8[5]));
            acc8[4] = acc2_30.x;
            acc8[5] = acc2_30.y;
            float yna_31 = ((mode != 0) ? yn8_7[6] * gw8[6] : yn8_7[6]);
            float ynb_31 = ((mode != 0) ? yn8_7[7] * gw8[7] : yn8_7[7]);
            float2 _f32x2_mul_a_136 = make_float2(yna_31, ynb_31);
            float2 ov_31;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ov_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&rs2_7));
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
                uint32_t _p16s_carrier_137 = static_cast<uint32_t>(acc8_bf16[_p16s_word]);
                uint32_t _p16s_lo_137 = _p16s_carrier_137 & 0x0000ffffu;
                uint32_t _p16s_hi_137 = (_p16s_carrier_137 >> 16) & 0x0000ffffu;
                if (_p16s_lo_137 == 0x00008000u) _p16s_lo_137 = 0u;
                if (_p16s_hi_137 == 0x00008000u) _p16s_hi_137 = 0u;
                _packed16_sanitize_8[_p16s_word] = _p16s_lo_137 | (_p16s_hi_137 << 16);
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
                auto _gst_138 = bcast_local + land;
                __builtin_assume(__isGlobal(_gst_138));
                reinterpret_cast<int4*>(_gst_138)[0] = reinterpret_cast<int4*>(sent4)[0];
            }
        }
    }
    if (tid == 0) {
        asm volatile("cp.async.bulk.wait_group 0;");
    }
    int s_end = s0 + steps;
    __syncthreads();
    if (tid == 0) {
        *(reinterpret_cast<int*>(ctl + b) + (0)) = s_end;
    }
}

} // extern "C"
