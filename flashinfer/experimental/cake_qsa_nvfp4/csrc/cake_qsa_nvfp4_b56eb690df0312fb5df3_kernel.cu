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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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
#define TMEM_NCOLS 192
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_OFFSET 32
#define TMEM_TMEM_K_OFFSET 64
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 2048
#define SMEM_SMEM_Q_STRIDE 2048
#define SMEM_SMEM_QT_OFF 1024
#define SMEM_SMEM_QT_STAGE_BYTES 8192
#define SMEM_SMEM_QT_STRIDE 8192
#define SMEM_SMEM_VT_OFF 9216
#define SMEM_SMEM_VT_STAGE_BYTES 16384
#define SMEM_SMEM_VT_STRIDE 16384
#define SMEM_SMEM_P_OFF 74752
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_XCHG_OFF 78848
#define SMEM_XCHG_STAGE_BYTES 1024
#define SMEM_XCHG_STRIDE 1024
#define SMEM_VALID_S_OFF 79872
#define SMEM_VALID_S_STAGE_BYTES 32
#define SMEM_VALID_S_STRIDE 32
#define SMEM_SLOT_VALID_OFF 79904
#define SMEM_SLOT_VALID_STAGE_BYTES 16
#define SMEM_SLOT_VALID_STRIDE 16
#define SMEM_ADDR_RING_OFF 80000
#define SMEM_ADDR_RING_STAGE_BYTES 2048
#define SMEM_ADDR_RING_STRIDE 2048
#define SMEM_RAW_U8_OFF 86144
#define SMEM_RAW_U8_STAGE_BYTES 73728
#define SMEM_RAW_U8_STRIDE 73728
#define SMEM_O_XCHG_OFF 86144
#define SMEM_O_XCHG_STAGE_BYTES 16640
#define SMEM_O_XCHG_STRIDE 16640
#define SMEM_LSE_TAB_OFF 86144
#define SMEM_LSE_TAB_STAGE_BYTES 36800
#define SMEM_LSE_TAB_STRIDE 36800
#define SMEM_MAX_TAB_OFF 122944
#define SMEM_MAX_TAB_STAGE_BYTES 64
#define SMEM_MAX_TAB_STRIDE 64
#define SMEM_TOTAL 159872
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}



__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}




__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}



__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_qsa_nvfp4_b56eb690df0312fb5df3(__nv_bfloat16* __restrict__ q, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int* __restrict__ indices, int* __restrict__ block_table, int* __restrict__ token_to_req, __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, float* partial_out, float* partial_lse, unsigned int* __restrict__ merge_counters, long long q_stride_row, long long q_stride_head, long long data_stride_block, long long data_stride_token, long long data_stride_head, long long sf_stride_block, long long sf_stride_token, long long sf_stride_head, long long indices_stride_row, long long table_stride_req, long long out_stride_row, long long out_stride_head, float score_scale, float output_scale, int num_rows, int num_requests, int num_blocks, CakeFastDivmod page_div, int table_width, int topk, int group_size, int num_qo_heads, int num_tiles, int num_splits)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define raw_full_k_addr (mbar_base + 0)
    #define raw_full_v_addr (mbar_base + 32)
    #define raw_empty_addr (mbar_base + 64)
    #define valid_full_addr (mbar_base + 96)
    #define valid_empty_addr (mbar_base + 128)
    #define k_full_addr (mbar_base + 160)
    #define k_empty_addr (mbar_base + 176)
    #define s_full_addr (mbar_base + 192)
    #define s_empty_addr (mbar_base + 208)
    #define v_full_addr (mbar_base + 224)
    #define v_empty_addr (mbar_base + 240)
    #define p_full_addr (mbar_base + 256)
    #define pv_done_addr (mbar_base + 272)
    #define q_full_addr (mbar_base + 288)
    #define tmem_dead_addr (mbar_base + 296)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qt_addr = smem + 1024;
    __nv_bfloat16* smem_vt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_vt_addr = smem + 9216;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_p_addr = smem + 74752;
    float* xchg = reinterpret_cast<float*>(smem_raw + 78848);
    const int xchg_addr = smem + 78848;
    unsigned int* valid_s = reinterpret_cast<unsigned int*>(smem_raw + 79872);
    const int valid_s_addr = smem + 79872;
    unsigned int* slot_valid = reinterpret_cast<unsigned int*>(smem_raw + 79904);
    const int slot_valid_addr = smem + 79904;
    unsigned int* addr_ring = reinterpret_cast<unsigned int*>(smem_raw + 80000);
    const int addr_ring_addr = smem + 80000;
    uint8_t* raw_u8 = reinterpret_cast<uint8_t*>(smem_raw + 86144);
    const int raw_u8_addr = smem + 86144;
    float* o_xchg = reinterpret_cast<float*>(smem_raw + 86144);
    const int o_xchg_addr = smem + 86144;
    float* lse_tab = reinterpret_cast<float*>(smem_raw + 86144);
    const int lse_tab_addr = smem + 86144;
    float* max_tab = reinterpret_cast<float*>(smem_raw + 122944);
    const int max_tab_addr = smem + 122944;
    int row = blockIdx.x;
    int kv_head = blockIdx.y;
    int split = blockIdx.z;
    long long row_i64 = (long long)row;
    int first_head = kv_head * group_size;
    long long indices_row = row_i64 * indices_stride_row;
    int request = token_to_req[row];
    int valid_count = indices[indices_row + (long long)topk];
    int idx[4];
    unsigned int qw_regs[32];
    if (tid >= 384) {
        int gtid0 = tid - 384;
        int column = split * 64 + gtid0;
        int safe_column = ((column) < (topk - 1) ? (column) : (topk - 1));
        int raw = indices[indices_row + (long long)safe_column];
        idx[0] = raw;
        int column_0 = (split + num_splits) * 64 + gtid0;
        int safe_column_1 = ((column_0) < (topk - 1) ? (column_0) : (topk - 1));
        int raw_2 = indices[indices_row + (long long)safe_column_1];
        idx[1] = raw_2;
        int column_3 = (split + 2 * num_splits) * 64 + gtid0;
        int safe_column_4 = ((column_3) < (topk - 1) ? (column_3) : (topk - 1));
        int raw_5 = indices[indices_row + (long long)safe_column_4];
        idx[2] = raw_5;
        int column_6 = (split + 3 * num_splits) * 64 + gtid0;
        int safe_column_7 = ((column_6) < (topk - 1) ? (column_6) : (topk - 1));
        int raw_8 = indices[indices_row + (long long)safe_column_7];
        idx[3] = raw_8;
        long long q_row = row_i64 * q_stride_row;
        unsigned int zero_word = 0;
        qw_regs[0] = zero_word;
        qw_regs[1] = zero_word;
        qw_regs[2] = zero_word;
        qw_regs[3] = zero_word;
        qw_regs[4] = zero_word;
        qw_regs[5] = zero_word;
        qw_regs[6] = zero_word;
        qw_regs[7] = zero_word;
        qw_regs[8] = zero_word;
        qw_regs[9] = zero_word;
        qw_regs[10] = zero_word;
        qw_regs[11] = zero_word;
        qw_regs[12] = zero_word;
        qw_regs[13] = zero_word;
        qw_regs[14] = zero_word;
        qw_regs[15] = zero_word;
        qw_regs[16] = zero_word;
        qw_regs[17] = zero_word;
        qw_regs[18] = zero_word;
        qw_regs[19] = zero_word;
        qw_regs[20] = zero_word;
        qw_regs[21] = zero_word;
        qw_regs[22] = zero_word;
        qw_regs[23] = zero_word;
        qw_regs[24] = zero_word;
        qw_regs[25] = zero_word;
        qw_regs[26] = zero_word;
        qw_regs[27] = zero_word;
        qw_regs[28] = zero_word;
        qw_regs[29] = zero_word;
        qw_regs[30] = zero_word;
        qw_regs[31] = zero_word;
        int e = gtid0;
        int head = e >> 5;
        int c = e & 31;
        if (head < group_size) {
            long long q_src = q_row + (long long)(first_head + head) * q_stride_head + (long long)(8 * c);
            unsigned int _vec_load_0[4];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + q_src + 0);
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&_vec_load_0[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
            qw_regs[0] = _vec_load_0[0];
            qw_regs[1] = _vec_load_0[1];
            qw_regs[2] = _vec_load_0[2];
            qw_regs[3] = _vec_load_0[3];
        }
        int e_9 = gtid0 + 64;
        int head_10 = e_9 >> 5;
        int c_11 = e_9 & 31;
        if (head_10 < group_size) {
            long long q_src_1 = q_row + (long long)(first_head + head_10) * q_stride_head + (long long)(8 * c_11);
            unsigned int _vec_load_1[4];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(q + q_src_1 + 0);
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&_vec_load_1[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
            qw_regs[4] = _vec_load_1[0];
            qw_regs[5] = _vec_load_1[1];
            qw_regs[6] = _vec_load_1[2];
            qw_regs[7] = _vec_load_1[3];
        }
        int e_12 = gtid0 + 128;
        int head_13 = e_12 >> 5;
        int c_14 = e_12 & 31;
        if (head_13 < group_size) {
            long long q_src_2 = q_row + (long long)(first_head + head_13) * q_stride_head + (long long)(8 * c_14);
            unsigned int _vec_load_2[4];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(q + q_src_2 + 0);
                uint4* _vdst_2 = reinterpret_cast<uint4*>(&_vec_load_2[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_2[_blk] = _vptr_2[_blk];
                }
            }
            qw_regs[8] = _vec_load_2[0];
            qw_regs[9] = _vec_load_2[1];
            qw_regs[10] = _vec_load_2[2];
            qw_regs[11] = _vec_load_2[3];
        }
        int e_15 = gtid0 + 192;
        int head_16 = e_15 >> 5;
        int c_17 = e_15 & 31;
        if (head_16 < group_size) {
            long long q_src_3 = q_row + (long long)(first_head + head_16) * q_stride_head + (long long)(8 * c_17);
            unsigned int _vec_load_3[4];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(q + q_src_3 + 0);
                uint4* _vdst_3 = reinterpret_cast<uint4*>(&_vec_load_3[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_3[_blk] = _vptr_3[_blk];
                }
            }
            qw_regs[12] = _vec_load_3[0];
            qw_regs[13] = _vec_load_3[1];
            qw_regs[14] = _vec_load_3[2];
            qw_regs[15] = _vec_load_3[3];
        }
        int e_18 = gtid0 + 256;
        int head_19 = e_18 >> 5;
        int c_20 = e_18 & 31;
        if (head_19 < group_size) {
            long long q_src_4 = q_row + (long long)(first_head + head_19) * q_stride_head + (long long)(8 * c_20);
            unsigned int _vec_load_4[4];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(q + q_src_4 + 0);
                uint4* _vdst_4 = reinterpret_cast<uint4*>(&_vec_load_4[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_4[_blk] = _vptr_4[_blk];
                }
            }
            qw_regs[16] = _vec_load_4[0];
            qw_regs[17] = _vec_load_4[1];
            qw_regs[18] = _vec_load_4[2];
            qw_regs[19] = _vec_load_4[3];
        }
        int e_21 = gtid0 + 320;
        int head_22 = e_21 >> 5;
        int c_23 = e_21 & 31;
        if (head_22 < group_size) {
            long long q_src_5 = q_row + (long long)(first_head + head_22) * q_stride_head + (long long)(8 * c_23);
            unsigned int _vec_load_5[4];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(q + q_src_5 + 0);
                uint4* _vdst_5 = reinterpret_cast<uint4*>(&_vec_load_5[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_5[_blk] = _vptr_5[_blk];
                }
            }
            qw_regs[20] = _vec_load_5[0];
            qw_regs[21] = _vec_load_5[1];
            qw_regs[22] = _vec_load_5[2];
            qw_regs[23] = _vec_load_5[3];
        }
        int e_24 = gtid0 + 384;
        int head_25 = e_24 >> 5;
        int c_26 = e_24 & 31;
        if (head_25 < group_size) {
            long long q_src_6 = q_row + (long long)(first_head + head_25) * q_stride_head + (long long)(8 * c_26);
            unsigned int _vec_load_6[4];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(q + q_src_6 + 0);
                uint4* _vdst_6 = reinterpret_cast<uint4*>(&_vec_load_6[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_6[_blk] = _vptr_6[_blk];
                }
            }
            qw_regs[24] = _vec_load_6[0];
            qw_regs[25] = _vec_load_6[1];
            qw_regs[26] = _vec_load_6[2];
            qw_regs[27] = _vec_load_6[3];
        }
        int e_27 = gtid0 + 448;
        int head_28 = e_27 >> 5;
        int c_29 = e_27 & 31;
        if (head_28 < group_size) {
            long long q_src_7 = q_row + (long long)(first_head + head_28) * q_stride_head + (long long)(8 * c_29);
            unsigned int _vec_load_7[4];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(q + q_src_7 + 0);
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&_vec_load_7[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_7[_blk] = _vptr_7[_blk];
                }
            }
            qw_regs[28] = _vec_load_7[0];
            qw_regs[29] = _vec_load_7[1];
            qw_regs[30] = _vec_load_7[2];
            qw_regs[31] = _vec_load_7[3];
        }
    }

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 38 barriers)
    // Mbarriers at smem_raw[0..304)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // raw_full_k: 4 barriers, init_count=64
            mbarrier_init(smem + 0, 64);
            mbarrier_init(smem + 8, 64);
            mbarrier_init(smem + 16, 64);
            mbarrier_init(smem + 24, 64);
            // raw_full_v: 4 barriers, init_count=64
            mbarrier_init(smem + 32, 64);
            mbarrier_init(smem + 40, 64);
            mbarrier_init(smem + 48, 64);
            mbarrier_init(smem + 56, 64);
            // raw_empty: 4 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // valid_full: 4 barriers, init_count=2
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // valid_empty: 4 barriers, init_count=4
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            mbarrier_init(smem + 144, 4);
            mbarrier_init(smem + 152, 4);
            // k_full: 2 barriers, init_count=4
            mbarrier_init(smem + 160, 4);
            mbarrier_init(smem + 168, 4);
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // s_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 208, 4);
            mbarrier_init(smem + 216, 4);
            // v_full: 2 barriers, init_count=4
            mbarrier_init(smem + 224, 4);
            mbarrier_init(smem + 232, 4);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // p_full: 2 barriers, init_count=4
            mbarrier_init(smem + 256, 4);
            mbarrier_init(smem + 264, 4);
            // pv_done: 2 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 288, 2);
            // tmem_dead: 1 barriers, init_count=4
            mbarrier_init(smem + 296, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 304);
    if (warp == 0) {
        int _tmem_hold = smem + 304;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o = taddr + 32;
    const int tmem_tmem_k = taddr + 64;
    bool request_ok = request >= 0 && request < num_requests;
    int _max_0 = ((request) > (0) ? (request) : (0));
    int safe_request = ((_max_0) < (num_requests - 1) ? (_max_0) : (num_requests - 1));
    long long safe_request_row = (long long)safe_request * table_stride_req;
    int _min_1 = ((valid_count) < (topk) ? (valid_count) : (topk));
    int tile_end = ((num_tiles) < ((_min_1 + 63) / 64) ? (num_tiles) : ((_min_1 + 63) / 64));
    int n_tiles = 0;
    if (tile_end > split) {
        n_tiles = (tile_end - split + num_splits - 1) / num_splits;
    }
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // softmax_main
            int tid_0 = tid;
            int warp_1 = tid_0 >> 5;
            int lane_1 = tid_0 & 31;
            int half = lane_1 >> 4;
            int tok = 16 * warp_1 + (lane_1 & 15);
            unsigned int tok_bit = (unsigned int)(tok & 31);
            bool is_lo = half == 0;
            unsigned int s_lane = taddr + (unsigned int)(32 * warp_1 << 16);
            unsigned int o_lane = taddr + 32 + (unsigned int)(32 * warp_1 << 16);
            unsigned int xchg_max = xchg_addr;
            bool b3 = (lane_1 >> 3 & 1) != 0;
            bool b2 = (lane_1 >> 2 & 1) != 0;
            bool b1 = (lane_1 >> 1 & 1) != 0;
            bool pub_lane = (lane_1 & 1) == 0;
            unsigned int pub_off = (unsigned int)(4 * (16 * warp_1 + 8 * half + (lane_1 >> 1 & 7)));
            unsigned int xb_half_off = (unsigned int)(32 * half);
            float m[8];
            float m_new[8];
            float alpha[8];
            float other[8];
            m[0] = -1e+20f;
            m[1] = -1e+20f;
            m[2] = -1e+20f;
            m[3] = -1e+20f;
            m[4] = -1e+20f;
            m[5] = -1e+20f;
            m[6] = -1e+20f;
            m[7] = -1e+20f;
            float l[8];
            float s[8];
            float mx[8];
            float p[8];
            l[0] = 0.0f;
            l[1] = 0.0f;
            l[2] = 0.0f;
            l[3] = 0.0f;
            l[4] = 0.0f;
            l[5] = 0.0f;
            l[6] = 0.0f;
            l[7] = 0.0f;
            float a4[4];
            float a2[2];
            float mw[32];
            float ov[32];
            float sv_next[8];
            unsigned int okn[1];
            int pv_seen = 0;
            if (n_tiles > 0) {
                int kk = 0;
                int ks = kk % 2;
                int kr = kk % 4;
                mbarrier_wait(s_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int s_addr = s_lane + (unsigned int)(16 * ks) + (unsigned int)(16 * ks << 16);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[7]))
                    : "r"(s_addr));
                mbarrier_wait(valid_full_addr, 0);
                unsigned int bits = valid_s[kr * 2 + (tok >> 5)];
                okn[0] = bits >> tok_bit & 1;
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(valid_empty_addr);
                }
            }
            #pragma unroll 1
            for (int n = 0; n < n_tiles; n++) {
                int ss = n % 2;
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(s_empty_addr + ((int)((unsigned int)n % 2)) * 8);
                }
                unsigned int ok = okn[0];
                s[0] = ((ok != 0) ? sv_next[0] * score_scale : -1e+20f);
                mx[0] = s[0];
                s[1] = ((ok != 0) ? sv_next[1] * score_scale : -1e+20f);
                mx[1] = s[1];
                s[2] = ((ok != 0) ? sv_next[2] * score_scale : -1e+20f);
                mx[2] = s[2];
                s[3] = ((ok != 0) ? sv_next[3] * score_scale : -1e+20f);
                mx[3] = s[3];
                s[4] = ((ok != 0) ? sv_next[4] * score_scale : -1e+20f);
                mx[4] = s[4];
                s[5] = ((ok != 0) ? sv_next[5] * score_scale : -1e+20f);
                mx[5] = s[5];
                s[6] = ((ok != 0) ? sv_next[6] * score_scale : -1e+20f);
                mx[6] = s[6];
                s[7] = ((ok != 0) ? sv_next[7] * score_scale : -1e+20f);
                mx[7] = s[7];
                float r16[16];
                float rv = ((half == 0) ? mx[0] : -1e+20f);
                float _warp_redux_0;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_0) : "f"(rv));
                r16[0] = _warp_redux_0;
                float rv_0 = ((half == 0) ? mx[1] : -1e+20f);
                float _warp_redux_1;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_1) : "f"(rv_0));
                r16[1] = _warp_redux_1;
                float rv_1 = ((half == 0) ? mx[2] : -1e+20f);
                float _warp_redux_2;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_2) : "f"(rv_1));
                r16[2] = _warp_redux_2;
                float rv_2 = ((half == 0) ? mx[3] : -1e+20f);
                float _warp_redux_3;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_3) : "f"(rv_2));
                r16[3] = _warp_redux_3;
                float rv_3 = ((half == 0) ? mx[4] : -1e+20f);
                float _warp_redux_4;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_4) : "f"(rv_3));
                r16[4] = _warp_redux_4;
                float rv_4 = ((half == 0) ? mx[5] : -1e+20f);
                float _warp_redux_5;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_5) : "f"(rv_4));
                r16[5] = _warp_redux_5;
                float rv_5 = ((half == 0) ? mx[6] : -1e+20f);
                float _warp_redux_6;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_6) : "f"(rv_5));
                r16[6] = _warp_redux_6;
                float rv_6 = ((half == 0) ? mx[7] : -1e+20f);
                float _warp_redux_7;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_7) : "f"(rv_6));
                r16[7] = _warp_redux_7;
                float rv_7 = ((half == 1) ? mx[0] : -1e+20f);
                float _warp_redux_8;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_8) : "f"(rv_7));
                r16[8] = _warp_redux_8;
                float rv_8 = ((half == 1) ? mx[1] : -1e+20f);
                float _warp_redux_9;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_9) : "f"(rv_8));
                r16[9] = _warp_redux_9;
                float rv_9 = ((half == 1) ? mx[2] : -1e+20f);
                float _warp_redux_10;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_10) : "f"(rv_9));
                r16[10] = _warp_redux_10;
                float rv_10 = ((half == 1) ? mx[3] : -1e+20f);
                float _warp_redux_11;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_11) : "f"(rv_10));
                r16[11] = _warp_redux_11;
                float rv_11 = ((half == 1) ? mx[4] : -1e+20f);
                float _warp_redux_12;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_12) : "f"(rv_11));
                r16[12] = _warp_redux_12;
                float rv_12 = ((half == 1) ? mx[5] : -1e+20f);
                float _warp_redux_13;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_13) : "f"(rv_12));
                r16[13] = _warp_redux_13;
                float rv_13 = ((half == 1) ? mx[6] : -1e+20f);
                float _warp_redux_14;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_14) : "f"(rv_13));
                r16[14] = _warp_redux_14;
                float rv_14 = ((half == 1) ? mx[7] : -1e+20f);
                float _warp_redux_15;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_15) : "f"(rv_14));
                r16[15] = _warp_redux_15;
                unsigned int xb = xchg_max + (unsigned int)(n & 1) * 768;
                if (lane_1 == 0) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xb + (unsigned int)(64 * warp_1)), "f"(r16[0]), "f"(r16[1]), "f"(r16[2]), "f"(r16[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xb + (unsigned int)(64 * warp_1 + 16)), "f"(r16[4]), "f"(r16[5]), "f"(r16[6]), "f"(r16[7]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xb + (unsigned int)(64 * warp_1 + 32)), "f"(r16[8]), "f"(r16[9]), "f"(r16[10]), "f"(r16[11]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xb + (unsigned int)(64 * warp_1 + 48)), "f"(r16[12]), "f"(r16[13]), "f"(r16[14]), "f"(r16[15]) : "memory");
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                unsigned int xb_half = xb + xb_half_off;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 3]))
                    : "r"(xb_half));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[4])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 3]))
                    : "r"(xb_half + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[8])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(8) + 3]))
                    : "r"(xb_half + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[12])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(12) + 3]))
                    : "r"(xb_half + 64 + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[16])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(16) + 3]))
                    : "r"(xb_half + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[20])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(20) + 3]))
                    : "r"(xb_half + 128 + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[24])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(24) + 3]))
                    : "r"(xb_half + 192));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw[28])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(28) + 3]))
                    : "r"(xb_half + 192 + 16));
                bool moved = 0;
                float _max3_17;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(m[0]), "f"(mw[0]), "f"(mw[8]));
                float _max3_16;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(_max3_17), "f"(mw[16]), "f"(mw[24]));
                m_new[0] = _max3_16;
                moved = moved || m_new[0] != m[0];
                float _max3_19;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_19) : "f"(m[1]), "f"(mw[1]), "f"(mw[9]));
                float _max3_18;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(_max3_19), "f"(mw[17]), "f"(mw[25]));
                m_new[1] = _max3_18;
                moved = moved || m_new[1] != m[1];
                float _max3_21;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(m[2]), "f"(mw[2]), "f"(mw[10]));
                float _max3_20;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_20) : "f"(_max3_21), "f"(mw[18]), "f"(mw[26]));
                m_new[2] = _max3_20;
                moved = moved || m_new[2] != m[2];
                float _max3_23;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_23) : "f"(m[3]), "f"(mw[3]), "f"(mw[11]));
                float _max3_22;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(_max3_23), "f"(mw[19]), "f"(mw[27]));
                m_new[3] = _max3_22;
                moved = moved || m_new[3] != m[3];
                float _max3_25;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(m[4]), "f"(mw[4]), "f"(mw[12]));
                float _max3_24;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(_max3_25), "f"(mw[20]), "f"(mw[28]));
                m_new[4] = _max3_24;
                moved = moved || m_new[4] != m[4];
                float _max3_27;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(m[5]), "f"(mw[5]), "f"(mw[13]));
                float _max3_26;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(_max3_27), "f"(mw[21]), "f"(mw[29]));
                m_new[5] = _max3_26;
                moved = moved || m_new[5] != m[5];
                float _max3_29;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(m[6]), "f"(mw[6]), "f"(mw[14]));
                float _max3_28;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(_max3_29), "f"(mw[22]), "f"(mw[30]));
                m_new[6] = _max3_28;
                moved = moved || m_new[6] != m[6];
                float _max3_31;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_31) : "f"(m[7]), "f"(mw[7]), "f"(mw[15]));
                float _max3_30;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(_max3_31), "f"(mw[23]), "f"(mw[31]));
                m_new[7] = _max3_30;
                moved = moved || m_new[7] != m[7];
                int _warp_vote_32 = __any_sync(0xFFFFFFFF, moved);
                bool any_rescale = _warp_vote_32;
                float _exp2_33 = approx_exp2(s[0] - m_new[0]);
                p[0] = ((ok != 0) ? _exp2_33 : 0.0f);
                float _exp2_34 = approx_exp2(s[1] - m_new[1]);
                p[1] = ((ok != 0) ? _exp2_34 : 0.0f);
                float _exp2_35 = approx_exp2(s[2] - m_new[2]);
                p[2] = ((ok != 0) ? _exp2_35 : 0.0f);
                float _exp2_36 = approx_exp2(s[3] - m_new[3]);
                p[3] = ((ok != 0) ? _exp2_36 : 0.0f);
                float _exp2_37 = approx_exp2(s[4] - m_new[4]);
                p[4] = ((ok != 0) ? _exp2_37 : 0.0f);
                float _exp2_38 = approx_exp2(s[5] - m_new[5]);
                p[5] = ((ok != 0) ? _exp2_38 : 0.0f);
                float _exp2_39 = approx_exp2(s[6] - m_new[6]);
                p[6] = ((ok != 0) ? _exp2_39 : 0.0f);
                float _exp2_40 = approx_exp2(s[7] - m_new[7]);
                p[7] = ((ok != 0) ? _exp2_40 : 0.0f);
                if (any_rescale) {
                    float _exp2_41 = approx_exp2(m[0] - m_new[0]);
                    alpha[0] = _exp2_41;
                    l[0] = __fmaf_rn(l[0], alpha[0], p[0]);
                    float _exp2_42 = approx_exp2(m[1] - m_new[1]);
                    alpha[1] = _exp2_42;
                    l[1] = __fmaf_rn(l[1], alpha[1], p[1]);
                    float _exp2_43 = approx_exp2(m[2] - m_new[2]);
                    alpha[2] = _exp2_43;
                    l[2] = __fmaf_rn(l[2], alpha[2], p[2]);
                    float _exp2_44 = approx_exp2(m[3] - m_new[3]);
                    alpha[3] = _exp2_44;
                    l[3] = __fmaf_rn(l[3], alpha[3], p[3]);
                    float _exp2_45 = approx_exp2(m[4] - m_new[4]);
                    alpha[4] = _exp2_45;
                    l[4] = __fmaf_rn(l[4], alpha[4], p[4]);
                    float _exp2_46 = approx_exp2(m[5] - m_new[5]);
                    alpha[5] = _exp2_46;
                    l[5] = __fmaf_rn(l[5], alpha[5], p[5]);
                    float _exp2_47 = approx_exp2(m[6] - m_new[6]);
                    alpha[6] = _exp2_47;
                    l[6] = __fmaf_rn(l[6], alpha[6], p[6]);
                    float _exp2_48 = approx_exp2(m[7] - m_new[7]);
                    alpha[7] = _exp2_48;
                    l[7] = __fmaf_rn(l[7], alpha[7], p[7]);
                } else {
                    l[0] = l[0] + p[0];
                    l[1] = l[1] + p[1];
                    l[2] = l[2] + p[2];
                    l[3] = l[3] + p[3];
                    l[4] = l[4] + p[4];
                    l[5] = l[5] + p[5];
                    l[6] = l[6] + p[6];
                    l[7] = l[7] + p[7];
                }
                m[0] = m_new[0];
                m[1] = m_new[1];
                m[2] = m_new[2];
                m[3] = m_new[3];
                m[4] = m_new[4];
                m[5] = m_new[5];
                m[6] = m_new[6];
                m[7] = m_new[7];
                if (n >= 2) {
                    int _stage_pv_done = (int)((unsigned int)(n - 2) % 2);
                    int _parity_pv_done = (int)((unsigned int)(n - 2) / 2 % 2);
                    mbarrier_wait(pv_done_addr + (_stage_pv_done) * 8, _parity_pv_done);
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)(ss * 2048) + (unsigned int)(tok * 32 + 16 * half ^ (tok * 32 + 16 * half >> 7 & 1) << 4))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[0], p[1])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[2], p[3])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[4], p[5])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[6], p[7])))) : "memory");
                if (n > 0 && any_rescale) {
                    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, alpha[0], 16);
                    other[0] = _shfl_xor_sync_49;
                    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, alpha[1], 16);
                    other[1] = _shfl_xor_sync_50;
                    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, alpha[2], 16);
                    other[2] = _shfl_xor_sync_51;
                    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, alpha[3], 16);
                    other[3] = _shfl_xor_sync_52;
                    float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, alpha[4], 16);
                    other[4] = _shfl_xor_sync_53;
                    float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, alpha[5], 16);
                    other[5] = _shfl_xor_sync_54;
                    float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, alpha[6], 16);
                    other[6] = _shfl_xor_sync_55;
                    float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, alpha[7], 16);
                    other[7] = _shfl_xor_sync_56;
                    int _stage_pv_done_1 = (int)((unsigned int)(n - 1) % 2);
                    int _parity_pv_done_1 = (int)((unsigned int)(n - 1) / 2 % 2);
                    mbarrier_wait(pv_done_addr + (_stage_pv_done_1) * 8, _parity_pv_done_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    tmem_ld_x16(&ov[0], o_lane);
                    tmem_ld_x16(&ov[16], o_lane + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    ov[0] = ov[0] * ((is_lo) ? alpha[0] : other[0]);
                    ov[8] = ov[8] * ((is_lo) ? other[0] : alpha[0]);
                    ov[1] = ov[1] * ((is_lo) ? alpha[1] : other[1]);
                    ov[9] = ov[9] * ((is_lo) ? other[1] : alpha[1]);
                    ov[2] = ov[2] * ((is_lo) ? alpha[2] : other[2]);
                    ov[10] = ov[10] * ((is_lo) ? other[2] : alpha[2]);
                    ov[3] = ov[3] * ((is_lo) ? alpha[3] : other[3]);
                    ov[11] = ov[11] * ((is_lo) ? other[3] : alpha[3]);
                    ov[4] = ov[4] * ((is_lo) ? alpha[4] : other[4]);
                    ov[12] = ov[12] * ((is_lo) ? other[4] : alpha[4]);
                    ov[5] = ov[5] * ((is_lo) ? alpha[5] : other[5]);
                    ov[13] = ov[13] * ((is_lo) ? other[5] : alpha[5]);
                    ov[6] = ov[6] * ((is_lo) ? alpha[6] : other[6]);
                    ov[14] = ov[14] * ((is_lo) ? other[6] : alpha[6]);
                    ov[7] = ov[7] * ((is_lo) ? alpha[7] : other[7]);
                    ov[15] = ov[15] * ((is_lo) ? other[7] : alpha[7]);
                    ov[16] = ov[16] * ((is_lo) ? alpha[0] : other[0]);
                    ov[24] = ov[24] * ((is_lo) ? other[0] : alpha[0]);
                    ov[17] = ov[17] * ((is_lo) ? alpha[1] : other[1]);
                    ov[25] = ov[25] * ((is_lo) ? other[1] : alpha[1]);
                    ov[18] = ov[18] * ((is_lo) ? alpha[2] : other[2]);
                    ov[26] = ov[26] * ((is_lo) ? other[2] : alpha[2]);
                    ov[19] = ov[19] * ((is_lo) ? alpha[3] : other[3]);
                    ov[27] = ov[27] * ((is_lo) ? other[3] : alpha[3]);
                    ov[20] = ov[20] * ((is_lo) ? alpha[4] : other[4]);
                    ov[28] = ov[28] * ((is_lo) ? other[4] : alpha[4]);
                    ov[21] = ov[21] * ((is_lo) ? alpha[5] : other[5]);
                    ov[29] = ov[29] * ((is_lo) ? other[5] : alpha[5]);
                    ov[22] = ov[22] * ((is_lo) ? alpha[6] : other[6]);
                    ov[30] = ov[30] * ((is_lo) ? other[6] : alpha[6]);
                    ov[23] = ov[23] * ((is_lo) ? alpha[7] : other[7]);
                    ov[31] = ov[31] * ((is_lo) ? other[7] : alpha[7]);
                    tmem_st_x16_f32(o_lane, ov);
                    tmem_st_x16_f32(o_lane + 16, (ov + 16));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(p_full_addr + ((int)((unsigned int)n % 2)) * 8);
                }
                if (n_tiles > n + 1) {
                    int kk_1 = n + 1;
                    int ks_1 = kk_1 % 2;
                    int kr_1 = kk_1 % 4;
                    int _stage_s_full = (int)((unsigned int)(n + 1) % 2);
                    int _parity_s_full = (int)((unsigned int)(n + 1) / 2 % 2);
                    mbarrier_wait(s_full_addr + (_stage_s_full) * 8, _parity_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int s_addr_1 = s_lane + (unsigned int)(16 * ks_1) + (unsigned int)(16 * ks_1 << 16);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sv_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[7]))
                        : "r"(s_addr_1));
                    int _stage_valid_full = (int)((unsigned int)(n + 1) % 4);
                    int _parity_valid_full = (int)((unsigned int)(n + 1) / 4 % 2);
                    mbarrier_wait(valid_full_addr + (_stage_valid_full) * 8, _parity_valid_full);
                    unsigned int bits_1 = valid_s[kr_1 * 2 + (tok >> 5)];
                    okn[0] = bits_1 >> tok_bit & 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(valid_empty_addr + ((int)((unsigned int)(n + 1) % 4)) * 8);
                    }
                }
            }
            float o_acc[32];
            float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, l[0], 1);
            l[0] = l[0] + _shfl_xor_sync_57;
            float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, l[1], 1);
            l[1] = l[1] + _shfl_xor_sync_58;
            float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, l[2], 1);
            l[2] = l[2] + _shfl_xor_sync_59;
            float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, l[3], 1);
            l[3] = l[3] + _shfl_xor_sync_60;
            float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, l[4], 1);
            l[4] = l[4] + _shfl_xor_sync_61;
            float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, l[5], 1);
            l[5] = l[5] + _shfl_xor_sync_62;
            float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, l[6], 1);
            l[6] = l[6] + _shfl_xor_sync_63;
            float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, l[7], 1);
            l[7] = l[7] + _shfl_xor_sync_64;
            float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, l[0], 2);
            l[0] = l[0] + _shfl_xor_sync_65;
            float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, l[1], 2);
            l[1] = l[1] + _shfl_xor_sync_66;
            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, l[2], 2);
            l[2] = l[2] + _shfl_xor_sync_67;
            float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, l[3], 2);
            l[3] = l[3] + _shfl_xor_sync_68;
            float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, l[4], 2);
            l[4] = l[4] + _shfl_xor_sync_69;
            float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, l[5], 2);
            l[5] = l[5] + _shfl_xor_sync_70;
            float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, l[6], 2);
            l[6] = l[6] + _shfl_xor_sync_71;
            float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, l[7], 2);
            l[7] = l[7] + _shfl_xor_sync_72;
            float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, l[0], 4);
            l[0] = l[0] + _shfl_xor_sync_73;
            float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, l[1], 4);
            l[1] = l[1] + _shfl_xor_sync_74;
            float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, l[2], 4);
            l[2] = l[2] + _shfl_xor_sync_75;
            float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, l[3], 4);
            l[3] = l[3] + _shfl_xor_sync_76;
            float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, l[4], 4);
            l[4] = l[4] + _shfl_xor_sync_77;
            float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, l[5], 4);
            l[5] = l[5] + _shfl_xor_sync_78;
            float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, l[6], 4);
            l[6] = l[6] + _shfl_xor_sync_79;
            float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, l[7], 4);
            l[7] = l[7] + _shfl_xor_sync_80;
            float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, l[0], 8);
            l[0] = l[0] + _shfl_xor_sync_81;
            float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, l[1], 8);
            l[1] = l[1] + _shfl_xor_sync_82;
            float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, l[2], 8);
            l[2] = l[2] + _shfl_xor_sync_83;
            float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, l[3], 8);
            l[3] = l[3] + _shfl_xor_sync_84;
            float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, l[4], 8);
            l[4] = l[4] + _shfl_xor_sync_85;
            float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, l[5], 8);
            l[5] = l[5] + _shfl_xor_sync_86;
            float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, l[6], 8);
            l[6] = l[6] + _shfl_xor_sync_87;
            float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, l[7], 8);
            l[7] = l[7] + _shfl_xor_sync_88;
            if ((lane_1 & 15) == 0) {
                unsigned int pub_l = xchg_addr + (unsigned int)(4 * (64 + 16 * warp_1 + 8 * half));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pub_l), "f"(l[0]), "f"(l[1]), "f"(l[2]), "f"(l[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pub_l + 16), "f"(l[4]), "f"(l[5]), "f"(l[6]), "f"(l[7]) : "memory");
            }
            if (tid_0 == 0 || tid_0 == 16) {
                unsigned int pub_m = xchg_addr + 512 + xb_half_off;
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pub_m), "f"(m[0]), "f"(m[1]), "f"(m[2]), "f"(m[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pub_m + 16), "f"(m[4]), "f"(m[5]), "f"(m[6]), "f"(m[7]) : "memory");
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            float l_sum[16];
            float inv_h[16];
            float lw[16];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[0])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 3]))
                : "r"(xchg_addr + 256));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[4])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 3]))
                : "r"(xchg_addr + 272));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[8])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 3]))
                : "r"(xchg_addr + 288));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[12])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 3]))
                : "r"(xchg_addr + 304));
            l_sum[0] = lw[0];
            l_sum[1] = lw[1];
            l_sum[2] = lw[2];
            l_sum[3] = lw[3];
            l_sum[4] = lw[4];
            l_sum[5] = lw[5];
            l_sum[6] = lw[6];
            l_sum[7] = lw[7];
            l_sum[8] = lw[8];
            l_sum[9] = lw[9];
            l_sum[10] = lw[10];
            l_sum[11] = lw[11];
            l_sum[12] = lw[12];
            l_sum[13] = lw[13];
            l_sum[14] = lw[14];
            l_sum[15] = lw[15];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[0])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 3]))
                : "r"(xchg_addr + 320));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[4])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 3]))
                : "r"(xchg_addr + 336));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[8])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 3]))
                : "r"(xchg_addr + 352));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[12])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 3]))
                : "r"(xchg_addr + 368));
            l_sum[0] = l_sum[0] + lw[0];
            l_sum[1] = l_sum[1] + lw[1];
            l_sum[2] = l_sum[2] + lw[2];
            l_sum[3] = l_sum[3] + lw[3];
            l_sum[4] = l_sum[4] + lw[4];
            l_sum[5] = l_sum[5] + lw[5];
            l_sum[6] = l_sum[6] + lw[6];
            l_sum[7] = l_sum[7] + lw[7];
            l_sum[8] = l_sum[8] + lw[8];
            l_sum[9] = l_sum[9] + lw[9];
            l_sum[10] = l_sum[10] + lw[10];
            l_sum[11] = l_sum[11] + lw[11];
            l_sum[12] = l_sum[12] + lw[12];
            l_sum[13] = l_sum[13] + lw[13];
            l_sum[14] = l_sum[14] + lw[14];
            l_sum[15] = l_sum[15] + lw[15];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[0])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 3]))
                : "r"(xchg_addr + 384));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[4])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 3]))
                : "r"(xchg_addr + 400));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[8])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 3]))
                : "r"(xchg_addr + 416));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[12])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 3]))
                : "r"(xchg_addr + 432));
            l_sum[0] = l_sum[0] + lw[0];
            l_sum[1] = l_sum[1] + lw[1];
            l_sum[2] = l_sum[2] + lw[2];
            l_sum[3] = l_sum[3] + lw[3];
            l_sum[4] = l_sum[4] + lw[4];
            l_sum[5] = l_sum[5] + lw[5];
            l_sum[6] = l_sum[6] + lw[6];
            l_sum[7] = l_sum[7] + lw[7];
            l_sum[8] = l_sum[8] + lw[8];
            l_sum[9] = l_sum[9] + lw[9];
            l_sum[10] = l_sum[10] + lw[10];
            l_sum[11] = l_sum[11] + lw[11];
            l_sum[12] = l_sum[12] + lw[12];
            l_sum[13] = l_sum[13] + lw[13];
            l_sum[14] = l_sum[14] + lw[14];
            l_sum[15] = l_sum[15] + lw[15];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[0])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(0) + 3]))
                : "r"(xchg_addr + 448));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[4])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(4) + 3]))
                : "r"(xchg_addr + 464));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[8])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(8) + 3]))
                : "r"(xchg_addr + 480));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lw[12])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lw[(12) + 3]))
                : "r"(xchg_addr + 496));
            l_sum[0] = l_sum[0] + lw[0];
            l_sum[1] = l_sum[1] + lw[1];
            l_sum[2] = l_sum[2] + lw[2];
            l_sum[3] = l_sum[3] + lw[3];
            l_sum[4] = l_sum[4] + lw[4];
            l_sum[5] = l_sum[5] + lw[5];
            l_sum[6] = l_sum[6] + lw[6];
            l_sum[7] = l_sum[7] + lw[7];
            l_sum[8] = l_sum[8] + lw[8];
            l_sum[9] = l_sum[9] + lw[9];
            l_sum[10] = l_sum[10] + lw[10];
            l_sum[11] = l_sum[11] + lw[11];
            l_sum[12] = l_sum[12] + lw[12];
            l_sum[13] = l_sum[13] + lw[13];
            l_sum[14] = l_sum[14] + lw[14];
            l_sum[15] = l_sum[15] + lw[15];
            float _rcp_89 = approx_rcp(fmaxf(l_sum[0], 1e-20f));
            inv_h[0] = output_scale * _rcp_89;
            float _rcp_90 = approx_rcp(fmaxf(l_sum[1], 1e-20f));
            inv_h[1] = output_scale * _rcp_90;
            float _rcp_91 = approx_rcp(fmaxf(l_sum[2], 1e-20f));
            inv_h[2] = output_scale * _rcp_91;
            float _rcp_92 = approx_rcp(fmaxf(l_sum[3], 1e-20f));
            inv_h[3] = output_scale * _rcp_92;
            float _rcp_93 = approx_rcp(fmaxf(l_sum[4], 1e-20f));
            inv_h[4] = output_scale * _rcp_93;
            float _rcp_94 = approx_rcp(fmaxf(l_sum[5], 1e-20f));
            inv_h[5] = output_scale * _rcp_94;
            float _rcp_95 = approx_rcp(fmaxf(l_sum[6], 1e-20f));
            inv_h[6] = output_scale * _rcp_95;
            float _rcp_96 = approx_rcp(fmaxf(l_sum[7], 1e-20f));
            inv_h[7] = output_scale * _rcp_96;
            float _rcp_97 = approx_rcp(fmaxf(l_sum[8], 1e-20f));
            inv_h[8] = output_scale * _rcp_97;
            float _rcp_98 = approx_rcp(fmaxf(l_sum[9], 1e-20f));
            inv_h[9] = output_scale * _rcp_98;
            float _rcp_99 = approx_rcp(fmaxf(l_sum[10], 1e-20f));
            inv_h[10] = output_scale * _rcp_99;
            float _rcp_100 = approx_rcp(fmaxf(l_sum[11], 1e-20f));
            inv_h[11] = output_scale * _rcp_100;
            float _rcp_101 = approx_rcp(fmaxf(l_sum[12], 1e-20f));
            inv_h[12] = output_scale * _rcp_101;
            float _rcp_102 = approx_rcp(fmaxf(l_sum[13], 1e-20f));
            inv_h[13] = output_scale * _rcp_102;
            float _rcp_103 = approx_rcp(fmaxf(l_sum[14], 1e-20f));
            inv_h[14] = output_scale * _rcp_103;
            float _rcp_104 = approx_rcp(fmaxf(l_sum[15], 1e-20f));
            inv_h[15] = output_scale * _rcp_104;
            if (n_tiles > 0) {
                pv_seen += n_tiles;
                int _stage_pv_done_2 = (int)((unsigned int)(pv_seen - 1) % 2);
                int _parity_pv_done_2 = (int)((unsigned int)(pv_seen - 1) / 2 % 2);
                mbarrier_wait(pv_done_addr + (_stage_pv_done_2) * 8, _parity_pv_done_2);
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x16(&o_acc[0], o_lane);
                tmem_ld_x16(&o_acc[16], o_lane + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
            } else {
                o_acc[0] = 0.0f;
                o_acc[1] = 0.0f;
                o_acc[2] = 0.0f;
                o_acc[3] = 0.0f;
                o_acc[4] = 0.0f;
                o_acc[5] = 0.0f;
                o_acc[6] = 0.0f;
                o_acc[7] = 0.0f;
                o_acc[8] = 0.0f;
                o_acc[9] = 0.0f;
                o_acc[10] = 0.0f;
                o_acc[11] = 0.0f;
                o_acc[12] = 0.0f;
                o_acc[13] = 0.0f;
                o_acc[14] = 0.0f;
                o_acc[15] = 0.0f;
                o_acc[16] = 0.0f;
                o_acc[17] = 0.0f;
                o_acc[18] = 0.0f;
                o_acc[19] = 0.0f;
                o_acc[20] = 0.0f;
                o_acc[21] = 0.0f;
                o_acc[22] = 0.0f;
                o_acc[23] = 0.0f;
                o_acc[24] = 0.0f;
                o_acc[25] = 0.0f;
                o_acc[26] = 0.0f;
                o_acc[27] = 0.0f;
                o_acc[28] = 0.0f;
                o_acc[29] = 0.0f;
                o_acc[30] = 0.0f;
                o_acc[31] = 0.0f;
            }
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_dead_addr);
            }
            int d0 = 32 * warp_1 + lane_1;
            if (num_splits == 1) {
                o_xchg[d0] = o_acc[0];
                o_xchg[128 + d0] = o_acc[16];
                o_xchg[260 + d0] = o_acc[1];
                o_xchg[388 + d0] = o_acc[17];
                o_xchg[520 + d0] = o_acc[2];
                o_xchg[648 + d0] = o_acc[18];
                o_xchg[780 + d0] = o_acc[3];
                o_xchg[908 + d0] = o_acc[19];
                o_xchg[1040 + d0] = o_acc[4];
                o_xchg[1168 + d0] = o_acc[20];
                o_xchg[1300 + d0] = o_acc[5];
                o_xchg[1428 + d0] = o_acc[21];
                o_xchg[1560 + d0] = o_acc[6];
                o_xchg[1688 + d0] = o_acc[22];
                o_xchg[1820 + d0] = o_acc[7];
                o_xchg[1948 + d0] = o_acc[23];
                o_xchg[2080 + d0] = o_acc[8];
                o_xchg[2208 + d0] = o_acc[24];
                o_xchg[2340 + d0] = o_acc[9];
                o_xchg[2468 + d0] = o_acc[25];
                o_xchg[2600 + d0] = o_acc[10];
                o_xchg[2728 + d0] = o_acc[26];
                o_xchg[2860 + d0] = o_acc[11];
                o_xchg[2988 + d0] = o_acc[27];
                o_xchg[3120 + d0] = o_acc[12];
                o_xchg[3248 + d0] = o_acc[28];
                o_xchg[3380 + d0] = o_acc[13];
                o_xchg[3508 + d0] = o_acc[29];
                o_xchg[3640 + d0] = o_acc[14];
                o_xchg[3768 + d0] = o_acc[30];
                o_xchg[3900 + d0] = o_acc[15];
                o_xchg[4028 + d0] = o_acc[31];
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                int g = lane_1 >> 2;
                int t = lane_1 & 3;
                int dim0 = 64 * warp_1 + 16 * t;
                float v[16];
                float gated[8];
                int head_1 = g;
                if (head_1 < group_size) {
                    float l_r = xchg[64 + head_1];
                    l_r = l_r + xchg[80 + head_1];
                    l_r = l_r + xchg[96 + head_1];
                    l_r = l_r + xchg[112 + head_1];
                    unsigned int o_src = o_xchg_addr + (unsigned int)(4 * (head_1 * 260 + dim0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[0])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 3]))
                        : "r"(o_src));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[4])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 3]))
                        : "r"(o_src + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[8])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 3]))
                        : "r"(o_src + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[12])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 3]))
                        : "r"(o_src + 48));
                    bool has_values = l_r > 0.0f;
                    float _rcp_105 = approx_rcp(fmaxf(l_r, 1e-20f));
                    float inv = output_scale * _rcp_105;
                    v[0] = ((has_values) ? v[0] * inv : 0.0f);
                    v[1] = ((has_values) ? v[1] * inv : 0.0f);
                    v[2] = ((has_values) ? v[2] * inv : 0.0f);
                    v[3] = ((has_values) ? v[3] * inv : 0.0f);
                    v[4] = ((has_values) ? v[4] * inv : 0.0f);
                    v[5] = ((has_values) ? v[5] * inv : 0.0f);
                    v[6] = ((has_values) ? v[6] * inv : 0.0f);
                    v[7] = ((has_values) ? v[7] * inv : 0.0f);
                    v[8] = ((has_values) ? v[8] * inv : 0.0f);
                    v[9] = ((has_values) ? v[9] * inv : 0.0f);
                    v[10] = ((has_values) ? v[10] * inv : 0.0f);
                    v[11] = ((has_values) ? v[11] * inv : 0.0f);
                    v[12] = ((has_values) ? v[12] * inv : 0.0f);
                    v[13] = ((has_values) ? v[13] * inv : 0.0f);
                    v[14] = ((has_values) ? v[14] * inv : 0.0f);
                    v[15] = ((has_values) ? v[15] * inv : 0.0f);
                    long long qo_head = (long long)(first_head + head_1);
                    long long gate_off = (row_i64 * (long long)num_qo_heads + qo_head) * 256 + (long long)dim0;
                    long long out_off = row_i64 * out_stride_row + qo_head * out_stride_head + (long long)dim0;
                    float _vec_load_8[8];
                    {
                        const uint4* _vptr_106 = reinterpret_cast<const uint4*>(gate + gate_off + 0);
                        uint4 _vld_106[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_106[_blk] = _vptr_106[_blk];
                            uint32_t* _vpairs_106 = reinterpret_cast<uint32_t*>(&_vld_106[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_106[_pair]) << 16);
                                (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_106[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float attn = (float)(__nv_bfloat16)v[0];
                    float _exp2_107 = approx_exp2((-_vec_load_8[0]) * 1.442695f);
                    float sigmoid = approx_rcp(1.0f + _exp2_107);
                    gated[0] = attn * sigmoid;
                    float attn_0 = (float)(__nv_bfloat16)v[1];
                    float _exp2_108 = approx_exp2((-_vec_load_8[1]) * 1.442695f);
                    float sigmoid_1 = approx_rcp(1.0f + _exp2_108);
                    gated[1] = attn_0 * sigmoid_1;
                    float attn_2 = (float)(__nv_bfloat16)v[2];
                    float _exp2_109 = approx_exp2((-_vec_load_8[2]) * 1.442695f);
                    float sigmoid_3 = approx_rcp(1.0f + _exp2_109);
                    gated[2] = attn_2 * sigmoid_3;
                    float attn_4 = (float)(__nv_bfloat16)v[3];
                    float _exp2_110 = approx_exp2((-_vec_load_8[3]) * 1.442695f);
                    float sigmoid_5 = approx_rcp(1.0f + _exp2_110);
                    gated[3] = attn_4 * sigmoid_5;
                    float attn_6 = (float)(__nv_bfloat16)v[4];
                    float _exp2_111 = approx_exp2((-_vec_load_8[4]) * 1.442695f);
                    float sigmoid_7 = approx_rcp(1.0f + _exp2_111);
                    gated[4] = attn_6 * sigmoid_7;
                    float attn_8 = (float)(__nv_bfloat16)v[5];
                    float _exp2_112 = approx_exp2((-_vec_load_8[5]) * 1.442695f);
                    float sigmoid_9 = approx_rcp(1.0f + _exp2_112);
                    gated[5] = attn_8 * sigmoid_9;
                    float attn_10 = (float)(__nv_bfloat16)v[6];
                    float _exp2_113 = approx_exp2((-_vec_load_8[6]) * 1.442695f);
                    float sigmoid_11 = approx_rcp(1.0f + _exp2_113);
                    gated[6] = attn_10 * sigmoid_11;
                    float attn_12 = (float)(__nv_bfloat16)v[7];
                    float _exp2_114 = approx_exp2((-_vec_load_8[7]) * 1.442695f);
                    float sigmoid_13 = approx_rcp(1.0f + _exp2_114);
                    gated[7] = attn_12 * sigmoid_13;
                    unsigned int _packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(out + out_off)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                    float _vec_load_9[8];
                    {
                        const uint4* _vptr_115 = reinterpret_cast<const uint4*>(gate + (gate_off + 8) + 0);
                        uint4 _vld_115[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_115[_blk] = _vptr_115[_blk];
                            uint32_t* _vpairs_115 = reinterpret_cast<uint32_t*>(&_vld_115[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_115[_pair]) << 16);
                                (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_115[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float attn_14 = (float)(__nv_bfloat16)v[8];
                    float _exp2_116 = approx_exp2((-_vec_load_9[0]) * 1.442695f);
                    float sigmoid_15 = approx_rcp(1.0f + _exp2_116);
                    gated[0] = attn_14 * sigmoid_15;
                    float attn_16 = (float)(__nv_bfloat16)v[9];
                    float _exp2_117 = approx_exp2((-_vec_load_9[1]) * 1.442695f);
                    float sigmoid_17 = approx_rcp(1.0f + _exp2_117);
                    gated[1] = attn_16 * sigmoid_17;
                    float attn_18 = (float)(__nv_bfloat16)v[10];
                    float _exp2_118 = approx_exp2((-_vec_load_9[2]) * 1.442695f);
                    float sigmoid_19 = approx_rcp(1.0f + _exp2_118);
                    gated[2] = attn_18 * sigmoid_19;
                    float attn_20 = (float)(__nv_bfloat16)v[11];
                    float _exp2_119 = approx_exp2((-_vec_load_9[3]) * 1.442695f);
                    float sigmoid_21 = approx_rcp(1.0f + _exp2_119);
                    gated[3] = attn_20 * sigmoid_21;
                    float attn_22 = (float)(__nv_bfloat16)v[12];
                    float _exp2_120 = approx_exp2((-_vec_load_9[4]) * 1.442695f);
                    float sigmoid_23 = approx_rcp(1.0f + _exp2_120);
                    gated[4] = attn_22 * sigmoid_23;
                    float attn_24 = (float)(__nv_bfloat16)v[13];
                    float _exp2_121 = approx_exp2((-_vec_load_9[5]) * 1.442695f);
                    float sigmoid_25 = approx_rcp(1.0f + _exp2_121);
                    gated[5] = attn_24 * sigmoid_25;
                    float attn_26 = (float)(__nv_bfloat16)v[14];
                    float _exp2_122 = approx_exp2((-_vec_load_9[6]) * 1.442695f);
                    float sigmoid_27 = approx_rcp(1.0f + _exp2_122);
                    gated[6] = attn_26 * sigmoid_27;
                    float attn_28 = (float)(__nv_bfloat16)v[15];
                    float _exp2_123 = approx_exp2((-_vec_load_9[7]) * 1.442695f);
                    float sigmoid_29 = approx_rcp(1.0f + _exp2_123);
                    gated[7] = attn_28 * sigmoid_29;
                    unsigned int _packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(out + (out_off + 8))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                }
                int head_0 = g + 8;
                if (head_0 < group_size) {
                    float l_r_1 = xchg[64 + head_0];
                    l_r_1 = l_r_1 + xchg[80 + head_0];
                    l_r_1 = l_r_1 + xchg[96 + head_0];
                    l_r_1 = l_r_1 + xchg[112 + head_0];
                    unsigned int o_src_1 = o_xchg_addr + (unsigned int)(4 * (head_0 * 260 + dim0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[0])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 3]))
                        : "r"(o_src_1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[4])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 3]))
                        : "r"(o_src_1 + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[8])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 3]))
                        : "r"(o_src_1 + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v[12])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 3]))
                        : "r"(o_src_1 + 48));
                    bool has_values_1 = l_r_1 > 0.0f;
                    float _rcp_124 = approx_rcp(fmaxf(l_r_1, 1e-20f));
                    float inv_1 = output_scale * _rcp_124;
                    v[0] = ((has_values_1) ? v[0] * inv_1 : 0.0f);
                    v[1] = ((has_values_1) ? v[1] * inv_1 : 0.0f);
                    v[2] = ((has_values_1) ? v[2] * inv_1 : 0.0f);
                    v[3] = ((has_values_1) ? v[3] * inv_1 : 0.0f);
                    v[4] = ((has_values_1) ? v[4] * inv_1 : 0.0f);
                    v[5] = ((has_values_1) ? v[5] * inv_1 : 0.0f);
                    v[6] = ((has_values_1) ? v[6] * inv_1 : 0.0f);
                    v[7] = ((has_values_1) ? v[7] * inv_1 : 0.0f);
                    v[8] = ((has_values_1) ? v[8] * inv_1 : 0.0f);
                    v[9] = ((has_values_1) ? v[9] * inv_1 : 0.0f);
                    v[10] = ((has_values_1) ? v[10] * inv_1 : 0.0f);
                    v[11] = ((has_values_1) ? v[11] * inv_1 : 0.0f);
                    v[12] = ((has_values_1) ? v[12] * inv_1 : 0.0f);
                    v[13] = ((has_values_1) ? v[13] * inv_1 : 0.0f);
                    v[14] = ((has_values_1) ? v[14] * inv_1 : 0.0f);
                    v[15] = ((has_values_1) ? v[15] * inv_1 : 0.0f);
                    long long qo_head_1 = (long long)(first_head + head_0);
                    long long gate_off_1 = (row_i64 * (long long)num_qo_heads + qo_head_1) * 256 + (long long)dim0;
                    long long out_off_1 = row_i64 * out_stride_row + qo_head_1 * out_stride_head + (long long)dim0;
                    float _vec_load_10[8];
                    {
                        const uint4* _vptr_125 = reinterpret_cast<const uint4*>(gate + gate_off_1 + 0);
                        uint4 _vld_125[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_125[_blk] = _vptr_125[_blk];
                            uint32_t* _vpairs_125 = reinterpret_cast<uint32_t*>(&_vld_125[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_125[_pair]) << 16);
                                (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_125[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float attn_1 = (float)(__nv_bfloat16)v[0];
                    float _exp2_126 = approx_exp2((-_vec_load_10[0]) * 1.442695f);
                    float sigmoid_2 = approx_rcp(1.0f + _exp2_126);
                    gated[0] = attn_1 * sigmoid_2;
                    float attn_0_1 = (float)(__nv_bfloat16)v[1];
                    float _exp2_127 = approx_exp2((-_vec_load_10[1]) * 1.442695f);
                    float sigmoid_1_1 = approx_rcp(1.0f + _exp2_127);
                    gated[1] = attn_0_1 * sigmoid_1_1;
                    float attn_2_1 = (float)(__nv_bfloat16)v[2];
                    float _exp2_128 = approx_exp2((-_vec_load_10[2]) * 1.442695f);
                    float sigmoid_3_1 = approx_rcp(1.0f + _exp2_128);
                    gated[2] = attn_2_1 * sigmoid_3_1;
                    float attn_4_1 = (float)(__nv_bfloat16)v[3];
                    float _exp2_129 = approx_exp2((-_vec_load_10[3]) * 1.442695f);
                    float sigmoid_5_1 = approx_rcp(1.0f + _exp2_129);
                    gated[3] = attn_4_1 * sigmoid_5_1;
                    float attn_6_1 = (float)(__nv_bfloat16)v[4];
                    float _exp2_130 = approx_exp2((-_vec_load_10[4]) * 1.442695f);
                    float sigmoid_7_1 = approx_rcp(1.0f + _exp2_130);
                    gated[4] = attn_6_1 * sigmoid_7_1;
                    float attn_8_1 = (float)(__nv_bfloat16)v[5];
                    float _exp2_131 = approx_exp2((-_vec_load_10[5]) * 1.442695f);
                    float sigmoid_9_1 = approx_rcp(1.0f + _exp2_131);
                    gated[5] = attn_8_1 * sigmoid_9_1;
                    float attn_10_1 = (float)(__nv_bfloat16)v[6];
                    float _exp2_132 = approx_exp2((-_vec_load_10[6]) * 1.442695f);
                    float sigmoid_11_1 = approx_rcp(1.0f + _exp2_132);
                    gated[6] = attn_10_1 * sigmoid_11_1;
                    float attn_12_1 = (float)(__nv_bfloat16)v[7];
                    float _exp2_133 = approx_exp2((-_vec_load_10[7]) * 1.442695f);
                    float sigmoid_13_1 = approx_rcp(1.0f + _exp2_133);
                    gated[7] = attn_12_1 * sigmoid_13_1;
                    unsigned int _packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                        _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                        _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                        _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(out + out_off_1)[0] = reinterpret_cast<int4*>(_packed_2)[0];
                    float _vec_load_11[8];
                    {
                        const uint4* _vptr_134 = reinterpret_cast<const uint4*>(gate + (gate_off_1 + 8) + 0);
                        uint4 _vld_134[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_134[_blk] = _vptr_134[_blk];
                            uint32_t* _vpairs_134 = reinterpret_cast<uint32_t*>(&_vld_134[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_134[_pair]) << 16);
                                (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_134[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float attn_14_1 = (float)(__nv_bfloat16)v[8];
                    float _exp2_135 = approx_exp2((-_vec_load_11[0]) * 1.442695f);
                    float sigmoid_15_1 = approx_rcp(1.0f + _exp2_135);
                    gated[0] = attn_14_1 * sigmoid_15_1;
                    float attn_16_1 = (float)(__nv_bfloat16)v[9];
                    float _exp2_136 = approx_exp2((-_vec_load_11[1]) * 1.442695f);
                    float sigmoid_17_1 = approx_rcp(1.0f + _exp2_136);
                    gated[1] = attn_16_1 * sigmoid_17_1;
                    float attn_18_1 = (float)(__nv_bfloat16)v[10];
                    float _exp2_137 = approx_exp2((-_vec_load_11[2]) * 1.442695f);
                    float sigmoid_19_1 = approx_rcp(1.0f + _exp2_137);
                    gated[2] = attn_18_1 * sigmoid_19_1;
                    float attn_20_1 = (float)(__nv_bfloat16)v[11];
                    float _exp2_138 = approx_exp2((-_vec_load_11[3]) * 1.442695f);
                    float sigmoid_21_1 = approx_rcp(1.0f + _exp2_138);
                    gated[3] = attn_20_1 * sigmoid_21_1;
                    float attn_22_1 = (float)(__nv_bfloat16)v[12];
                    float _exp2_139 = approx_exp2((-_vec_load_11[4]) * 1.442695f);
                    float sigmoid_23_1 = approx_rcp(1.0f + _exp2_139);
                    gated[4] = attn_22_1 * sigmoid_23_1;
                    float attn_24_1 = (float)(__nv_bfloat16)v[13];
                    float _exp2_140 = approx_exp2((-_vec_load_11[5]) * 1.442695f);
                    float sigmoid_25_1 = approx_rcp(1.0f + _exp2_140);
                    gated[5] = attn_24_1 * sigmoid_25_1;
                    float attn_26_1 = (float)(__nv_bfloat16)v[14];
                    float _exp2_141 = approx_exp2((-_vec_load_11[6]) * 1.442695f);
                    float sigmoid_27_1 = approx_rcp(1.0f + _exp2_141);
                    gated[6] = attn_26_1 * sigmoid_27_1;
                    float attn_28_1 = (float)(__nv_bfloat16)v[15];
                    float _exp2_142 = approx_exp2((-_vec_load_11[7]) * 1.442695f);
                    float sigmoid_29_1 = approx_rcp(1.0f + _exp2_142);
                    gated[7] = attn_28_1 * sigmoid_29_1;
                    unsigned int _packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 0], gated[_lp*2+1 + 0]));
                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 2], gated[_lp*2+1 + 2]));
                        _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 4], gated[_lp*2+1 + 4]));
                        _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gated[_lp*2 + 6], gated[_lp*2+1 + 6]));
                        _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(out + (out_off_1 + 8))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                }
            } else {
                long long p_base = ((long long)split * (long long)num_rows + row_i64) * (long long)num_qo_heads + (long long)first_head;
                long long p_off = p_base * 256 + (long long)d0;
                bool in_group = group_size > 0;
                float lo = ((l_sum[0] > 0.0f) ? o_acc[0] * inv_h[0] : 0.0f);
                float hi = ((l_sum[0] > 0.0f) ? o_acc[16] * inv_h[0] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + p_off)), "f"(lo), "r"((unsigned int)(in_group)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 128))), "f"(hi), "r"((unsigned int)(in_group)) : "memory");
                bool in_group_0 = group_size > 1;
                float lo_1 = ((l_sum[1] > 0.0f) ? o_acc[1] * inv_h[1] : 0.0f);
                float hi_2 = ((l_sum[1] > 0.0f) ? o_acc[17] * inv_h[1] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 256))), "f"(lo_1), "r"((unsigned int)(in_group_0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 256 + 128))), "f"(hi_2), "r"((unsigned int)(in_group_0)) : "memory");
                bool in_group_3 = group_size > 2;
                float lo_4 = ((l_sum[2] > 0.0f) ? o_acc[2] * inv_h[2] : 0.0f);
                float hi_5 = ((l_sum[2] > 0.0f) ? o_acc[18] * inv_h[2] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 512))), "f"(lo_4), "r"((unsigned int)(in_group_3)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 512 + 128))), "f"(hi_5), "r"((unsigned int)(in_group_3)) : "memory");
                bool in_group_6 = group_size > 3;
                float lo_7 = ((l_sum[3] > 0.0f) ? o_acc[3] * inv_h[3] : 0.0f);
                float hi_8 = ((l_sum[3] > 0.0f) ? o_acc[19] * inv_h[3] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 768))), "f"(lo_7), "r"((unsigned int)(in_group_6)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 768 + 128))), "f"(hi_8), "r"((unsigned int)(in_group_6)) : "memory");
                bool in_group_9 = group_size > 4;
                float lo_10 = ((l_sum[4] > 0.0f) ? o_acc[4] * inv_h[4] : 0.0f);
                float hi_11 = ((l_sum[4] > 0.0f) ? o_acc[20] * inv_h[4] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1024))), "f"(lo_10), "r"((unsigned int)(in_group_9)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1024 + 128))), "f"(hi_11), "r"((unsigned int)(in_group_9)) : "memory");
                bool in_group_12 = group_size > 5;
                float lo_13 = ((l_sum[5] > 0.0f) ? o_acc[5] * inv_h[5] : 0.0f);
                float hi_14 = ((l_sum[5] > 0.0f) ? o_acc[21] * inv_h[5] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1280))), "f"(lo_13), "r"((unsigned int)(in_group_12)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1280 + 128))), "f"(hi_14), "r"((unsigned int)(in_group_12)) : "memory");
                bool in_group_15 = group_size > 6;
                float lo_16 = ((l_sum[6] > 0.0f) ? o_acc[6] * inv_h[6] : 0.0f);
                float hi_17 = ((l_sum[6] > 0.0f) ? o_acc[22] * inv_h[6] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1536))), "f"(lo_16), "r"((unsigned int)(in_group_15)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1536 + 128))), "f"(hi_17), "r"((unsigned int)(in_group_15)) : "memory");
                bool in_group_18 = group_size > 7;
                float lo_19 = ((l_sum[7] > 0.0f) ? o_acc[7] * inv_h[7] : 0.0f);
                float hi_20 = ((l_sum[7] > 0.0f) ? o_acc[23] * inv_h[7] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1792))), "f"(lo_19), "r"((unsigned int)(in_group_18)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 1792 + 128))), "f"(hi_20), "r"((unsigned int)(in_group_18)) : "memory");
                bool in_group_21 = group_size > 8;
                float lo_22 = ((l_sum[8] > 0.0f) ? o_acc[8] * inv_h[8] : 0.0f);
                float hi_23 = ((l_sum[8] > 0.0f) ? o_acc[24] * inv_h[8] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2048))), "f"(lo_22), "r"((unsigned int)(in_group_21)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2048 + 128))), "f"(hi_23), "r"((unsigned int)(in_group_21)) : "memory");
                bool in_group_24 = group_size > 9;
                float lo_25 = ((l_sum[9] > 0.0f) ? o_acc[9] * inv_h[9] : 0.0f);
                float hi_26 = ((l_sum[9] > 0.0f) ? o_acc[25] * inv_h[9] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2304))), "f"(lo_25), "r"((unsigned int)(in_group_24)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2304 + 128))), "f"(hi_26), "r"((unsigned int)(in_group_24)) : "memory");
                bool in_group_27 = group_size > 10;
                float lo_28 = ((l_sum[10] > 0.0f) ? o_acc[10] * inv_h[10] : 0.0f);
                float hi_29 = ((l_sum[10] > 0.0f) ? o_acc[26] * inv_h[10] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2560))), "f"(lo_28), "r"((unsigned int)(in_group_27)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2560 + 128))), "f"(hi_29), "r"((unsigned int)(in_group_27)) : "memory");
                bool in_group_30 = group_size > 11;
                float lo_31 = ((l_sum[11] > 0.0f) ? o_acc[11] * inv_h[11] : 0.0f);
                float hi_32 = ((l_sum[11] > 0.0f) ? o_acc[27] * inv_h[11] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2816))), "f"(lo_31), "r"((unsigned int)(in_group_30)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 2816 + 128))), "f"(hi_32), "r"((unsigned int)(in_group_30)) : "memory");
                bool in_group_33 = group_size > 12;
                float lo_34 = ((l_sum[12] > 0.0f) ? o_acc[12] * inv_h[12] : 0.0f);
                float hi_35 = ((l_sum[12] > 0.0f) ? o_acc[28] * inv_h[12] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3072))), "f"(lo_34), "r"((unsigned int)(in_group_33)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3072 + 128))), "f"(hi_35), "r"((unsigned int)(in_group_33)) : "memory");
                bool in_group_36 = group_size > 13;
                float lo_37 = ((l_sum[13] > 0.0f) ? o_acc[13] * inv_h[13] : 0.0f);
                float hi_38 = ((l_sum[13] > 0.0f) ? o_acc[29] * inv_h[13] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3328))), "f"(lo_37), "r"((unsigned int)(in_group_36)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3328 + 128))), "f"(hi_38), "r"((unsigned int)(in_group_36)) : "memory");
                bool in_group_39 = group_size > 14;
                float lo_40 = ((l_sum[14] > 0.0f) ? o_acc[14] * inv_h[14] : 0.0f);
                float hi_41 = ((l_sum[14] > 0.0f) ? o_acc[30] * inv_h[14] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3584))), "f"(lo_40), "r"((unsigned int)(in_group_39)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3584 + 128))), "f"(hi_41), "r"((unsigned int)(in_group_39)) : "memory");
                bool in_group_42 = group_size > 15;
                float lo_43 = ((l_sum[15] > 0.0f) ? o_acc[15] * inv_h[15] : 0.0f);
                float hi_44 = ((l_sum[15] > 0.0f) ? o_acc[31] * inv_h[15] : 0.0f);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3840))), "f"(lo_43), "r"((unsigned int)(in_group_42)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (p_off + 3840 + 128))), "f"(hi_44), "r"((unsigned int)(in_group_42)) : "memory");
                if (tid_0 < 16) {
                    if (tid_0 < group_size) {
                        float l_t = xchg[64 + tid_0];
                        l_t = l_t + xchg[80 + tid_0];
                        l_t = l_t + xchg[96 + tid_0];
                        l_t = l_t + xchg[112 + tid_0];
                        float m_t = xchg[128 + tid_0];
                        float _log2_143;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_143) : "f"(l_t));
                        *(reinterpret_cast<float*>(partial_lse + (p_base + (long long)tid_0)) + (0)) = ((l_t > 0.0f) ? m_t + _log2_143 : -CUDART_INF_F);
                    }
                }
            }
            if ((unsigned long long)merge_counters != 0) {
                if (num_splits > 1) {
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (tid_0 == 0) {
                        int cidx = row * gridDim.y + kv_head;
                        unsigned int flip = 2147483648;
                        unsigned int nb = 1;
                        if (split == 0) {
                            nb = flip - (unsigned int)(num_splits - 1);
                        }
                        unsigned int old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(old) : "l"(&merge_counters[cidx]), "r"(static_cast<uint32_t>(nb)) : "memory");
                        unsigned int target = (old ^ flip) & flip;
                        {
                            unsigned int _poll_v_144;
                            unsigned long long _poll_t0_144;
                            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_poll_t0_144) :: "memory");
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_poll_v_144) : "l"((reinterpret_cast<unsigned int*>(merge_counters) + (cidx))) : "memory");
                            while (((_poll_v_144 ^ static_cast<unsigned int>(target)) & static_cast<unsigned int>(2147483648)) != 0) {
                                unsigned long long _poll_now_144;
                                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_poll_now_144) :: "memory");
                                if (_poll_now_144 - _poll_t0_144 > 4000000000ull) {
                                    asm volatile("trap;" ::: "memory");
                                }
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_poll_v_144) : "l"((reinterpret_cast<unsigned int*>(merge_counters) + (cidx))) : "memory");
                            }
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int pairs_total = group_size * 128;
                    int chunk = (pairs_total + num_splits - 1) / num_splits;
                    int begin = split * chunk;
                    int end = ((begin + chunk) < (pairs_total) ? (begin + chunk) : (pairs_total));
                    int p_first = begin + tid_0;
                    long long split_pitch = (long long)num_rows * (long long)num_qo_heads;
                    long long row_base = row_i64 * (long long)num_qo_heads + (long long)first_head;
                    int head_first = begin >> 7;
                    int n_heads = 0;
                    if (end > begin) {
                        n_heads = (end - 1 >> 7) - head_first + 1;
                    }
                    if (num_splits > 36) {
                        int n_lse = n_heads * num_splits;
                        #pragma unroll 1
                        for (int i = tid_0; i < n_lse; i += 128) {
                            int h = i / num_splits;
                            int s_0 = i - h * num_splits;
                            lse_tab[i] = partial_lse[(long long)s_0 * split_pitch + row_base + (long long)(head_first + h)];
                        }
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        #pragma unroll 1
                        for (int h_1 = warp_1; h_1 < n_heads; h_1 += 4) {
                            float hmax = -CUDART_INF_F;
                            #pragma unroll 1
                            for (int s_1 = lane_1; s_1 < num_splits; s_1 += 32) {
                                hmax = fmaxf(hmax, lse_tab[h_1 * num_splits + s_1]);
                            }
                            float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, hmax, 16);
                            hmax = fmaxf(hmax, _shfl_xor_sync_145);
                            float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, hmax, 8);
                            hmax = fmaxf(hmax, _shfl_xor_sync_146);
                            float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, hmax, 4);
                            hmax = fmaxf(hmax, _shfl_xor_sync_147);
                            float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, hmax, 2);
                            hmax = fmaxf(hmax, _shfl_xor_sync_148);
                            float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, hmax, 1);
                            hmax = fmaxf(hmax, _shfl_xor_sync_149);
                            if (lane_1 == 0) {
                                max_tab[h_1] = hmax;
                            }
                        }
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        float lse[16];
                        float p0[16];
                        float p1[16];
                        float acc0[4];
                        float acc1[4];
                        float den[4];
                        int chunk_w = (num_splits + 3) / 4;
                        if (num_splits <= 4) {
                            chunk_w = num_splits;
                        }
                        float lse_max[1];
                        int valid[1];
                        int tab[1];
                        long long dim[1];
                        long long qo_head_2[1];
                        long long row_head[1];
                        float gate0[1];
                        float gate1[1];
                        unsigned int word[1];
                        long long nqo = (long long)num_qo_heads;
                        #pragma unroll 1
                        for (int p_1 = p_first; p_1 < end; p_1 += 128) {
                            int pk = ((p_1) < (end - 1) ? (p_1) : (end - 1));
                            valid[0] = ((end > p_1) ? 1 : 0);
                            int head_2 = pk >> 7;
                            dim[0] = (long long)((pk & 127) * 2);
                            qo_head_2[0] = (long long)(first_head + head_2);
                            row_head[0] = row_i64 * nqo + qo_head_2[0];
                            acc0[0] = 0.0f;
                            acc1[0] = 0.0f;
                            den[0] = 0.0f;
                            acc0[1] = 0.0f;
                            acc1[1] = 0.0f;
                            den[1] = 0.0f;
                            acc0[2] = 0.0f;
                            acc1[2] = 0.0f;
                            den[2] = 0.0f;
                            acc0[3] = 0.0f;
                            acc1[3] = 0.0f;
                            den[3] = 0.0f;
                            tab[0] = (head_2 - head_first) * num_splits;
                            lse_max[0] = max_tab[head_2 - head_first];
                            float _vec_load_12[2];
                            {
                                uint32_t _bf16x2_bits_150;
                                _bf16x2_bits_150 = *reinterpret_cast<const uint32_t*>(gate + (row_head[0] * 256 + dim[0]) + 0);
                                (&_vec_load_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_150) << 16);
                                (&_vec_load_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_150) & 0xffff0000u);
                            }
                            gate0[0] = _vec_load_12[0];
                            gate1[0] = _vec_load_12[1];
                            int c_end = ((chunk_w) < (num_splits) ? (chunk_w) : (num_splits));
                            int c_begin = ((0) < (c_end) ? (0) : (c_end));
                            #pragma unroll 1
                            for (int base = c_begin; base < c_end; base += 16) {
                                if (c_end > base) {
                                    long long part_row = (long long)base * split_pitch + row_head[0];
                                    float _vec_load_13[2];
                                    {
                                        float2 _v2_151 = *reinterpret_cast<const float2*>(partial_out + (part_row * 256 + dim[0]) + 0);
                                        _vec_load_13[0] = _v2_151.x;
                                        _vec_load_13[0 + 1] = _v2_151.y;
                                    }
                                    p0[0] = _vec_load_13[0];
                                    p1[0] = _vec_load_13[1];
                                }
                                if (c_end > base + 1) {
                                    long long part_row_1 = (long long)(base + 1) * split_pitch + row_head[0];
                                    float _vec_load_14[2];
                                    {
                                        float2 _v2_152 = *reinterpret_cast<const float2*>(partial_out + (part_row_1 * 256 + dim[0]) + 0);
                                        _vec_load_14[0] = _v2_152.x;
                                        _vec_load_14[0 + 1] = _v2_152.y;
                                    }
                                    p0[1] = _vec_load_14[0];
                                    p1[1] = _vec_load_14[1];
                                }
                                if (c_end > base + 2) {
                                    long long part_row_2 = (long long)(base + 2) * split_pitch + row_head[0];
                                    float _vec_load_15[2];
                                    {
                                        float2 _v2_153 = *reinterpret_cast<const float2*>(partial_out + (part_row_2 * 256 + dim[0]) + 0);
                                        _vec_load_15[0] = _v2_153.x;
                                        _vec_load_15[0 + 1] = _v2_153.y;
                                    }
                                    p0[2] = _vec_load_15[0];
                                    p1[2] = _vec_load_15[1];
                                }
                                if (c_end > base + 3) {
                                    long long part_row_3 = (long long)(base + 3) * split_pitch + row_head[0];
                                    float _vec_load_16[2];
                                    {
                                        float2 _v2_154 = *reinterpret_cast<const float2*>(partial_out + (part_row_3 * 256 + dim[0]) + 0);
                                        _vec_load_16[0] = _v2_154.x;
                                        _vec_load_16[0 + 1] = _v2_154.y;
                                    }
                                    p0[3] = _vec_load_16[0];
                                    p1[3] = _vec_load_16[1];
                                }
                                if (c_end > base + 4) {
                                    long long part_row_4 = (long long)(base + 4) * split_pitch + row_head[0];
                                    float _vec_load_17[2];
                                    {
                                        float2 _v2_155 = *reinterpret_cast<const float2*>(partial_out + (part_row_4 * 256 + dim[0]) + 0);
                                        _vec_load_17[0] = _v2_155.x;
                                        _vec_load_17[0 + 1] = _v2_155.y;
                                    }
                                    p0[4] = _vec_load_17[0];
                                    p1[4] = _vec_load_17[1];
                                }
                                if (c_end > base + 5) {
                                    long long part_row_5 = (long long)(base + 5) * split_pitch + row_head[0];
                                    float _vec_load_18[2];
                                    {
                                        float2 _v2_156 = *reinterpret_cast<const float2*>(partial_out + (part_row_5 * 256 + dim[0]) + 0);
                                        _vec_load_18[0] = _v2_156.x;
                                        _vec_load_18[0 + 1] = _v2_156.y;
                                    }
                                    p0[5] = _vec_load_18[0];
                                    p1[5] = _vec_load_18[1];
                                }
                                if (c_end > base + 6) {
                                    long long part_row_6 = (long long)(base + 6) * split_pitch + row_head[0];
                                    float _vec_load_19[2];
                                    {
                                        float2 _v2_157 = *reinterpret_cast<const float2*>(partial_out + (part_row_6 * 256 + dim[0]) + 0);
                                        _vec_load_19[0] = _v2_157.x;
                                        _vec_load_19[0 + 1] = _v2_157.y;
                                    }
                                    p0[6] = _vec_load_19[0];
                                    p1[6] = _vec_load_19[1];
                                }
                                if (c_end > base + 7) {
                                    long long part_row_7 = (long long)(base + 7) * split_pitch + row_head[0];
                                    float _vec_load_20[2];
                                    {
                                        float2 _v2_158 = *reinterpret_cast<const float2*>(partial_out + (part_row_7 * 256 + dim[0]) + 0);
                                        _vec_load_20[0] = _v2_158.x;
                                        _vec_load_20[0 + 1] = _v2_158.y;
                                    }
                                    p0[7] = _vec_load_20[0];
                                    p1[7] = _vec_load_20[1];
                                }
                                if (c_end > base + 8) {
                                    long long part_row_8 = (long long)(base + 8) * split_pitch + row_head[0];
                                    float _vec_load_21[2];
                                    {
                                        float2 _v2_159 = *reinterpret_cast<const float2*>(partial_out + (part_row_8 * 256 + dim[0]) + 0);
                                        _vec_load_21[0] = _v2_159.x;
                                        _vec_load_21[0 + 1] = _v2_159.y;
                                    }
                                    p0[8] = _vec_load_21[0];
                                    p1[8] = _vec_load_21[1];
                                }
                                if (c_end > base + 9) {
                                    long long part_row_9 = (long long)(base + 9) * split_pitch + row_head[0];
                                    float _vec_load_22[2];
                                    {
                                        float2 _v2_160 = *reinterpret_cast<const float2*>(partial_out + (part_row_9 * 256 + dim[0]) + 0);
                                        _vec_load_22[0] = _v2_160.x;
                                        _vec_load_22[0 + 1] = _v2_160.y;
                                    }
                                    p0[9] = _vec_load_22[0];
                                    p1[9] = _vec_load_22[1];
                                }
                                if (c_end > base + 10) {
                                    long long part_row_10 = (long long)(base + 10) * split_pitch + row_head[0];
                                    float _vec_load_23[2];
                                    {
                                        float2 _v2_161 = *reinterpret_cast<const float2*>(partial_out + (part_row_10 * 256 + dim[0]) + 0);
                                        _vec_load_23[0] = _v2_161.x;
                                        _vec_load_23[0 + 1] = _v2_161.y;
                                    }
                                    p0[10] = _vec_load_23[0];
                                    p1[10] = _vec_load_23[1];
                                }
                                if (c_end > base + 11) {
                                    long long part_row_11 = (long long)(base + 11) * split_pitch + row_head[0];
                                    float _vec_load_24[2];
                                    {
                                        float2 _v2_162 = *reinterpret_cast<const float2*>(partial_out + (part_row_11 * 256 + dim[0]) + 0);
                                        _vec_load_24[0] = _v2_162.x;
                                        _vec_load_24[0 + 1] = _v2_162.y;
                                    }
                                    p0[11] = _vec_load_24[0];
                                    p1[11] = _vec_load_24[1];
                                }
                                if (c_end > base + 12) {
                                    long long part_row_12 = (long long)(base + 12) * split_pitch + row_head[0];
                                    float _vec_load_25[2];
                                    {
                                        float2 _v2_163 = *reinterpret_cast<const float2*>(partial_out + (part_row_12 * 256 + dim[0]) + 0);
                                        _vec_load_25[0] = _v2_163.x;
                                        _vec_load_25[0 + 1] = _v2_163.y;
                                    }
                                    p0[12] = _vec_load_25[0];
                                    p1[12] = _vec_load_25[1];
                                }
                                if (c_end > base + 13) {
                                    long long part_row_13 = (long long)(base + 13) * split_pitch + row_head[0];
                                    float _vec_load_26[2];
                                    {
                                        float2 _v2_164 = *reinterpret_cast<const float2*>(partial_out + (part_row_13 * 256 + dim[0]) + 0);
                                        _vec_load_26[0] = _v2_164.x;
                                        _vec_load_26[0 + 1] = _v2_164.y;
                                    }
                                    p0[13] = _vec_load_26[0];
                                    p1[13] = _vec_load_26[1];
                                }
                                if (c_end > base + 14) {
                                    long long part_row_14 = (long long)(base + 14) * split_pitch + row_head[0];
                                    float _vec_load_27[2];
                                    {
                                        float2 _v2_165 = *reinterpret_cast<const float2*>(partial_out + (part_row_14 * 256 + dim[0]) + 0);
                                        _vec_load_27[0] = _v2_165.x;
                                        _vec_load_27[0 + 1] = _v2_165.y;
                                    }
                                    p0[14] = _vec_load_27[0];
                                    p1[14] = _vec_load_27[1];
                                }
                                if (c_end > base + 15) {
                                    long long part_row_15 = (long long)(base + 15) * split_pitch + row_head[0];
                                    float _vec_load_28[2];
                                    {
                                        float2 _v2_166 = *reinterpret_cast<const float2*>(partial_out + (part_row_15 * 256 + dim[0]) + 0);
                                        _vec_load_28[0] = _v2_166.x;
                                        _vec_load_28[0 + 1] = _v2_166.y;
                                    }
                                    p0[15] = _vec_load_28[0];
                                    p1[15] = _vec_load_28[1];
                                }
                                bool live = c_end > base && lse_max[0] > -CUDART_INF_F;
                                if (live) {
                                    float weight = approx_exp2(lse_tab[tab[0] + base] - lse_max[0]);
                                    den[0] = den[0] + weight;
                                    acc0[0] = __fmaf_rn(weight, p0[0], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight, p1[0], acc1[0]);
                                }
                                bool live_0 = c_end > base + 1 && lse_max[0] > -CUDART_INF_F;
                                if (live_0) {
                                    float weight_1 = approx_exp2(lse_tab[tab[0] + base + 1] - lse_max[0]);
                                    den[0] = den[0] + weight_1;
                                    acc0[0] = __fmaf_rn(weight_1, p0[1], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_1, p1[1], acc1[0]);
                                }
                                bool live_1 = c_end > base + 2 && lse_max[0] > -CUDART_INF_F;
                                if (live_1) {
                                    float weight_2 = approx_exp2(lse_tab[tab[0] + base + 2] - lse_max[0]);
                                    den[0] = den[0] + weight_2;
                                    acc0[0] = __fmaf_rn(weight_2, p0[2], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_2, p1[2], acc1[0]);
                                }
                                bool live_2 = c_end > base + 3 && lse_max[0] > -CUDART_INF_F;
                                if (live_2) {
                                    float weight_3 = approx_exp2(lse_tab[tab[0] + base + 3] - lse_max[0]);
                                    den[0] = den[0] + weight_3;
                                    acc0[0] = __fmaf_rn(weight_3, p0[3], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_3, p1[3], acc1[0]);
                                }
                                bool live_3 = c_end > base + 4 && lse_max[0] > -CUDART_INF_F;
                                if (live_3) {
                                    float weight_4 = approx_exp2(lse_tab[tab[0] + base + 4] - lse_max[0]);
                                    den[0] = den[0] + weight_4;
                                    acc0[0] = __fmaf_rn(weight_4, p0[4], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_4, p1[4], acc1[0]);
                                }
                                bool live_4 = c_end > base + 5 && lse_max[0] > -CUDART_INF_F;
                                if (live_4) {
                                    float weight_5 = approx_exp2(lse_tab[tab[0] + base + 5] - lse_max[0]);
                                    den[0] = den[0] + weight_5;
                                    acc0[0] = __fmaf_rn(weight_5, p0[5], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_5, p1[5], acc1[0]);
                                }
                                bool live_5 = c_end > base + 6 && lse_max[0] > -CUDART_INF_F;
                                if (live_5) {
                                    float weight_6 = approx_exp2(lse_tab[tab[0] + base + 6] - lse_max[0]);
                                    den[0] = den[0] + weight_6;
                                    acc0[0] = __fmaf_rn(weight_6, p0[6], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_6, p1[6], acc1[0]);
                                }
                                bool live_6 = c_end > base + 7 && lse_max[0] > -CUDART_INF_F;
                                if (live_6) {
                                    float weight_7 = approx_exp2(lse_tab[tab[0] + base + 7] - lse_max[0]);
                                    den[0] = den[0] + weight_7;
                                    acc0[0] = __fmaf_rn(weight_7, p0[7], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_7, p1[7], acc1[0]);
                                }
                                bool live_7 = c_end > base + 8 && lse_max[0] > -CUDART_INF_F;
                                if (live_7) {
                                    float weight_8 = approx_exp2(lse_tab[tab[0] + base + 8] - lse_max[0]);
                                    den[0] = den[0] + weight_8;
                                    acc0[0] = __fmaf_rn(weight_8, p0[8], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_8, p1[8], acc1[0]);
                                }
                                bool live_8 = c_end > base + 9 && lse_max[0] > -CUDART_INF_F;
                                if (live_8) {
                                    float weight_9 = approx_exp2(lse_tab[tab[0] + base + 9] - lse_max[0]);
                                    den[0] = den[0] + weight_9;
                                    acc0[0] = __fmaf_rn(weight_9, p0[9], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_9, p1[9], acc1[0]);
                                }
                                bool live_9 = c_end > base + 10 && lse_max[0] > -CUDART_INF_F;
                                if (live_9) {
                                    float weight_10 = approx_exp2(lse_tab[tab[0] + base + 10] - lse_max[0]);
                                    den[0] = den[0] + weight_10;
                                    acc0[0] = __fmaf_rn(weight_10, p0[10], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_10, p1[10], acc1[0]);
                                }
                                bool live_10 = c_end > base + 11 && lse_max[0] > -CUDART_INF_F;
                                if (live_10) {
                                    float weight_11 = approx_exp2(lse_tab[tab[0] + base + 11] - lse_max[0]);
                                    den[0] = den[0] + weight_11;
                                    acc0[0] = __fmaf_rn(weight_11, p0[11], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_11, p1[11], acc1[0]);
                                }
                                bool live_11 = c_end > base + 12 && lse_max[0] > -CUDART_INF_F;
                                if (live_11) {
                                    float weight_12 = approx_exp2(lse_tab[tab[0] + base + 12] - lse_max[0]);
                                    den[0] = den[0] + weight_12;
                                    acc0[0] = __fmaf_rn(weight_12, p0[12], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_12, p1[12], acc1[0]);
                                }
                                bool live_12 = c_end > base + 13 && lse_max[0] > -CUDART_INF_F;
                                if (live_12) {
                                    float weight_13 = approx_exp2(lse_tab[tab[0] + base + 13] - lse_max[0]);
                                    den[0] = den[0] + weight_13;
                                    acc0[0] = __fmaf_rn(weight_13, p0[13], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_13, p1[13], acc1[0]);
                                }
                                bool live_13 = c_end > base + 14 && lse_max[0] > -CUDART_INF_F;
                                if (live_13) {
                                    float weight_14 = approx_exp2(lse_tab[tab[0] + base + 14] - lse_max[0]);
                                    den[0] = den[0] + weight_14;
                                    acc0[0] = __fmaf_rn(weight_14, p0[14], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_14, p1[14], acc1[0]);
                                }
                                bool live_14 = c_end > base + 15 && lse_max[0] > -CUDART_INF_F;
                                if (live_14) {
                                    float weight_15 = approx_exp2(lse_tab[tab[0] + base + 15] - lse_max[0]);
                                    den[0] = den[0] + weight_15;
                                    acc0[0] = __fmaf_rn(weight_15, p0[15], acc0[0]);
                                    acc1[0] = __fmaf_rn(weight_15, p1[15], acc1[0]);
                                }
                            }
                            int c_end_0 = ((chunk_w * 2) < (num_splits) ? (chunk_w * 2) : (num_splits));
                            int c_begin_1 = ((chunk_w) < (c_end_0) ? (chunk_w) : (c_end_0));
                            #pragma unroll 1
                            for (int base_1 = c_begin_1; base_1 < c_end_0; base_1 += 16) {
                                if (c_end_0 > base_1) {
                                    long long part_row_16 = (long long)base_1 * split_pitch + row_head[0];
                                    float _vec_load_29[2];
                                    {
                                        float2 _v2_167 = *reinterpret_cast<const float2*>(partial_out + (part_row_16 * 256 + dim[0]) + 0);
                                        _vec_load_29[0] = _v2_167.x;
                                        _vec_load_29[0 + 1] = _v2_167.y;
                                    }
                                    p0[0] = _vec_load_29[0];
                                    p1[0] = _vec_load_29[1];
                                }
                                if (c_end_0 > base_1 + 1) {
                                    long long part_row_17 = (long long)(base_1 + 1) * split_pitch + row_head[0];
                                    float _vec_load_30[2];
                                    {
                                        float2 _v2_168 = *reinterpret_cast<const float2*>(partial_out + (part_row_17 * 256 + dim[0]) + 0);
                                        _vec_load_30[0] = _v2_168.x;
                                        _vec_load_30[0 + 1] = _v2_168.y;
                                    }
                                    p0[1] = _vec_load_30[0];
                                    p1[1] = _vec_load_30[1];
                                }
                                if (c_end_0 > base_1 + 2) {
                                    long long part_row_18 = (long long)(base_1 + 2) * split_pitch + row_head[0];
                                    float _vec_load_31[2];
                                    {
                                        float2 _v2_169 = *reinterpret_cast<const float2*>(partial_out + (part_row_18 * 256 + dim[0]) + 0);
                                        _vec_load_31[0] = _v2_169.x;
                                        _vec_load_31[0 + 1] = _v2_169.y;
                                    }
                                    p0[2] = _vec_load_31[0];
                                    p1[2] = _vec_load_31[1];
                                }
                                if (c_end_0 > base_1 + 3) {
                                    long long part_row_19 = (long long)(base_1 + 3) * split_pitch + row_head[0];
                                    float _vec_load_32[2];
                                    {
                                        float2 _v2_170 = *reinterpret_cast<const float2*>(partial_out + (part_row_19 * 256 + dim[0]) + 0);
                                        _vec_load_32[0] = _v2_170.x;
                                        _vec_load_32[0 + 1] = _v2_170.y;
                                    }
                                    p0[3] = _vec_load_32[0];
                                    p1[3] = _vec_load_32[1];
                                }
                                if (c_end_0 > base_1 + 4) {
                                    long long part_row_20 = (long long)(base_1 + 4) * split_pitch + row_head[0];
                                    float _vec_load_33[2];
                                    {
                                        float2 _v2_171 = *reinterpret_cast<const float2*>(partial_out + (part_row_20 * 256 + dim[0]) + 0);
                                        _vec_load_33[0] = _v2_171.x;
                                        _vec_load_33[0 + 1] = _v2_171.y;
                                    }
                                    p0[4] = _vec_load_33[0];
                                    p1[4] = _vec_load_33[1];
                                }
                                if (c_end_0 > base_1 + 5) {
                                    long long part_row_21 = (long long)(base_1 + 5) * split_pitch + row_head[0];
                                    float _vec_load_34[2];
                                    {
                                        float2 _v2_172 = *reinterpret_cast<const float2*>(partial_out + (part_row_21 * 256 + dim[0]) + 0);
                                        _vec_load_34[0] = _v2_172.x;
                                        _vec_load_34[0 + 1] = _v2_172.y;
                                    }
                                    p0[5] = _vec_load_34[0];
                                    p1[5] = _vec_load_34[1];
                                }
                                if (c_end_0 > base_1 + 6) {
                                    long long part_row_22 = (long long)(base_1 + 6) * split_pitch + row_head[0];
                                    float _vec_load_35[2];
                                    {
                                        float2 _v2_173 = *reinterpret_cast<const float2*>(partial_out + (part_row_22 * 256 + dim[0]) + 0);
                                        _vec_load_35[0] = _v2_173.x;
                                        _vec_load_35[0 + 1] = _v2_173.y;
                                    }
                                    p0[6] = _vec_load_35[0];
                                    p1[6] = _vec_load_35[1];
                                }
                                if (c_end_0 > base_1 + 7) {
                                    long long part_row_23 = (long long)(base_1 + 7) * split_pitch + row_head[0];
                                    float _vec_load_36[2];
                                    {
                                        float2 _v2_174 = *reinterpret_cast<const float2*>(partial_out + (part_row_23 * 256 + dim[0]) + 0);
                                        _vec_load_36[0] = _v2_174.x;
                                        _vec_load_36[0 + 1] = _v2_174.y;
                                    }
                                    p0[7] = _vec_load_36[0];
                                    p1[7] = _vec_load_36[1];
                                }
                                if (c_end_0 > base_1 + 8) {
                                    long long part_row_24 = (long long)(base_1 + 8) * split_pitch + row_head[0];
                                    float _vec_load_37[2];
                                    {
                                        float2 _v2_175 = *reinterpret_cast<const float2*>(partial_out + (part_row_24 * 256 + dim[0]) + 0);
                                        _vec_load_37[0] = _v2_175.x;
                                        _vec_load_37[0 + 1] = _v2_175.y;
                                    }
                                    p0[8] = _vec_load_37[0];
                                    p1[8] = _vec_load_37[1];
                                }
                                if (c_end_0 > base_1 + 9) {
                                    long long part_row_25 = (long long)(base_1 + 9) * split_pitch + row_head[0];
                                    float _vec_load_38[2];
                                    {
                                        float2 _v2_176 = *reinterpret_cast<const float2*>(partial_out + (part_row_25 * 256 + dim[0]) + 0);
                                        _vec_load_38[0] = _v2_176.x;
                                        _vec_load_38[0 + 1] = _v2_176.y;
                                    }
                                    p0[9] = _vec_load_38[0];
                                    p1[9] = _vec_load_38[1];
                                }
                                if (c_end_0 > base_1 + 10) {
                                    long long part_row_26 = (long long)(base_1 + 10) * split_pitch + row_head[0];
                                    float _vec_load_39[2];
                                    {
                                        float2 _v2_177 = *reinterpret_cast<const float2*>(partial_out + (part_row_26 * 256 + dim[0]) + 0);
                                        _vec_load_39[0] = _v2_177.x;
                                        _vec_load_39[0 + 1] = _v2_177.y;
                                    }
                                    p0[10] = _vec_load_39[0];
                                    p1[10] = _vec_load_39[1];
                                }
                                if (c_end_0 > base_1 + 11) {
                                    long long part_row_27 = (long long)(base_1 + 11) * split_pitch + row_head[0];
                                    float _vec_load_40[2];
                                    {
                                        float2 _v2_178 = *reinterpret_cast<const float2*>(partial_out + (part_row_27 * 256 + dim[0]) + 0);
                                        _vec_load_40[0] = _v2_178.x;
                                        _vec_load_40[0 + 1] = _v2_178.y;
                                    }
                                    p0[11] = _vec_load_40[0];
                                    p1[11] = _vec_load_40[1];
                                }
                                if (c_end_0 > base_1 + 12) {
                                    long long part_row_28 = (long long)(base_1 + 12) * split_pitch + row_head[0];
                                    float _vec_load_41[2];
                                    {
                                        float2 _v2_179 = *reinterpret_cast<const float2*>(partial_out + (part_row_28 * 256 + dim[0]) + 0);
                                        _vec_load_41[0] = _v2_179.x;
                                        _vec_load_41[0 + 1] = _v2_179.y;
                                    }
                                    p0[12] = _vec_load_41[0];
                                    p1[12] = _vec_load_41[1];
                                }
                                if (c_end_0 > base_1 + 13) {
                                    long long part_row_29 = (long long)(base_1 + 13) * split_pitch + row_head[0];
                                    float _vec_load_42[2];
                                    {
                                        float2 _v2_180 = *reinterpret_cast<const float2*>(partial_out + (part_row_29 * 256 + dim[0]) + 0);
                                        _vec_load_42[0] = _v2_180.x;
                                        _vec_load_42[0 + 1] = _v2_180.y;
                                    }
                                    p0[13] = _vec_load_42[0];
                                    p1[13] = _vec_load_42[1];
                                }
                                if (c_end_0 > base_1 + 14) {
                                    long long part_row_30 = (long long)(base_1 + 14) * split_pitch + row_head[0];
                                    float _vec_load_43[2];
                                    {
                                        float2 _v2_181 = *reinterpret_cast<const float2*>(partial_out + (part_row_30 * 256 + dim[0]) + 0);
                                        _vec_load_43[0] = _v2_181.x;
                                        _vec_load_43[0 + 1] = _v2_181.y;
                                    }
                                    p0[14] = _vec_load_43[0];
                                    p1[14] = _vec_load_43[1];
                                }
                                if (c_end_0 > base_1 + 15) {
                                    long long part_row_31 = (long long)(base_1 + 15) * split_pitch + row_head[0];
                                    float _vec_load_44[2];
                                    {
                                        float2 _v2_182 = *reinterpret_cast<const float2*>(partial_out + (part_row_31 * 256 + dim[0]) + 0);
                                        _vec_load_44[0] = _v2_182.x;
                                        _vec_load_44[0 + 1] = _v2_182.y;
                                    }
                                    p0[15] = _vec_load_44[0];
                                    p1[15] = _vec_load_44[1];
                                }
                                bool live_15 = c_end_0 > base_1 && lse_max[0] > -CUDART_INF_F;
                                if (live_15) {
                                    float weight_16 = approx_exp2(lse_tab[tab[0] + base_1] - lse_max[0]);
                                    den[1] = den[1] + weight_16;
                                    acc0[1] = __fmaf_rn(weight_16, p0[0], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_16, p1[0], acc1[1]);
                                }
                                bool live_0_1 = c_end_0 > base_1 + 1 && lse_max[0] > -CUDART_INF_F;
                                if (live_0_1) {
                                    float weight_17 = approx_exp2(lse_tab[tab[0] + base_1 + 1] - lse_max[0]);
                                    den[1] = den[1] + weight_17;
                                    acc0[1] = __fmaf_rn(weight_17, p0[1], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_17, p1[1], acc1[1]);
                                }
                                bool live_1_1 = c_end_0 > base_1 + 2 && lse_max[0] > -CUDART_INF_F;
                                if (live_1_1) {
                                    float weight_18 = approx_exp2(lse_tab[tab[0] + base_1 + 2] - lse_max[0]);
                                    den[1] = den[1] + weight_18;
                                    acc0[1] = __fmaf_rn(weight_18, p0[2], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_18, p1[2], acc1[1]);
                                }
                                bool live_2_1 = c_end_0 > base_1 + 3 && lse_max[0] > -CUDART_INF_F;
                                if (live_2_1) {
                                    float weight_19 = approx_exp2(lse_tab[tab[0] + base_1 + 3] - lse_max[0]);
                                    den[1] = den[1] + weight_19;
                                    acc0[1] = __fmaf_rn(weight_19, p0[3], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_19, p1[3], acc1[1]);
                                }
                                bool live_3_1 = c_end_0 > base_1 + 4 && lse_max[0] > -CUDART_INF_F;
                                if (live_3_1) {
                                    float weight_20 = approx_exp2(lse_tab[tab[0] + base_1 + 4] - lse_max[0]);
                                    den[1] = den[1] + weight_20;
                                    acc0[1] = __fmaf_rn(weight_20, p0[4], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_20, p1[4], acc1[1]);
                                }
                                bool live_4_1 = c_end_0 > base_1 + 5 && lse_max[0] > -CUDART_INF_F;
                                if (live_4_1) {
                                    float weight_21 = approx_exp2(lse_tab[tab[0] + base_1 + 5] - lse_max[0]);
                                    den[1] = den[1] + weight_21;
                                    acc0[1] = __fmaf_rn(weight_21, p0[5], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_21, p1[5], acc1[1]);
                                }
                                bool live_5_1 = c_end_0 > base_1 + 6 && lse_max[0] > -CUDART_INF_F;
                                if (live_5_1) {
                                    float weight_22 = approx_exp2(lse_tab[tab[0] + base_1 + 6] - lse_max[0]);
                                    den[1] = den[1] + weight_22;
                                    acc0[1] = __fmaf_rn(weight_22, p0[6], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_22, p1[6], acc1[1]);
                                }
                                bool live_6_1 = c_end_0 > base_1 + 7 && lse_max[0] > -CUDART_INF_F;
                                if (live_6_1) {
                                    float weight_23 = approx_exp2(lse_tab[tab[0] + base_1 + 7] - lse_max[0]);
                                    den[1] = den[1] + weight_23;
                                    acc0[1] = __fmaf_rn(weight_23, p0[7], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_23, p1[7], acc1[1]);
                                }
                                bool live_7_1 = c_end_0 > base_1 + 8 && lse_max[0] > -CUDART_INF_F;
                                if (live_7_1) {
                                    float weight_24 = approx_exp2(lse_tab[tab[0] + base_1 + 8] - lse_max[0]);
                                    den[1] = den[1] + weight_24;
                                    acc0[1] = __fmaf_rn(weight_24, p0[8], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_24, p1[8], acc1[1]);
                                }
                                bool live_8_1 = c_end_0 > base_1 + 9 && lse_max[0] > -CUDART_INF_F;
                                if (live_8_1) {
                                    float weight_25 = approx_exp2(lse_tab[tab[0] + base_1 + 9] - lse_max[0]);
                                    den[1] = den[1] + weight_25;
                                    acc0[1] = __fmaf_rn(weight_25, p0[9], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_25, p1[9], acc1[1]);
                                }
                                bool live_9_1 = c_end_0 > base_1 + 10 && lse_max[0] > -CUDART_INF_F;
                                if (live_9_1) {
                                    float weight_26 = approx_exp2(lse_tab[tab[0] + base_1 + 10] - lse_max[0]);
                                    den[1] = den[1] + weight_26;
                                    acc0[1] = __fmaf_rn(weight_26, p0[10], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_26, p1[10], acc1[1]);
                                }
                                bool live_10_1 = c_end_0 > base_1 + 11 && lse_max[0] > -CUDART_INF_F;
                                if (live_10_1) {
                                    float weight_27 = approx_exp2(lse_tab[tab[0] + base_1 + 11] - lse_max[0]);
                                    den[1] = den[1] + weight_27;
                                    acc0[1] = __fmaf_rn(weight_27, p0[11], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_27, p1[11], acc1[1]);
                                }
                                bool live_11_1 = c_end_0 > base_1 + 12 && lse_max[0] > -CUDART_INF_F;
                                if (live_11_1) {
                                    float weight_28 = approx_exp2(lse_tab[tab[0] + base_1 + 12] - lse_max[0]);
                                    den[1] = den[1] + weight_28;
                                    acc0[1] = __fmaf_rn(weight_28, p0[12], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_28, p1[12], acc1[1]);
                                }
                                bool live_12_1 = c_end_0 > base_1 + 13 && lse_max[0] > -CUDART_INF_F;
                                if (live_12_1) {
                                    float weight_29 = approx_exp2(lse_tab[tab[0] + base_1 + 13] - lse_max[0]);
                                    den[1] = den[1] + weight_29;
                                    acc0[1] = __fmaf_rn(weight_29, p0[13], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_29, p1[13], acc1[1]);
                                }
                                bool live_13_1 = c_end_0 > base_1 + 14 && lse_max[0] > -CUDART_INF_F;
                                if (live_13_1) {
                                    float weight_30 = approx_exp2(lse_tab[tab[0] + base_1 + 14] - lse_max[0]);
                                    den[1] = den[1] + weight_30;
                                    acc0[1] = __fmaf_rn(weight_30, p0[14], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_30, p1[14], acc1[1]);
                                }
                                bool live_14_1 = c_end_0 > base_1 + 15 && lse_max[0] > -CUDART_INF_F;
                                if (live_14_1) {
                                    float weight_31 = approx_exp2(lse_tab[tab[0] + base_1 + 15] - lse_max[0]);
                                    den[1] = den[1] + weight_31;
                                    acc0[1] = __fmaf_rn(weight_31, p0[15], acc0[1]);
                                    acc1[1] = __fmaf_rn(weight_31, p1[15], acc1[1]);
                                }
                            }
                            int c_end_2 = ((chunk_w * 3) < (num_splits) ? (chunk_w * 3) : (num_splits));
                            int c_begin_3 = ((chunk_w * 2) < (c_end_2) ? (chunk_w * 2) : (c_end_2));
                            #pragma unroll 1
                            for (int base_2 = c_begin_3; base_2 < c_end_2; base_2 += 16) {
                                if (c_end_2 > base_2) {
                                    long long part_row_32 = (long long)base_2 * split_pitch + row_head[0];
                                    float _vec_load_45[2];
                                    {
                                        float2 _v2_183 = *reinterpret_cast<const float2*>(partial_out + (part_row_32 * 256 + dim[0]) + 0);
                                        _vec_load_45[0] = _v2_183.x;
                                        _vec_load_45[0 + 1] = _v2_183.y;
                                    }
                                    p0[0] = _vec_load_45[0];
                                    p1[0] = _vec_load_45[1];
                                }
                                if (c_end_2 > base_2 + 1) {
                                    long long part_row_33 = (long long)(base_2 + 1) * split_pitch + row_head[0];
                                    float _vec_load_46[2];
                                    {
                                        float2 _v2_184 = *reinterpret_cast<const float2*>(partial_out + (part_row_33 * 256 + dim[0]) + 0);
                                        _vec_load_46[0] = _v2_184.x;
                                        _vec_load_46[0 + 1] = _v2_184.y;
                                    }
                                    p0[1] = _vec_load_46[0];
                                    p1[1] = _vec_load_46[1];
                                }
                                if (c_end_2 > base_2 + 2) {
                                    long long part_row_34 = (long long)(base_2 + 2) * split_pitch + row_head[0];
                                    float _vec_load_47[2];
                                    {
                                        float2 _v2_185 = *reinterpret_cast<const float2*>(partial_out + (part_row_34 * 256 + dim[0]) + 0);
                                        _vec_load_47[0] = _v2_185.x;
                                        _vec_load_47[0 + 1] = _v2_185.y;
                                    }
                                    p0[2] = _vec_load_47[0];
                                    p1[2] = _vec_load_47[1];
                                }
                                if (c_end_2 > base_2 + 3) {
                                    long long part_row_35 = (long long)(base_2 + 3) * split_pitch + row_head[0];
                                    float _vec_load_48[2];
                                    {
                                        float2 _v2_186 = *reinterpret_cast<const float2*>(partial_out + (part_row_35 * 256 + dim[0]) + 0);
                                        _vec_load_48[0] = _v2_186.x;
                                        _vec_load_48[0 + 1] = _v2_186.y;
                                    }
                                    p0[3] = _vec_load_48[0];
                                    p1[3] = _vec_load_48[1];
                                }
                                if (c_end_2 > base_2 + 4) {
                                    long long part_row_36 = (long long)(base_2 + 4) * split_pitch + row_head[0];
                                    float _vec_load_49[2];
                                    {
                                        float2 _v2_187 = *reinterpret_cast<const float2*>(partial_out + (part_row_36 * 256 + dim[0]) + 0);
                                        _vec_load_49[0] = _v2_187.x;
                                        _vec_load_49[0 + 1] = _v2_187.y;
                                    }
                                    p0[4] = _vec_load_49[0];
                                    p1[4] = _vec_load_49[1];
                                }
                                if (c_end_2 > base_2 + 5) {
                                    long long part_row_37 = (long long)(base_2 + 5) * split_pitch + row_head[0];
                                    float _vec_load_50[2];
                                    {
                                        float2 _v2_188 = *reinterpret_cast<const float2*>(partial_out + (part_row_37 * 256 + dim[0]) + 0);
                                        _vec_load_50[0] = _v2_188.x;
                                        _vec_load_50[0 + 1] = _v2_188.y;
                                    }
                                    p0[5] = _vec_load_50[0];
                                    p1[5] = _vec_load_50[1];
                                }
                                if (c_end_2 > base_2 + 6) {
                                    long long part_row_38 = (long long)(base_2 + 6) * split_pitch + row_head[0];
                                    float _vec_load_51[2];
                                    {
                                        float2 _v2_189 = *reinterpret_cast<const float2*>(partial_out + (part_row_38 * 256 + dim[0]) + 0);
                                        _vec_load_51[0] = _v2_189.x;
                                        _vec_load_51[0 + 1] = _v2_189.y;
                                    }
                                    p0[6] = _vec_load_51[0];
                                    p1[6] = _vec_load_51[1];
                                }
                                if (c_end_2 > base_2 + 7) {
                                    long long part_row_39 = (long long)(base_2 + 7) * split_pitch + row_head[0];
                                    float _vec_load_52[2];
                                    {
                                        float2 _v2_190 = *reinterpret_cast<const float2*>(partial_out + (part_row_39 * 256 + dim[0]) + 0);
                                        _vec_load_52[0] = _v2_190.x;
                                        _vec_load_52[0 + 1] = _v2_190.y;
                                    }
                                    p0[7] = _vec_load_52[0];
                                    p1[7] = _vec_load_52[1];
                                }
                                if (c_end_2 > base_2 + 8) {
                                    long long part_row_40 = (long long)(base_2 + 8) * split_pitch + row_head[0];
                                    float _vec_load_53[2];
                                    {
                                        float2 _v2_191 = *reinterpret_cast<const float2*>(partial_out + (part_row_40 * 256 + dim[0]) + 0);
                                        _vec_load_53[0] = _v2_191.x;
                                        _vec_load_53[0 + 1] = _v2_191.y;
                                    }
                                    p0[8] = _vec_load_53[0];
                                    p1[8] = _vec_load_53[1];
                                }
                                if (c_end_2 > base_2 + 9) {
                                    long long part_row_41 = (long long)(base_2 + 9) * split_pitch + row_head[0];
                                    float _vec_load_54[2];
                                    {
                                        float2 _v2_192 = *reinterpret_cast<const float2*>(partial_out + (part_row_41 * 256 + dim[0]) + 0);
                                        _vec_load_54[0] = _v2_192.x;
                                        _vec_load_54[0 + 1] = _v2_192.y;
                                    }
                                    p0[9] = _vec_load_54[0];
                                    p1[9] = _vec_load_54[1];
                                }
                                if (c_end_2 > base_2 + 10) {
                                    long long part_row_42 = (long long)(base_2 + 10) * split_pitch + row_head[0];
                                    float _vec_load_55[2];
                                    {
                                        float2 _v2_193 = *reinterpret_cast<const float2*>(partial_out + (part_row_42 * 256 + dim[0]) + 0);
                                        _vec_load_55[0] = _v2_193.x;
                                        _vec_load_55[0 + 1] = _v2_193.y;
                                    }
                                    p0[10] = _vec_load_55[0];
                                    p1[10] = _vec_load_55[1];
                                }
                                if (c_end_2 > base_2 + 11) {
                                    long long part_row_43 = (long long)(base_2 + 11) * split_pitch + row_head[0];
                                    float _vec_load_56[2];
                                    {
                                        float2 _v2_194 = *reinterpret_cast<const float2*>(partial_out + (part_row_43 * 256 + dim[0]) + 0);
                                        _vec_load_56[0] = _v2_194.x;
                                        _vec_load_56[0 + 1] = _v2_194.y;
                                    }
                                    p0[11] = _vec_load_56[0];
                                    p1[11] = _vec_load_56[1];
                                }
                                if (c_end_2 > base_2 + 12) {
                                    long long part_row_44 = (long long)(base_2 + 12) * split_pitch + row_head[0];
                                    float _vec_load_57[2];
                                    {
                                        float2 _v2_195 = *reinterpret_cast<const float2*>(partial_out + (part_row_44 * 256 + dim[0]) + 0);
                                        _vec_load_57[0] = _v2_195.x;
                                        _vec_load_57[0 + 1] = _v2_195.y;
                                    }
                                    p0[12] = _vec_load_57[0];
                                    p1[12] = _vec_load_57[1];
                                }
                                if (c_end_2 > base_2 + 13) {
                                    long long part_row_45 = (long long)(base_2 + 13) * split_pitch + row_head[0];
                                    float _vec_load_58[2];
                                    {
                                        float2 _v2_196 = *reinterpret_cast<const float2*>(partial_out + (part_row_45 * 256 + dim[0]) + 0);
                                        _vec_load_58[0] = _v2_196.x;
                                        _vec_load_58[0 + 1] = _v2_196.y;
                                    }
                                    p0[13] = _vec_load_58[0];
                                    p1[13] = _vec_load_58[1];
                                }
                                if (c_end_2 > base_2 + 14) {
                                    long long part_row_46 = (long long)(base_2 + 14) * split_pitch + row_head[0];
                                    float _vec_load_59[2];
                                    {
                                        float2 _v2_197 = *reinterpret_cast<const float2*>(partial_out + (part_row_46 * 256 + dim[0]) + 0);
                                        _vec_load_59[0] = _v2_197.x;
                                        _vec_load_59[0 + 1] = _v2_197.y;
                                    }
                                    p0[14] = _vec_load_59[0];
                                    p1[14] = _vec_load_59[1];
                                }
                                if (c_end_2 > base_2 + 15) {
                                    long long part_row_47 = (long long)(base_2 + 15) * split_pitch + row_head[0];
                                    float _vec_load_60[2];
                                    {
                                        float2 _v2_198 = *reinterpret_cast<const float2*>(partial_out + (part_row_47 * 256 + dim[0]) + 0);
                                        _vec_load_60[0] = _v2_198.x;
                                        _vec_load_60[0 + 1] = _v2_198.y;
                                    }
                                    p0[15] = _vec_load_60[0];
                                    p1[15] = _vec_load_60[1];
                                }
                                bool live_16 = c_end_2 > base_2 && lse_max[0] > -CUDART_INF_F;
                                if (live_16) {
                                    float weight_32 = approx_exp2(lse_tab[tab[0] + base_2] - lse_max[0]);
                                    den[2] = den[2] + weight_32;
                                    acc0[2] = __fmaf_rn(weight_32, p0[0], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_32, p1[0], acc1[2]);
                                }
                                bool live_0_2 = c_end_2 > base_2 + 1 && lse_max[0] > -CUDART_INF_F;
                                if (live_0_2) {
                                    float weight_33 = approx_exp2(lse_tab[tab[0] + base_2 + 1] - lse_max[0]);
                                    den[2] = den[2] + weight_33;
                                    acc0[2] = __fmaf_rn(weight_33, p0[1], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_33, p1[1], acc1[2]);
                                }
                                bool live_1_2 = c_end_2 > base_2 + 2 && lse_max[0] > -CUDART_INF_F;
                                if (live_1_2) {
                                    float weight_34 = approx_exp2(lse_tab[tab[0] + base_2 + 2] - lse_max[0]);
                                    den[2] = den[2] + weight_34;
                                    acc0[2] = __fmaf_rn(weight_34, p0[2], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_34, p1[2], acc1[2]);
                                }
                                bool live_2_2 = c_end_2 > base_2 + 3 && lse_max[0] > -CUDART_INF_F;
                                if (live_2_2) {
                                    float weight_35 = approx_exp2(lse_tab[tab[0] + base_2 + 3] - lse_max[0]);
                                    den[2] = den[2] + weight_35;
                                    acc0[2] = __fmaf_rn(weight_35, p0[3], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_35, p1[3], acc1[2]);
                                }
                                bool live_3_2 = c_end_2 > base_2 + 4 && lse_max[0] > -CUDART_INF_F;
                                if (live_3_2) {
                                    float weight_36 = approx_exp2(lse_tab[tab[0] + base_2 + 4] - lse_max[0]);
                                    den[2] = den[2] + weight_36;
                                    acc0[2] = __fmaf_rn(weight_36, p0[4], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_36, p1[4], acc1[2]);
                                }
                                bool live_4_2 = c_end_2 > base_2 + 5 && lse_max[0] > -CUDART_INF_F;
                                if (live_4_2) {
                                    float weight_37 = approx_exp2(lse_tab[tab[0] + base_2 + 5] - lse_max[0]);
                                    den[2] = den[2] + weight_37;
                                    acc0[2] = __fmaf_rn(weight_37, p0[5], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_37, p1[5], acc1[2]);
                                }
                                bool live_5_2 = c_end_2 > base_2 + 6 && lse_max[0] > -CUDART_INF_F;
                                if (live_5_2) {
                                    float weight_38 = approx_exp2(lse_tab[tab[0] + base_2 + 6] - lse_max[0]);
                                    den[2] = den[2] + weight_38;
                                    acc0[2] = __fmaf_rn(weight_38, p0[6], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_38, p1[6], acc1[2]);
                                }
                                bool live_6_2 = c_end_2 > base_2 + 7 && lse_max[0] > -CUDART_INF_F;
                                if (live_6_2) {
                                    float weight_39 = approx_exp2(lse_tab[tab[0] + base_2 + 7] - lse_max[0]);
                                    den[2] = den[2] + weight_39;
                                    acc0[2] = __fmaf_rn(weight_39, p0[7], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_39, p1[7], acc1[2]);
                                }
                                bool live_7_2 = c_end_2 > base_2 + 8 && lse_max[0] > -CUDART_INF_F;
                                if (live_7_2) {
                                    float weight_40 = approx_exp2(lse_tab[tab[0] + base_2 + 8] - lse_max[0]);
                                    den[2] = den[2] + weight_40;
                                    acc0[2] = __fmaf_rn(weight_40, p0[8], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_40, p1[8], acc1[2]);
                                }
                                bool live_8_2 = c_end_2 > base_2 + 9 && lse_max[0] > -CUDART_INF_F;
                                if (live_8_2) {
                                    float weight_41 = approx_exp2(lse_tab[tab[0] + base_2 + 9] - lse_max[0]);
                                    den[2] = den[2] + weight_41;
                                    acc0[2] = __fmaf_rn(weight_41, p0[9], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_41, p1[9], acc1[2]);
                                }
                                bool live_9_2 = c_end_2 > base_2 + 10 && lse_max[0] > -CUDART_INF_F;
                                if (live_9_2) {
                                    float weight_42 = approx_exp2(lse_tab[tab[0] + base_2 + 10] - lse_max[0]);
                                    den[2] = den[2] + weight_42;
                                    acc0[2] = __fmaf_rn(weight_42, p0[10], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_42, p1[10], acc1[2]);
                                }
                                bool live_10_2 = c_end_2 > base_2 + 11 && lse_max[0] > -CUDART_INF_F;
                                if (live_10_2) {
                                    float weight_43 = approx_exp2(lse_tab[tab[0] + base_2 + 11] - lse_max[0]);
                                    den[2] = den[2] + weight_43;
                                    acc0[2] = __fmaf_rn(weight_43, p0[11], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_43, p1[11], acc1[2]);
                                }
                                bool live_11_2 = c_end_2 > base_2 + 12 && lse_max[0] > -CUDART_INF_F;
                                if (live_11_2) {
                                    float weight_44 = approx_exp2(lse_tab[tab[0] + base_2 + 12] - lse_max[0]);
                                    den[2] = den[2] + weight_44;
                                    acc0[2] = __fmaf_rn(weight_44, p0[12], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_44, p1[12], acc1[2]);
                                }
                                bool live_12_2 = c_end_2 > base_2 + 13 && lse_max[0] > -CUDART_INF_F;
                                if (live_12_2) {
                                    float weight_45 = approx_exp2(lse_tab[tab[0] + base_2 + 13] - lse_max[0]);
                                    den[2] = den[2] + weight_45;
                                    acc0[2] = __fmaf_rn(weight_45, p0[13], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_45, p1[13], acc1[2]);
                                }
                                bool live_13_2 = c_end_2 > base_2 + 14 && lse_max[0] > -CUDART_INF_F;
                                if (live_13_2) {
                                    float weight_46 = approx_exp2(lse_tab[tab[0] + base_2 + 14] - lse_max[0]);
                                    den[2] = den[2] + weight_46;
                                    acc0[2] = __fmaf_rn(weight_46, p0[14], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_46, p1[14], acc1[2]);
                                }
                                bool live_14_2 = c_end_2 > base_2 + 15 && lse_max[0] > -CUDART_INF_F;
                                if (live_14_2) {
                                    float weight_47 = approx_exp2(lse_tab[tab[0] + base_2 + 15] - lse_max[0]);
                                    den[2] = den[2] + weight_47;
                                    acc0[2] = __fmaf_rn(weight_47, p0[15], acc0[2]);
                                    acc1[2] = __fmaf_rn(weight_47, p1[15], acc1[2]);
                                }
                            }
                            int c_end_4 = ((chunk_w * 4) < (num_splits) ? (chunk_w * 4) : (num_splits));
                            int c_begin_5 = ((chunk_w * 3) < (c_end_4) ? (chunk_w * 3) : (c_end_4));
                            #pragma unroll 1
                            for (int base_3 = c_begin_5; base_3 < c_end_4; base_3 += 16) {
                                if (c_end_4 > base_3) {
                                    long long part_row_48 = (long long)base_3 * split_pitch + row_head[0];
                                    float _vec_load_61[2];
                                    {
                                        float2 _v2_199 = *reinterpret_cast<const float2*>(partial_out + (part_row_48 * 256 + dim[0]) + 0);
                                        _vec_load_61[0] = _v2_199.x;
                                        _vec_load_61[0 + 1] = _v2_199.y;
                                    }
                                    p0[0] = _vec_load_61[0];
                                    p1[0] = _vec_load_61[1];
                                }
                                if (c_end_4 > base_3 + 1) {
                                    long long part_row_49 = (long long)(base_3 + 1) * split_pitch + row_head[0];
                                    float _vec_load_62[2];
                                    {
                                        float2 _v2_200 = *reinterpret_cast<const float2*>(partial_out + (part_row_49 * 256 + dim[0]) + 0);
                                        _vec_load_62[0] = _v2_200.x;
                                        _vec_load_62[0 + 1] = _v2_200.y;
                                    }
                                    p0[1] = _vec_load_62[0];
                                    p1[1] = _vec_load_62[1];
                                }
                                if (c_end_4 > base_3 + 2) {
                                    long long part_row_50 = (long long)(base_3 + 2) * split_pitch + row_head[0];
                                    float _vec_load_63[2];
                                    {
                                        float2 _v2_201 = *reinterpret_cast<const float2*>(partial_out + (part_row_50 * 256 + dim[0]) + 0);
                                        _vec_load_63[0] = _v2_201.x;
                                        _vec_load_63[0 + 1] = _v2_201.y;
                                    }
                                    p0[2] = _vec_load_63[0];
                                    p1[2] = _vec_load_63[1];
                                }
                                if (c_end_4 > base_3 + 3) {
                                    long long part_row_51 = (long long)(base_3 + 3) * split_pitch + row_head[0];
                                    float _vec_load_64[2];
                                    {
                                        float2 _v2_202 = *reinterpret_cast<const float2*>(partial_out + (part_row_51 * 256 + dim[0]) + 0);
                                        _vec_load_64[0] = _v2_202.x;
                                        _vec_load_64[0 + 1] = _v2_202.y;
                                    }
                                    p0[3] = _vec_load_64[0];
                                    p1[3] = _vec_load_64[1];
                                }
                                if (c_end_4 > base_3 + 4) {
                                    long long part_row_52 = (long long)(base_3 + 4) * split_pitch + row_head[0];
                                    float _vec_load_65[2];
                                    {
                                        float2 _v2_203 = *reinterpret_cast<const float2*>(partial_out + (part_row_52 * 256 + dim[0]) + 0);
                                        _vec_load_65[0] = _v2_203.x;
                                        _vec_load_65[0 + 1] = _v2_203.y;
                                    }
                                    p0[4] = _vec_load_65[0];
                                    p1[4] = _vec_load_65[1];
                                }
                                if (c_end_4 > base_3 + 5) {
                                    long long part_row_53 = (long long)(base_3 + 5) * split_pitch + row_head[0];
                                    float _vec_load_66[2];
                                    {
                                        float2 _v2_204 = *reinterpret_cast<const float2*>(partial_out + (part_row_53 * 256 + dim[0]) + 0);
                                        _vec_load_66[0] = _v2_204.x;
                                        _vec_load_66[0 + 1] = _v2_204.y;
                                    }
                                    p0[5] = _vec_load_66[0];
                                    p1[5] = _vec_load_66[1];
                                }
                                if (c_end_4 > base_3 + 6) {
                                    long long part_row_54 = (long long)(base_3 + 6) * split_pitch + row_head[0];
                                    float _vec_load_67[2];
                                    {
                                        float2 _v2_205 = *reinterpret_cast<const float2*>(partial_out + (part_row_54 * 256 + dim[0]) + 0);
                                        _vec_load_67[0] = _v2_205.x;
                                        _vec_load_67[0 + 1] = _v2_205.y;
                                    }
                                    p0[6] = _vec_load_67[0];
                                    p1[6] = _vec_load_67[1];
                                }
                                if (c_end_4 > base_3 + 7) {
                                    long long part_row_55 = (long long)(base_3 + 7) * split_pitch + row_head[0];
                                    float _vec_load_68[2];
                                    {
                                        float2 _v2_206 = *reinterpret_cast<const float2*>(partial_out + (part_row_55 * 256 + dim[0]) + 0);
                                        _vec_load_68[0] = _v2_206.x;
                                        _vec_load_68[0 + 1] = _v2_206.y;
                                    }
                                    p0[7] = _vec_load_68[0];
                                    p1[7] = _vec_load_68[1];
                                }
                                if (c_end_4 > base_3 + 8) {
                                    long long part_row_56 = (long long)(base_3 + 8) * split_pitch + row_head[0];
                                    float _vec_load_69[2];
                                    {
                                        float2 _v2_207 = *reinterpret_cast<const float2*>(partial_out + (part_row_56 * 256 + dim[0]) + 0);
                                        _vec_load_69[0] = _v2_207.x;
                                        _vec_load_69[0 + 1] = _v2_207.y;
                                    }
                                    p0[8] = _vec_load_69[0];
                                    p1[8] = _vec_load_69[1];
                                }
                                if (c_end_4 > base_3 + 9) {
                                    long long part_row_57 = (long long)(base_3 + 9) * split_pitch + row_head[0];
                                    float _vec_load_70[2];
                                    {
                                        float2 _v2_208 = *reinterpret_cast<const float2*>(partial_out + (part_row_57 * 256 + dim[0]) + 0);
                                        _vec_load_70[0] = _v2_208.x;
                                        _vec_load_70[0 + 1] = _v2_208.y;
                                    }
                                    p0[9] = _vec_load_70[0];
                                    p1[9] = _vec_load_70[1];
                                }
                                if (c_end_4 > base_3 + 10) {
                                    long long part_row_58 = (long long)(base_3 + 10) * split_pitch + row_head[0];
                                    float _vec_load_71[2];
                                    {
                                        float2 _v2_209 = *reinterpret_cast<const float2*>(partial_out + (part_row_58 * 256 + dim[0]) + 0);
                                        _vec_load_71[0] = _v2_209.x;
                                        _vec_load_71[0 + 1] = _v2_209.y;
                                    }
                                    p0[10] = _vec_load_71[0];
                                    p1[10] = _vec_load_71[1];
                                }
                                if (c_end_4 > base_3 + 11) {
                                    long long part_row_59 = (long long)(base_3 + 11) * split_pitch + row_head[0];
                                    float _vec_load_72[2];
                                    {
                                        float2 _v2_210 = *reinterpret_cast<const float2*>(partial_out + (part_row_59 * 256 + dim[0]) + 0);
                                        _vec_load_72[0] = _v2_210.x;
                                        _vec_load_72[0 + 1] = _v2_210.y;
                                    }
                                    p0[11] = _vec_load_72[0];
                                    p1[11] = _vec_load_72[1];
                                }
                                if (c_end_4 > base_3 + 12) {
                                    long long part_row_60 = (long long)(base_3 + 12) * split_pitch + row_head[0];
                                    float _vec_load_73[2];
                                    {
                                        float2 _v2_211 = *reinterpret_cast<const float2*>(partial_out + (part_row_60 * 256 + dim[0]) + 0);
                                        _vec_load_73[0] = _v2_211.x;
                                        _vec_load_73[0 + 1] = _v2_211.y;
                                    }
                                    p0[12] = _vec_load_73[0];
                                    p1[12] = _vec_load_73[1];
                                }
                                if (c_end_4 > base_3 + 13) {
                                    long long part_row_61 = (long long)(base_3 + 13) * split_pitch + row_head[0];
                                    float _vec_load_74[2];
                                    {
                                        float2 _v2_212 = *reinterpret_cast<const float2*>(partial_out + (part_row_61 * 256 + dim[0]) + 0);
                                        _vec_load_74[0] = _v2_212.x;
                                        _vec_load_74[0 + 1] = _v2_212.y;
                                    }
                                    p0[13] = _vec_load_74[0];
                                    p1[13] = _vec_load_74[1];
                                }
                                if (c_end_4 > base_3 + 14) {
                                    long long part_row_62 = (long long)(base_3 + 14) * split_pitch + row_head[0];
                                    float _vec_load_75[2];
                                    {
                                        float2 _v2_213 = *reinterpret_cast<const float2*>(partial_out + (part_row_62 * 256 + dim[0]) + 0);
                                        _vec_load_75[0] = _v2_213.x;
                                        _vec_load_75[0 + 1] = _v2_213.y;
                                    }
                                    p0[14] = _vec_load_75[0];
                                    p1[14] = _vec_load_75[1];
                                }
                                if (c_end_4 > base_3 + 15) {
                                    long long part_row_63 = (long long)(base_3 + 15) * split_pitch + row_head[0];
                                    float _vec_load_76[2];
                                    {
                                        float2 _v2_214 = *reinterpret_cast<const float2*>(partial_out + (part_row_63 * 256 + dim[0]) + 0);
                                        _vec_load_76[0] = _v2_214.x;
                                        _vec_load_76[0 + 1] = _v2_214.y;
                                    }
                                    p0[15] = _vec_load_76[0];
                                    p1[15] = _vec_load_76[1];
                                }
                                bool live_17 = c_end_4 > base_3 && lse_max[0] > -CUDART_INF_F;
                                if (live_17) {
                                    float weight_48 = approx_exp2(lse_tab[tab[0] + base_3] - lse_max[0]);
                                    den[3] = den[3] + weight_48;
                                    acc0[3] = __fmaf_rn(weight_48, p0[0], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_48, p1[0], acc1[3]);
                                }
                                bool live_0_3 = c_end_4 > base_3 + 1 && lse_max[0] > -CUDART_INF_F;
                                if (live_0_3) {
                                    float weight_49 = approx_exp2(lse_tab[tab[0] + base_3 + 1] - lse_max[0]);
                                    den[3] = den[3] + weight_49;
                                    acc0[3] = __fmaf_rn(weight_49, p0[1], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_49, p1[1], acc1[3]);
                                }
                                bool live_1_3 = c_end_4 > base_3 + 2 && lse_max[0] > -CUDART_INF_F;
                                if (live_1_3) {
                                    float weight_50 = approx_exp2(lse_tab[tab[0] + base_3 + 2] - lse_max[0]);
                                    den[3] = den[3] + weight_50;
                                    acc0[3] = __fmaf_rn(weight_50, p0[2], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_50, p1[2], acc1[3]);
                                }
                                bool live_2_3 = c_end_4 > base_3 + 3 && lse_max[0] > -CUDART_INF_F;
                                if (live_2_3) {
                                    float weight_51 = approx_exp2(lse_tab[tab[0] + base_3 + 3] - lse_max[0]);
                                    den[3] = den[3] + weight_51;
                                    acc0[3] = __fmaf_rn(weight_51, p0[3], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_51, p1[3], acc1[3]);
                                }
                                bool live_3_3 = c_end_4 > base_3 + 4 && lse_max[0] > -CUDART_INF_F;
                                if (live_3_3) {
                                    float weight_52 = approx_exp2(lse_tab[tab[0] + base_3 + 4] - lse_max[0]);
                                    den[3] = den[3] + weight_52;
                                    acc0[3] = __fmaf_rn(weight_52, p0[4], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_52, p1[4], acc1[3]);
                                }
                                bool live_4_3 = c_end_4 > base_3 + 5 && lse_max[0] > -CUDART_INF_F;
                                if (live_4_3) {
                                    float weight_53 = approx_exp2(lse_tab[tab[0] + base_3 + 5] - lse_max[0]);
                                    den[3] = den[3] + weight_53;
                                    acc0[3] = __fmaf_rn(weight_53, p0[5], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_53, p1[5], acc1[3]);
                                }
                                bool live_5_3 = c_end_4 > base_3 + 6 && lse_max[0] > -CUDART_INF_F;
                                if (live_5_3) {
                                    float weight_54 = approx_exp2(lse_tab[tab[0] + base_3 + 6] - lse_max[0]);
                                    den[3] = den[3] + weight_54;
                                    acc0[3] = __fmaf_rn(weight_54, p0[6], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_54, p1[6], acc1[3]);
                                }
                                bool live_6_3 = c_end_4 > base_3 + 7 && lse_max[0] > -CUDART_INF_F;
                                if (live_6_3) {
                                    float weight_55 = approx_exp2(lse_tab[tab[0] + base_3 + 7] - lse_max[0]);
                                    den[3] = den[3] + weight_55;
                                    acc0[3] = __fmaf_rn(weight_55, p0[7], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_55, p1[7], acc1[3]);
                                }
                                bool live_7_3 = c_end_4 > base_3 + 8 && lse_max[0] > -CUDART_INF_F;
                                if (live_7_3) {
                                    float weight_56 = approx_exp2(lse_tab[tab[0] + base_3 + 8] - lse_max[0]);
                                    den[3] = den[3] + weight_56;
                                    acc0[3] = __fmaf_rn(weight_56, p0[8], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_56, p1[8], acc1[3]);
                                }
                                bool live_8_3 = c_end_4 > base_3 + 9 && lse_max[0] > -CUDART_INF_F;
                                if (live_8_3) {
                                    float weight_57 = approx_exp2(lse_tab[tab[0] + base_3 + 9] - lse_max[0]);
                                    den[3] = den[3] + weight_57;
                                    acc0[3] = __fmaf_rn(weight_57, p0[9], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_57, p1[9], acc1[3]);
                                }
                                bool live_9_3 = c_end_4 > base_3 + 10 && lse_max[0] > -CUDART_INF_F;
                                if (live_9_3) {
                                    float weight_58 = approx_exp2(lse_tab[tab[0] + base_3 + 10] - lse_max[0]);
                                    den[3] = den[3] + weight_58;
                                    acc0[3] = __fmaf_rn(weight_58, p0[10], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_58, p1[10], acc1[3]);
                                }
                                bool live_10_3 = c_end_4 > base_3 + 11 && lse_max[0] > -CUDART_INF_F;
                                if (live_10_3) {
                                    float weight_59 = approx_exp2(lse_tab[tab[0] + base_3 + 11] - lse_max[0]);
                                    den[3] = den[3] + weight_59;
                                    acc0[3] = __fmaf_rn(weight_59, p0[11], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_59, p1[11], acc1[3]);
                                }
                                bool live_11_3 = c_end_4 > base_3 + 12 && lse_max[0] > -CUDART_INF_F;
                                if (live_11_3) {
                                    float weight_60 = approx_exp2(lse_tab[tab[0] + base_3 + 12] - lse_max[0]);
                                    den[3] = den[3] + weight_60;
                                    acc0[3] = __fmaf_rn(weight_60, p0[12], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_60, p1[12], acc1[3]);
                                }
                                bool live_12_3 = c_end_4 > base_3 + 13 && lse_max[0] > -CUDART_INF_F;
                                if (live_12_3) {
                                    float weight_61 = approx_exp2(lse_tab[tab[0] + base_3 + 13] - lse_max[0]);
                                    den[3] = den[3] + weight_61;
                                    acc0[3] = __fmaf_rn(weight_61, p0[13], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_61, p1[13], acc1[3]);
                                }
                                bool live_13_3 = c_end_4 > base_3 + 14 && lse_max[0] > -CUDART_INF_F;
                                if (live_13_3) {
                                    float weight_62 = approx_exp2(lse_tab[tab[0] + base_3 + 14] - lse_max[0]);
                                    den[3] = den[3] + weight_62;
                                    acc0[3] = __fmaf_rn(weight_62, p0[14], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_62, p1[14], acc1[3]);
                                }
                                bool live_14_3 = c_end_4 > base_3 + 15 && lse_max[0] > -CUDART_INF_F;
                                if (live_14_3) {
                                    float weight_63 = approx_exp2(lse_tab[tab[0] + base_3 + 15] - lse_max[0]);
                                    den[3] = den[3] + weight_63;
                                    acc0[3] = __fmaf_rn(weight_63, p0[15], acc0[3]);
                                    acc1[3] = __fmaf_rn(weight_63, p1[15], acc1[3]);
                                }
                            }
                            if (valid[0] != 0) {
                                float den_k = den[0];
                                float acc0_k = acc0[0];
                                float acc1_k = acc1[0];
                                den_k = den_k + den[1];
                                acc0_k = acc0_k + acc0[1];
                                acc1_k = acc1_k + acc1[1];
                                den_k = den_k + den[2];
                                acc0_k = acc0_k + acc0[2];
                                acc1_k = acc1_k + acc1[2];
                                den_k = den_k + den[3];
                                acc0_k = acc0_k + acc0[3];
                                acc1_k = acc1_k + acc1[3];
                                float inv_2 = 0.0f;
                                if (den_k > 0.0f) {
                                    float _rcp_215 = approx_rcp(den_k);
                                    inv_2 = _rcp_215;
                                }
                                float attn0 = (float)(__nv_bfloat16)(acc0_k * inv_2);
                                float attn1 = (float)(__nv_bfloat16)(acc1_k * inv_2);
                                float _exp2_217 = approx_exp2((-gate0[0]) * 1.442695f);
                                float _rcp_216 = approx_rcp(1.0f + _exp2_217);
                                float result0 = attn0 * _rcp_216;
                                float _exp2_219 = approx_exp2((-gate1[0]) * 1.442695f);
                                float _rcp_218 = approx_rcp(1.0f + _exp2_219);
                                float result1 = attn1 * _rcp_218;
                                __nv_bfloat162 _bit_cast_src_220 = __float22bfloat162_rn(make_float2(result0, result1));
                                word[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_220)[0];
                                long long out_offset = row_i64 * out_stride_row + qo_head_2[0] * out_stride_head + dim[0];
                                *(reinterpret_cast<unsigned int*>(out + out_offset) + (0)) = word[0];
                            }
                        }
                    } else if (num_splits > 16) {
                        float lse_1[36];
                        float p0_1[36];
                        float p1_1[36];
                        float acc0_1[4];
                        float acc1_1[4];
                        float den_1[4];
                        int chunk_w_1 = (num_splits + 3) / 4;
                        if (num_splits <= 4) {
                            chunk_w_1 = num_splits;
                        }
                        int bound[4];
                        bound[0] = chunk_w_1;
                        bound[1] = chunk_w_1 * 2;
                        bound[2] = chunk_w_1 * 3;
                        bound[3] = chunk_w_1 * 4;
                        float lse_max_1[1];
                        int valid_1[1];
                        int tab_1[1];
                        long long dim_1[1];
                        long long qo_head_3[1];
                        long long row_head_1[1];
                        float gate0_1[1];
                        float gate1_1[1];
                        unsigned int word_1[1];
                        long long nqo_1 = (long long)num_qo_heads;
                        #pragma unroll 1
                        for (int p_1_1 = p_first; p_1_1 < end; p_1_1 += 128) {
                            int pk_1 = ((p_1_1) < (end - 1) ? (p_1_1) : (end - 1));
                            valid_1[0] = ((end > p_1_1) ? 1 : 0);
                            int head_3 = pk_1 >> 7;
                            dim_1[0] = (long long)((pk_1 & 127) * 2);
                            qo_head_3[0] = (long long)(first_head + head_3);
                            row_head_1[0] = row_i64 * nqo_1 + qo_head_3[0];
                            acc0_1[0] = 0.0f;
                            acc1_1[0] = 0.0f;
                            den_1[0] = 0.0f;
                            acc0_1[1] = 0.0f;
                            acc1_1[1] = 0.0f;
                            den_1[1] = 0.0f;
                            acc0_1[2] = 0.0f;
                            acc1_1[2] = 0.0f;
                            den_1[2] = 0.0f;
                            acc0_1[3] = 0.0f;
                            acc1_1[3] = 0.0f;
                            den_1[3] = 0.0f;
                            lse_max_1[0] = -CUDART_INF_F;
                            float _vec_load_77[2];
                            {
                                uint32_t _bf16x2_bits_221;
                                _bf16x2_bits_221 = *reinterpret_cast<const uint32_t*>(gate + (row_head_1[0] * 256 + dim_1[0]) + 0);
                                (&_vec_load_77[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_221) << 16);
                                (&_vec_load_77[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_221) & 0xffff0000u);
                            }
                            gate0_1[0] = _vec_load_77[0];
                            gate1_1[0] = _vec_load_77[1];
                            lse_1[0] = -CUDART_INF_F;
                            if (num_splits > 0) {
                                long long part_row_64 = row_head_1[0];
                                lse_1[0] = partial_lse[part_row_64];
                                float _vec_load_78[2];
                                {
                                    float2 _v2_222 = *reinterpret_cast<const float2*>(partial_out + (part_row_64 * 256 + dim_1[0]) + 0);
                                    _vec_load_78[0] = _v2_222.x;
                                    _vec_load_78[0 + 1] = _v2_222.y;
                                }
                                p0_1[0] = _vec_load_78[0];
                                p1_1[0] = _vec_load_78[1];
                            }
                            lse_1[1] = -CUDART_INF_F;
                            if (num_splits > 1) {
                                long long part_row_65 = split_pitch + row_head_1[0];
                                lse_1[1] = partial_lse[part_row_65];
                                float _vec_load_79[2];
                                {
                                    float2 _v2_223 = *reinterpret_cast<const float2*>(partial_out + (part_row_65 * 256 + dim_1[0]) + 0);
                                    _vec_load_79[0] = _v2_223.x;
                                    _vec_load_79[0 + 1] = _v2_223.y;
                                }
                                p0_1[1] = _vec_load_79[0];
                                p1_1[1] = _vec_load_79[1];
                            }
                            lse_1[2] = -CUDART_INF_F;
                            if (num_splits > 2) {
                                long long part_row_66 = 2 * split_pitch + row_head_1[0];
                                lse_1[2] = partial_lse[part_row_66];
                                float _vec_load_80[2];
                                {
                                    float2 _v2_224 = *reinterpret_cast<const float2*>(partial_out + (part_row_66 * 256 + dim_1[0]) + 0);
                                    _vec_load_80[0] = _v2_224.x;
                                    _vec_load_80[0 + 1] = _v2_224.y;
                                }
                                p0_1[2] = _vec_load_80[0];
                                p1_1[2] = _vec_load_80[1];
                            }
                            lse_1[3] = -CUDART_INF_F;
                            if (num_splits > 3) {
                                long long part_row_67 = 3 * split_pitch + row_head_1[0];
                                lse_1[3] = partial_lse[part_row_67];
                                float _vec_load_81[2];
                                {
                                    float2 _v2_225 = *reinterpret_cast<const float2*>(partial_out + (part_row_67 * 256 + dim_1[0]) + 0);
                                    _vec_load_81[0] = _v2_225.x;
                                    _vec_load_81[0 + 1] = _v2_225.y;
                                }
                                p0_1[3] = _vec_load_81[0];
                                p1_1[3] = _vec_load_81[1];
                            }
                            lse_1[4] = -CUDART_INF_F;
                            if (num_splits > 4) {
                                long long part_row_68 = 4 * split_pitch + row_head_1[0];
                                lse_1[4] = partial_lse[part_row_68];
                                float _vec_load_82[2];
                                {
                                    float2 _v2_226 = *reinterpret_cast<const float2*>(partial_out + (part_row_68 * 256 + dim_1[0]) + 0);
                                    _vec_load_82[0] = _v2_226.x;
                                    _vec_load_82[0 + 1] = _v2_226.y;
                                }
                                p0_1[4] = _vec_load_82[0];
                                p1_1[4] = _vec_load_82[1];
                            }
                            lse_1[5] = -CUDART_INF_F;
                            if (num_splits > 5) {
                                long long part_row_69 = 5 * split_pitch + row_head_1[0];
                                lse_1[5] = partial_lse[part_row_69];
                                float _vec_load_83[2];
                                {
                                    float2 _v2_227 = *reinterpret_cast<const float2*>(partial_out + (part_row_69 * 256 + dim_1[0]) + 0);
                                    _vec_load_83[0] = _v2_227.x;
                                    _vec_load_83[0 + 1] = _v2_227.y;
                                }
                                p0_1[5] = _vec_load_83[0];
                                p1_1[5] = _vec_load_83[1];
                            }
                            lse_1[6] = -CUDART_INF_F;
                            if (num_splits > 6) {
                                long long part_row_70 = 6 * split_pitch + row_head_1[0];
                                lse_1[6] = partial_lse[part_row_70];
                                float _vec_load_84[2];
                                {
                                    float2 _v2_228 = *reinterpret_cast<const float2*>(partial_out + (part_row_70 * 256 + dim_1[0]) + 0);
                                    _vec_load_84[0] = _v2_228.x;
                                    _vec_load_84[0 + 1] = _v2_228.y;
                                }
                                p0_1[6] = _vec_load_84[0];
                                p1_1[6] = _vec_load_84[1];
                            }
                            lse_1[7] = -CUDART_INF_F;
                            if (num_splits > 7) {
                                long long part_row_71 = 7 * split_pitch + row_head_1[0];
                                lse_1[7] = partial_lse[part_row_71];
                                float _vec_load_85[2];
                                {
                                    float2 _v2_229 = *reinterpret_cast<const float2*>(partial_out + (part_row_71 * 256 + dim_1[0]) + 0);
                                    _vec_load_85[0] = _v2_229.x;
                                    _vec_load_85[0 + 1] = _v2_229.y;
                                }
                                p0_1[7] = _vec_load_85[0];
                                p1_1[7] = _vec_load_85[1];
                            }
                            lse_1[8] = -CUDART_INF_F;
                            if (num_splits > 8) {
                                long long part_row_72 = 8 * split_pitch + row_head_1[0];
                                lse_1[8] = partial_lse[part_row_72];
                                float _vec_load_86[2];
                                {
                                    float2 _v2_230 = *reinterpret_cast<const float2*>(partial_out + (part_row_72 * 256 + dim_1[0]) + 0);
                                    _vec_load_86[0] = _v2_230.x;
                                    _vec_load_86[0 + 1] = _v2_230.y;
                                }
                                p0_1[8] = _vec_load_86[0];
                                p1_1[8] = _vec_load_86[1];
                            }
                            lse_1[9] = -CUDART_INF_F;
                            if (num_splits > 9) {
                                long long part_row_73 = 9 * split_pitch + row_head_1[0];
                                lse_1[9] = partial_lse[part_row_73];
                                float _vec_load_87[2];
                                {
                                    float2 _v2_231 = *reinterpret_cast<const float2*>(partial_out + (part_row_73 * 256 + dim_1[0]) + 0);
                                    _vec_load_87[0] = _v2_231.x;
                                    _vec_load_87[0 + 1] = _v2_231.y;
                                }
                                p0_1[9] = _vec_load_87[0];
                                p1_1[9] = _vec_load_87[1];
                            }
                            lse_1[10] = -CUDART_INF_F;
                            if (num_splits > 10) {
                                long long part_row_74 = 10 * split_pitch + row_head_1[0];
                                lse_1[10] = partial_lse[part_row_74];
                                float _vec_load_88[2];
                                {
                                    float2 _v2_232 = *reinterpret_cast<const float2*>(partial_out + (part_row_74 * 256 + dim_1[0]) + 0);
                                    _vec_load_88[0] = _v2_232.x;
                                    _vec_load_88[0 + 1] = _v2_232.y;
                                }
                                p0_1[10] = _vec_load_88[0];
                                p1_1[10] = _vec_load_88[1];
                            }
                            lse_1[11] = -CUDART_INF_F;
                            if (num_splits > 11) {
                                long long part_row_75 = 11 * split_pitch + row_head_1[0];
                                lse_1[11] = partial_lse[part_row_75];
                                float _vec_load_89[2];
                                {
                                    float2 _v2_233 = *reinterpret_cast<const float2*>(partial_out + (part_row_75 * 256 + dim_1[0]) + 0);
                                    _vec_load_89[0] = _v2_233.x;
                                    _vec_load_89[0 + 1] = _v2_233.y;
                                }
                                p0_1[11] = _vec_load_89[0];
                                p1_1[11] = _vec_load_89[1];
                            }
                            lse_1[12] = -CUDART_INF_F;
                            if (num_splits > 12) {
                                long long part_row_76 = 12 * split_pitch + row_head_1[0];
                                lse_1[12] = partial_lse[part_row_76];
                                float _vec_load_90[2];
                                {
                                    float2 _v2_234 = *reinterpret_cast<const float2*>(partial_out + (part_row_76 * 256 + dim_1[0]) + 0);
                                    _vec_load_90[0] = _v2_234.x;
                                    _vec_load_90[0 + 1] = _v2_234.y;
                                }
                                p0_1[12] = _vec_load_90[0];
                                p1_1[12] = _vec_load_90[1];
                            }
                            lse_1[13] = -CUDART_INF_F;
                            if (num_splits > 13) {
                                long long part_row_77 = 13 * split_pitch + row_head_1[0];
                                lse_1[13] = partial_lse[part_row_77];
                                float _vec_load_91[2];
                                {
                                    float2 _v2_235 = *reinterpret_cast<const float2*>(partial_out + (part_row_77 * 256 + dim_1[0]) + 0);
                                    _vec_load_91[0] = _v2_235.x;
                                    _vec_load_91[0 + 1] = _v2_235.y;
                                }
                                p0_1[13] = _vec_load_91[0];
                                p1_1[13] = _vec_load_91[1];
                            }
                            lse_1[14] = -CUDART_INF_F;
                            if (num_splits > 14) {
                                long long part_row_78 = 14 * split_pitch + row_head_1[0];
                                lse_1[14] = partial_lse[part_row_78];
                                float _vec_load_92[2];
                                {
                                    float2 _v2_236 = *reinterpret_cast<const float2*>(partial_out + (part_row_78 * 256 + dim_1[0]) + 0);
                                    _vec_load_92[0] = _v2_236.x;
                                    _vec_load_92[0 + 1] = _v2_236.y;
                                }
                                p0_1[14] = _vec_load_92[0];
                                p1_1[14] = _vec_load_92[1];
                            }
                            lse_1[15] = -CUDART_INF_F;
                            if (num_splits > 15) {
                                long long part_row_79 = 15 * split_pitch + row_head_1[0];
                                lse_1[15] = partial_lse[part_row_79];
                                float _vec_load_93[2];
                                {
                                    float2 _v2_237 = *reinterpret_cast<const float2*>(partial_out + (part_row_79 * 256 + dim_1[0]) + 0);
                                    _vec_load_93[0] = _v2_237.x;
                                    _vec_load_93[0 + 1] = _v2_237.y;
                                }
                                p0_1[15] = _vec_load_93[0];
                                p1_1[15] = _vec_load_93[1];
                            }
                            lse_1[16] = -CUDART_INF_F;
                            if (num_splits > 16) {
                                long long part_row_80 = 16 * split_pitch + row_head_1[0];
                                lse_1[16] = partial_lse[part_row_80];
                                float _vec_load_94[2];
                                {
                                    float2 _v2_238 = *reinterpret_cast<const float2*>(partial_out + (part_row_80 * 256 + dim_1[0]) + 0);
                                    _vec_load_94[0] = _v2_238.x;
                                    _vec_load_94[0 + 1] = _v2_238.y;
                                }
                                p0_1[16] = _vec_load_94[0];
                                p1_1[16] = _vec_load_94[1];
                            }
                            lse_1[17] = -CUDART_INF_F;
                            if (num_splits > 17) {
                                long long part_row_81 = 17 * split_pitch + row_head_1[0];
                                lse_1[17] = partial_lse[part_row_81];
                                float _vec_load_95[2];
                                {
                                    float2 _v2_239 = *reinterpret_cast<const float2*>(partial_out + (part_row_81 * 256 + dim_1[0]) + 0);
                                    _vec_load_95[0] = _v2_239.x;
                                    _vec_load_95[0 + 1] = _v2_239.y;
                                }
                                p0_1[17] = _vec_load_95[0];
                                p1_1[17] = _vec_load_95[1];
                            }
                            lse_1[18] = -CUDART_INF_F;
                            if (num_splits > 18) {
                                long long part_row_82 = 18 * split_pitch + row_head_1[0];
                                lse_1[18] = partial_lse[part_row_82];
                                float _vec_load_96[2];
                                {
                                    float2 _v2_240 = *reinterpret_cast<const float2*>(partial_out + (part_row_82 * 256 + dim_1[0]) + 0);
                                    _vec_load_96[0] = _v2_240.x;
                                    _vec_load_96[0 + 1] = _v2_240.y;
                                }
                                p0_1[18] = _vec_load_96[0];
                                p1_1[18] = _vec_load_96[1];
                            }
                            lse_1[19] = -CUDART_INF_F;
                            if (num_splits > 19) {
                                long long part_row_83 = 19 * split_pitch + row_head_1[0];
                                lse_1[19] = partial_lse[part_row_83];
                                float _vec_load_97[2];
                                {
                                    float2 _v2_241 = *reinterpret_cast<const float2*>(partial_out + (part_row_83 * 256 + dim_1[0]) + 0);
                                    _vec_load_97[0] = _v2_241.x;
                                    _vec_load_97[0 + 1] = _v2_241.y;
                                }
                                p0_1[19] = _vec_load_97[0];
                                p1_1[19] = _vec_load_97[1];
                            }
                            lse_1[20] = -CUDART_INF_F;
                            if (num_splits > 20) {
                                long long part_row_84 = 20 * split_pitch + row_head_1[0];
                                lse_1[20] = partial_lse[part_row_84];
                                float _vec_load_98[2];
                                {
                                    float2 _v2_242 = *reinterpret_cast<const float2*>(partial_out + (part_row_84 * 256 + dim_1[0]) + 0);
                                    _vec_load_98[0] = _v2_242.x;
                                    _vec_load_98[0 + 1] = _v2_242.y;
                                }
                                p0_1[20] = _vec_load_98[0];
                                p1_1[20] = _vec_load_98[1];
                            }
                            lse_1[21] = -CUDART_INF_F;
                            if (num_splits > 21) {
                                long long part_row_85 = 21 * split_pitch + row_head_1[0];
                                lse_1[21] = partial_lse[part_row_85];
                                float _vec_load_99[2];
                                {
                                    float2 _v2_243 = *reinterpret_cast<const float2*>(partial_out + (part_row_85 * 256 + dim_1[0]) + 0);
                                    _vec_load_99[0] = _v2_243.x;
                                    _vec_load_99[0 + 1] = _v2_243.y;
                                }
                                p0_1[21] = _vec_load_99[0];
                                p1_1[21] = _vec_load_99[1];
                            }
                            lse_1[22] = -CUDART_INF_F;
                            if (num_splits > 22) {
                                long long part_row_86 = 22 * split_pitch + row_head_1[0];
                                lse_1[22] = partial_lse[part_row_86];
                                float _vec_load_100[2];
                                {
                                    float2 _v2_244 = *reinterpret_cast<const float2*>(partial_out + (part_row_86 * 256 + dim_1[0]) + 0);
                                    _vec_load_100[0] = _v2_244.x;
                                    _vec_load_100[0 + 1] = _v2_244.y;
                                }
                                p0_1[22] = _vec_load_100[0];
                                p1_1[22] = _vec_load_100[1];
                            }
                            lse_1[23] = -CUDART_INF_F;
                            if (num_splits > 23) {
                                long long part_row_87 = 23 * split_pitch + row_head_1[0];
                                lse_1[23] = partial_lse[part_row_87];
                                float _vec_load_101[2];
                                {
                                    float2 _v2_245 = *reinterpret_cast<const float2*>(partial_out + (part_row_87 * 256 + dim_1[0]) + 0);
                                    _vec_load_101[0] = _v2_245.x;
                                    _vec_load_101[0 + 1] = _v2_245.y;
                                }
                                p0_1[23] = _vec_load_101[0];
                                p1_1[23] = _vec_load_101[1];
                            }
                            lse_1[24] = -CUDART_INF_F;
                            if (num_splits > 24) {
                                long long part_row_88 = 24 * split_pitch + row_head_1[0];
                                lse_1[24] = partial_lse[part_row_88];
                                float _vec_load_102[2];
                                {
                                    float2 _v2_246 = *reinterpret_cast<const float2*>(partial_out + (part_row_88 * 256 + dim_1[0]) + 0);
                                    _vec_load_102[0] = _v2_246.x;
                                    _vec_load_102[0 + 1] = _v2_246.y;
                                }
                                p0_1[24] = _vec_load_102[0];
                                p1_1[24] = _vec_load_102[1];
                            }
                            lse_1[25] = -CUDART_INF_F;
                            if (num_splits > 25) {
                                long long part_row_89 = 25 * split_pitch + row_head_1[0];
                                lse_1[25] = partial_lse[part_row_89];
                                float _vec_load_103[2];
                                {
                                    float2 _v2_247 = *reinterpret_cast<const float2*>(partial_out + (part_row_89 * 256 + dim_1[0]) + 0);
                                    _vec_load_103[0] = _v2_247.x;
                                    _vec_load_103[0 + 1] = _v2_247.y;
                                }
                                p0_1[25] = _vec_load_103[0];
                                p1_1[25] = _vec_load_103[1];
                            }
                            lse_1[26] = -CUDART_INF_F;
                            if (num_splits > 26) {
                                long long part_row_90 = 26 * split_pitch + row_head_1[0];
                                lse_1[26] = partial_lse[part_row_90];
                                float _vec_load_104[2];
                                {
                                    float2 _v2_248 = *reinterpret_cast<const float2*>(partial_out + (part_row_90 * 256 + dim_1[0]) + 0);
                                    _vec_load_104[0] = _v2_248.x;
                                    _vec_load_104[0 + 1] = _v2_248.y;
                                }
                                p0_1[26] = _vec_load_104[0];
                                p1_1[26] = _vec_load_104[1];
                            }
                            lse_1[27] = -CUDART_INF_F;
                            if (num_splits > 27) {
                                long long part_row_91 = 27 * split_pitch + row_head_1[0];
                                lse_1[27] = partial_lse[part_row_91];
                                float _vec_load_105[2];
                                {
                                    float2 _v2_249 = *reinterpret_cast<const float2*>(partial_out + (part_row_91 * 256 + dim_1[0]) + 0);
                                    _vec_load_105[0] = _v2_249.x;
                                    _vec_load_105[0 + 1] = _v2_249.y;
                                }
                                p0_1[27] = _vec_load_105[0];
                                p1_1[27] = _vec_load_105[1];
                            }
                            lse_1[28] = -CUDART_INF_F;
                            if (num_splits > 28) {
                                long long part_row_92 = 28 * split_pitch + row_head_1[0];
                                lse_1[28] = partial_lse[part_row_92];
                                float _vec_load_106[2];
                                {
                                    float2 _v2_250 = *reinterpret_cast<const float2*>(partial_out + (part_row_92 * 256 + dim_1[0]) + 0);
                                    _vec_load_106[0] = _v2_250.x;
                                    _vec_load_106[0 + 1] = _v2_250.y;
                                }
                                p0_1[28] = _vec_load_106[0];
                                p1_1[28] = _vec_load_106[1];
                            }
                            lse_1[29] = -CUDART_INF_F;
                            if (num_splits > 29) {
                                long long part_row_93 = 29 * split_pitch + row_head_1[0];
                                lse_1[29] = partial_lse[part_row_93];
                                float _vec_load_107[2];
                                {
                                    float2 _v2_251 = *reinterpret_cast<const float2*>(partial_out + (part_row_93 * 256 + dim_1[0]) + 0);
                                    _vec_load_107[0] = _v2_251.x;
                                    _vec_load_107[0 + 1] = _v2_251.y;
                                }
                                p0_1[29] = _vec_load_107[0];
                                p1_1[29] = _vec_load_107[1];
                            }
                            lse_1[30] = -CUDART_INF_F;
                            if (num_splits > 30) {
                                long long part_row_94 = 30 * split_pitch + row_head_1[0];
                                lse_1[30] = partial_lse[part_row_94];
                                float _vec_load_108[2];
                                {
                                    float2 _v2_252 = *reinterpret_cast<const float2*>(partial_out + (part_row_94 * 256 + dim_1[0]) + 0);
                                    _vec_load_108[0] = _v2_252.x;
                                    _vec_load_108[0 + 1] = _v2_252.y;
                                }
                                p0_1[30] = _vec_load_108[0];
                                p1_1[30] = _vec_load_108[1];
                            }
                            lse_1[31] = -CUDART_INF_F;
                            if (num_splits > 31) {
                                long long part_row_95 = 31 * split_pitch + row_head_1[0];
                                lse_1[31] = partial_lse[part_row_95];
                                float _vec_load_109[2];
                                {
                                    float2 _v2_253 = *reinterpret_cast<const float2*>(partial_out + (part_row_95 * 256 + dim_1[0]) + 0);
                                    _vec_load_109[0] = _v2_253.x;
                                    _vec_load_109[0 + 1] = _v2_253.y;
                                }
                                p0_1[31] = _vec_load_109[0];
                                p1_1[31] = _vec_load_109[1];
                            }
                            lse_1[32] = -CUDART_INF_F;
                            if (num_splits > 32) {
                                long long part_row_96 = 32 * split_pitch + row_head_1[0];
                                lse_1[32] = partial_lse[part_row_96];
                                float _vec_load_110[2];
                                {
                                    float2 _v2_254 = *reinterpret_cast<const float2*>(partial_out + (part_row_96 * 256 + dim_1[0]) + 0);
                                    _vec_load_110[0] = _v2_254.x;
                                    _vec_load_110[0 + 1] = _v2_254.y;
                                }
                                p0_1[32] = _vec_load_110[0];
                                p1_1[32] = _vec_load_110[1];
                            }
                            lse_1[33] = -CUDART_INF_F;
                            if (num_splits > 33) {
                                long long part_row_97 = 33 * split_pitch + row_head_1[0];
                                lse_1[33] = partial_lse[part_row_97];
                                float _vec_load_111[2];
                                {
                                    float2 _v2_255 = *reinterpret_cast<const float2*>(partial_out + (part_row_97 * 256 + dim_1[0]) + 0);
                                    _vec_load_111[0] = _v2_255.x;
                                    _vec_load_111[0 + 1] = _v2_255.y;
                                }
                                p0_1[33] = _vec_load_111[0];
                                p1_1[33] = _vec_load_111[1];
                            }
                            lse_1[34] = -CUDART_INF_F;
                            if (num_splits > 34) {
                                long long part_row_98 = 34 * split_pitch + row_head_1[0];
                                lse_1[34] = partial_lse[part_row_98];
                                float _vec_load_112[2];
                                {
                                    float2 _v2_256 = *reinterpret_cast<const float2*>(partial_out + (part_row_98 * 256 + dim_1[0]) + 0);
                                    _vec_load_112[0] = _v2_256.x;
                                    _vec_load_112[0 + 1] = _v2_256.y;
                                }
                                p0_1[34] = _vec_load_112[0];
                                p1_1[34] = _vec_load_112[1];
                            }
                            lse_1[35] = -CUDART_INF_F;
                            if (num_splits > 35) {
                                long long part_row_99 = 35 * split_pitch + row_head_1[0];
                                lse_1[35] = partial_lse[part_row_99];
                                float _vec_load_113[2];
                                {
                                    float2 _v2_257 = *reinterpret_cast<const float2*>(partial_out + (part_row_99 * 256 + dim_1[0]) + 0);
                                    _vec_load_113[0] = _v2_257.x;
                                    _vec_load_113[0 + 1] = _v2_257.y;
                                }
                                p0_1[35] = _vec_load_113[0];
                                p1_1[35] = _vec_load_113[1];
                            }
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[0]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[1]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[2]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[3]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[4]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[5]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[6]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[7]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[8]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[9]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[10]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[11]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[12]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[13]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[14]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[15]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[16]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[17]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[18]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[19]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[20]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[21]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[22]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[23]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[24]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[25]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[26]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[27]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[28]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[29]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[30]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[31]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[32]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[33]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[34]);
                            lse_max_1[0] = fmaxf(lse_max_1[0], lse_1[35]);
                            bool live_18 = num_splits > 0 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_18) {
                                float weight_64 = approx_exp2(lse_1[0] - lse_max_1[0]);
                                if (bound[0] > 0) {
                                    den_1[0] = den_1[0] + weight_64;
                                    acc0_1[0] = __fmaf_rn(weight_64, p0_1[0], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_64, p1_1[0], acc1_1[0]);
                                } else if (bound[1] > 0) {
                                    den_1[1] = den_1[1] + weight_64;
                                    acc0_1[1] = __fmaf_rn(weight_64, p0_1[0], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_64, p1_1[0], acc1_1[1]);
                                } else {
                                    if (bound[2] > 0) {
                                        den_1[2] = den_1[2] + weight_64;
                                        acc0_1[2] = __fmaf_rn(weight_64, p0_1[0], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_64, p1_1[0], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_64;
                                        acc0_1[3] = __fmaf_rn(weight_64, p0_1[0], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_64, p1_1[0], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_0_4 = num_splits > 1 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_0_4) {
                                float weight_65 = approx_exp2(lse_1[1] - lse_max_1[0]);
                                if (bound[0] > 1) {
                                    den_1[0] = den_1[0] + weight_65;
                                    acc0_1[0] = __fmaf_rn(weight_65, p0_1[1], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_65, p1_1[1], acc1_1[0]);
                                } else if (bound[1] > 1) {
                                    den_1[1] = den_1[1] + weight_65;
                                    acc0_1[1] = __fmaf_rn(weight_65, p0_1[1], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_65, p1_1[1], acc1_1[1]);
                                } else {
                                    if (bound[2] > 1) {
                                        den_1[2] = den_1[2] + weight_65;
                                        acc0_1[2] = __fmaf_rn(weight_65, p0_1[1], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_65, p1_1[1], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_65;
                                        acc0_1[3] = __fmaf_rn(weight_65, p0_1[1], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_65, p1_1[1], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_1_4 = num_splits > 2 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_1_4) {
                                float weight_66 = approx_exp2(lse_1[2] - lse_max_1[0]);
                                if (bound[0] > 2) {
                                    den_1[0] = den_1[0] + weight_66;
                                    acc0_1[0] = __fmaf_rn(weight_66, p0_1[2], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_66, p1_1[2], acc1_1[0]);
                                } else if (bound[1] > 2) {
                                    den_1[1] = den_1[1] + weight_66;
                                    acc0_1[1] = __fmaf_rn(weight_66, p0_1[2], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_66, p1_1[2], acc1_1[1]);
                                } else {
                                    if (bound[2] > 2) {
                                        den_1[2] = den_1[2] + weight_66;
                                        acc0_1[2] = __fmaf_rn(weight_66, p0_1[2], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_66, p1_1[2], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_66;
                                        acc0_1[3] = __fmaf_rn(weight_66, p0_1[2], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_66, p1_1[2], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_2_4 = num_splits > 3 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_2_4) {
                                float weight_67 = approx_exp2(lse_1[3] - lse_max_1[0]);
                                if (bound[0] > 3) {
                                    den_1[0] = den_1[0] + weight_67;
                                    acc0_1[0] = __fmaf_rn(weight_67, p0_1[3], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_67, p1_1[3], acc1_1[0]);
                                } else if (bound[1] > 3) {
                                    den_1[1] = den_1[1] + weight_67;
                                    acc0_1[1] = __fmaf_rn(weight_67, p0_1[3], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_67, p1_1[3], acc1_1[1]);
                                } else {
                                    if (bound[2] > 3) {
                                        den_1[2] = den_1[2] + weight_67;
                                        acc0_1[2] = __fmaf_rn(weight_67, p0_1[3], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_67, p1_1[3], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_67;
                                        acc0_1[3] = __fmaf_rn(weight_67, p0_1[3], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_67, p1_1[3], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_3_4 = num_splits > 4 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_3_4) {
                                float weight_68 = approx_exp2(lse_1[4] - lse_max_1[0]);
                                if (bound[0] > 4) {
                                    den_1[0] = den_1[0] + weight_68;
                                    acc0_1[0] = __fmaf_rn(weight_68, p0_1[4], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_68, p1_1[4], acc1_1[0]);
                                } else if (bound[1] > 4) {
                                    den_1[1] = den_1[1] + weight_68;
                                    acc0_1[1] = __fmaf_rn(weight_68, p0_1[4], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_68, p1_1[4], acc1_1[1]);
                                } else {
                                    if (bound[2] > 4) {
                                        den_1[2] = den_1[2] + weight_68;
                                        acc0_1[2] = __fmaf_rn(weight_68, p0_1[4], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_68, p1_1[4], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_68;
                                        acc0_1[3] = __fmaf_rn(weight_68, p0_1[4], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_68, p1_1[4], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_4_4 = num_splits > 5 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_4_4) {
                                float weight_69 = approx_exp2(lse_1[5] - lse_max_1[0]);
                                if (bound[0] > 5) {
                                    den_1[0] = den_1[0] + weight_69;
                                    acc0_1[0] = __fmaf_rn(weight_69, p0_1[5], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_69, p1_1[5], acc1_1[0]);
                                } else if (bound[1] > 5) {
                                    den_1[1] = den_1[1] + weight_69;
                                    acc0_1[1] = __fmaf_rn(weight_69, p0_1[5], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_69, p1_1[5], acc1_1[1]);
                                } else {
                                    if (bound[2] > 5) {
                                        den_1[2] = den_1[2] + weight_69;
                                        acc0_1[2] = __fmaf_rn(weight_69, p0_1[5], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_69, p1_1[5], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_69;
                                        acc0_1[3] = __fmaf_rn(weight_69, p0_1[5], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_69, p1_1[5], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_5_4 = num_splits > 6 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_5_4) {
                                float weight_70 = approx_exp2(lse_1[6] - lse_max_1[0]);
                                if (bound[0] > 6) {
                                    den_1[0] = den_1[0] + weight_70;
                                    acc0_1[0] = __fmaf_rn(weight_70, p0_1[6], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_70, p1_1[6], acc1_1[0]);
                                } else if (bound[1] > 6) {
                                    den_1[1] = den_1[1] + weight_70;
                                    acc0_1[1] = __fmaf_rn(weight_70, p0_1[6], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_70, p1_1[6], acc1_1[1]);
                                } else {
                                    if (bound[2] > 6) {
                                        den_1[2] = den_1[2] + weight_70;
                                        acc0_1[2] = __fmaf_rn(weight_70, p0_1[6], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_70, p1_1[6], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_70;
                                        acc0_1[3] = __fmaf_rn(weight_70, p0_1[6], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_70, p1_1[6], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_6_4 = num_splits > 7 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_6_4) {
                                float weight_71 = approx_exp2(lse_1[7] - lse_max_1[0]);
                                if (bound[0] > 7) {
                                    den_1[0] = den_1[0] + weight_71;
                                    acc0_1[0] = __fmaf_rn(weight_71, p0_1[7], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_71, p1_1[7], acc1_1[0]);
                                } else if (bound[1] > 7) {
                                    den_1[1] = den_1[1] + weight_71;
                                    acc0_1[1] = __fmaf_rn(weight_71, p0_1[7], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_71, p1_1[7], acc1_1[1]);
                                } else {
                                    if (bound[2] > 7) {
                                        den_1[2] = den_1[2] + weight_71;
                                        acc0_1[2] = __fmaf_rn(weight_71, p0_1[7], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_71, p1_1[7], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_71;
                                        acc0_1[3] = __fmaf_rn(weight_71, p0_1[7], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_71, p1_1[7], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_7_4 = num_splits > 8 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_7_4) {
                                float weight_72 = approx_exp2(lse_1[8] - lse_max_1[0]);
                                if (bound[0] > 8) {
                                    den_1[0] = den_1[0] + weight_72;
                                    acc0_1[0] = __fmaf_rn(weight_72, p0_1[8], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_72, p1_1[8], acc1_1[0]);
                                } else if (bound[1] > 8) {
                                    den_1[1] = den_1[1] + weight_72;
                                    acc0_1[1] = __fmaf_rn(weight_72, p0_1[8], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_72, p1_1[8], acc1_1[1]);
                                } else {
                                    if (bound[2] > 8) {
                                        den_1[2] = den_1[2] + weight_72;
                                        acc0_1[2] = __fmaf_rn(weight_72, p0_1[8], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_72, p1_1[8], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_72;
                                        acc0_1[3] = __fmaf_rn(weight_72, p0_1[8], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_72, p1_1[8], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_8_4 = num_splits > 9 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_8_4) {
                                float weight_73 = approx_exp2(lse_1[9] - lse_max_1[0]);
                                if (bound[0] > 9) {
                                    den_1[0] = den_1[0] + weight_73;
                                    acc0_1[0] = __fmaf_rn(weight_73, p0_1[9], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_73, p1_1[9], acc1_1[0]);
                                } else if (bound[1] > 9) {
                                    den_1[1] = den_1[1] + weight_73;
                                    acc0_1[1] = __fmaf_rn(weight_73, p0_1[9], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_73, p1_1[9], acc1_1[1]);
                                } else {
                                    if (bound[2] > 9) {
                                        den_1[2] = den_1[2] + weight_73;
                                        acc0_1[2] = __fmaf_rn(weight_73, p0_1[9], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_73, p1_1[9], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_73;
                                        acc0_1[3] = __fmaf_rn(weight_73, p0_1[9], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_73, p1_1[9], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_9_4 = num_splits > 10 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_9_4) {
                                float weight_74 = approx_exp2(lse_1[10] - lse_max_1[0]);
                                if (bound[0] > 10) {
                                    den_1[0] = den_1[0] + weight_74;
                                    acc0_1[0] = __fmaf_rn(weight_74, p0_1[10], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_74, p1_1[10], acc1_1[0]);
                                } else if (bound[1] > 10) {
                                    den_1[1] = den_1[1] + weight_74;
                                    acc0_1[1] = __fmaf_rn(weight_74, p0_1[10], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_74, p1_1[10], acc1_1[1]);
                                } else {
                                    if (bound[2] > 10) {
                                        den_1[2] = den_1[2] + weight_74;
                                        acc0_1[2] = __fmaf_rn(weight_74, p0_1[10], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_74, p1_1[10], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_74;
                                        acc0_1[3] = __fmaf_rn(weight_74, p0_1[10], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_74, p1_1[10], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_10_4 = num_splits > 11 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_10_4) {
                                float weight_75 = approx_exp2(lse_1[11] - lse_max_1[0]);
                                if (bound[0] > 11) {
                                    den_1[0] = den_1[0] + weight_75;
                                    acc0_1[0] = __fmaf_rn(weight_75, p0_1[11], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_75, p1_1[11], acc1_1[0]);
                                } else if (bound[1] > 11) {
                                    den_1[1] = den_1[1] + weight_75;
                                    acc0_1[1] = __fmaf_rn(weight_75, p0_1[11], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_75, p1_1[11], acc1_1[1]);
                                } else {
                                    if (bound[2] > 11) {
                                        den_1[2] = den_1[2] + weight_75;
                                        acc0_1[2] = __fmaf_rn(weight_75, p0_1[11], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_75, p1_1[11], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_75;
                                        acc0_1[3] = __fmaf_rn(weight_75, p0_1[11], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_75, p1_1[11], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_11_4 = num_splits > 12 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_11_4) {
                                float weight_76 = approx_exp2(lse_1[12] - lse_max_1[0]);
                                if (bound[0] > 12) {
                                    den_1[0] = den_1[0] + weight_76;
                                    acc0_1[0] = __fmaf_rn(weight_76, p0_1[12], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_76, p1_1[12], acc1_1[0]);
                                } else if (bound[1] > 12) {
                                    den_1[1] = den_1[1] + weight_76;
                                    acc0_1[1] = __fmaf_rn(weight_76, p0_1[12], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_76, p1_1[12], acc1_1[1]);
                                } else {
                                    if (bound[2] > 12) {
                                        den_1[2] = den_1[2] + weight_76;
                                        acc0_1[2] = __fmaf_rn(weight_76, p0_1[12], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_76, p1_1[12], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_76;
                                        acc0_1[3] = __fmaf_rn(weight_76, p0_1[12], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_76, p1_1[12], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_12_4 = num_splits > 13 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_12_4) {
                                float weight_77 = approx_exp2(lse_1[13] - lse_max_1[0]);
                                if (bound[0] > 13) {
                                    den_1[0] = den_1[0] + weight_77;
                                    acc0_1[0] = __fmaf_rn(weight_77, p0_1[13], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_77, p1_1[13], acc1_1[0]);
                                } else if (bound[1] > 13) {
                                    den_1[1] = den_1[1] + weight_77;
                                    acc0_1[1] = __fmaf_rn(weight_77, p0_1[13], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_77, p1_1[13], acc1_1[1]);
                                } else {
                                    if (bound[2] > 13) {
                                        den_1[2] = den_1[2] + weight_77;
                                        acc0_1[2] = __fmaf_rn(weight_77, p0_1[13], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_77, p1_1[13], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_77;
                                        acc0_1[3] = __fmaf_rn(weight_77, p0_1[13], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_77, p1_1[13], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_13_4 = num_splits > 14 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_13_4) {
                                float weight_78 = approx_exp2(lse_1[14] - lse_max_1[0]);
                                if (bound[0] > 14) {
                                    den_1[0] = den_1[0] + weight_78;
                                    acc0_1[0] = __fmaf_rn(weight_78, p0_1[14], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_78, p1_1[14], acc1_1[0]);
                                } else if (bound[1] > 14) {
                                    den_1[1] = den_1[1] + weight_78;
                                    acc0_1[1] = __fmaf_rn(weight_78, p0_1[14], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_78, p1_1[14], acc1_1[1]);
                                } else {
                                    if (bound[2] > 14) {
                                        den_1[2] = den_1[2] + weight_78;
                                        acc0_1[2] = __fmaf_rn(weight_78, p0_1[14], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_78, p1_1[14], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_78;
                                        acc0_1[3] = __fmaf_rn(weight_78, p0_1[14], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_78, p1_1[14], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_14_4 = num_splits > 15 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_14_4) {
                                float weight_79 = approx_exp2(lse_1[15] - lse_max_1[0]);
                                if (bound[0] > 15) {
                                    den_1[0] = den_1[0] + weight_79;
                                    acc0_1[0] = __fmaf_rn(weight_79, p0_1[15], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_79, p1_1[15], acc1_1[0]);
                                } else if (bound[1] > 15) {
                                    den_1[1] = den_1[1] + weight_79;
                                    acc0_1[1] = __fmaf_rn(weight_79, p0_1[15], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_79, p1_1[15], acc1_1[1]);
                                } else {
                                    if (bound[2] > 15) {
                                        den_1[2] = den_1[2] + weight_79;
                                        acc0_1[2] = __fmaf_rn(weight_79, p0_1[15], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_79, p1_1[15], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_79;
                                        acc0_1[3] = __fmaf_rn(weight_79, p0_1[15], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_79, p1_1[15], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_15_1 = num_splits > 16 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_15_1) {
                                float weight_80 = approx_exp2(lse_1[16] - lse_max_1[0]);
                                if (bound[0] > 16) {
                                    den_1[0] = den_1[0] + weight_80;
                                    acc0_1[0] = __fmaf_rn(weight_80, p0_1[16], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_80, p1_1[16], acc1_1[0]);
                                } else if (bound[1] > 16) {
                                    den_1[1] = den_1[1] + weight_80;
                                    acc0_1[1] = __fmaf_rn(weight_80, p0_1[16], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_80, p1_1[16], acc1_1[1]);
                                } else {
                                    if (bound[2] > 16) {
                                        den_1[2] = den_1[2] + weight_80;
                                        acc0_1[2] = __fmaf_rn(weight_80, p0_1[16], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_80, p1_1[16], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_80;
                                        acc0_1[3] = __fmaf_rn(weight_80, p0_1[16], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_80, p1_1[16], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_16_1 = num_splits > 17 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_16_1) {
                                float weight_81 = approx_exp2(lse_1[17] - lse_max_1[0]);
                                if (bound[0] > 17) {
                                    den_1[0] = den_1[0] + weight_81;
                                    acc0_1[0] = __fmaf_rn(weight_81, p0_1[17], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_81, p1_1[17], acc1_1[0]);
                                } else if (bound[1] > 17) {
                                    den_1[1] = den_1[1] + weight_81;
                                    acc0_1[1] = __fmaf_rn(weight_81, p0_1[17], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_81, p1_1[17], acc1_1[1]);
                                } else {
                                    if (bound[2] > 17) {
                                        den_1[2] = den_1[2] + weight_81;
                                        acc0_1[2] = __fmaf_rn(weight_81, p0_1[17], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_81, p1_1[17], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_81;
                                        acc0_1[3] = __fmaf_rn(weight_81, p0_1[17], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_81, p1_1[17], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_17_1 = num_splits > 18 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_17_1) {
                                float weight_82 = approx_exp2(lse_1[18] - lse_max_1[0]);
                                if (bound[0] > 18) {
                                    den_1[0] = den_1[0] + weight_82;
                                    acc0_1[0] = __fmaf_rn(weight_82, p0_1[18], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_82, p1_1[18], acc1_1[0]);
                                } else if (bound[1] > 18) {
                                    den_1[1] = den_1[1] + weight_82;
                                    acc0_1[1] = __fmaf_rn(weight_82, p0_1[18], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_82, p1_1[18], acc1_1[1]);
                                } else {
                                    if (bound[2] > 18) {
                                        den_1[2] = den_1[2] + weight_82;
                                        acc0_1[2] = __fmaf_rn(weight_82, p0_1[18], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_82, p1_1[18], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_82;
                                        acc0_1[3] = __fmaf_rn(weight_82, p0_1[18], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_82, p1_1[18], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_18_1 = num_splits > 19 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_18_1) {
                                float weight_83 = approx_exp2(lse_1[19] - lse_max_1[0]);
                                if (bound[0] > 19) {
                                    den_1[0] = den_1[0] + weight_83;
                                    acc0_1[0] = __fmaf_rn(weight_83, p0_1[19], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_83, p1_1[19], acc1_1[0]);
                                } else if (bound[1] > 19) {
                                    den_1[1] = den_1[1] + weight_83;
                                    acc0_1[1] = __fmaf_rn(weight_83, p0_1[19], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_83, p1_1[19], acc1_1[1]);
                                } else {
                                    if (bound[2] > 19) {
                                        den_1[2] = den_1[2] + weight_83;
                                        acc0_1[2] = __fmaf_rn(weight_83, p0_1[19], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_83, p1_1[19], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_83;
                                        acc0_1[3] = __fmaf_rn(weight_83, p0_1[19], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_83, p1_1[19], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_19 = num_splits > 20 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_19) {
                                float weight_84 = approx_exp2(lse_1[20] - lse_max_1[0]);
                                if (bound[0] > 20) {
                                    den_1[0] = den_1[0] + weight_84;
                                    acc0_1[0] = __fmaf_rn(weight_84, p0_1[20], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_84, p1_1[20], acc1_1[0]);
                                } else if (bound[1] > 20) {
                                    den_1[1] = den_1[1] + weight_84;
                                    acc0_1[1] = __fmaf_rn(weight_84, p0_1[20], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_84, p1_1[20], acc1_1[1]);
                                } else {
                                    if (bound[2] > 20) {
                                        den_1[2] = den_1[2] + weight_84;
                                        acc0_1[2] = __fmaf_rn(weight_84, p0_1[20], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_84, p1_1[20], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_84;
                                        acc0_1[3] = __fmaf_rn(weight_84, p0_1[20], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_84, p1_1[20], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_20 = num_splits > 21 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_20) {
                                float weight_85 = approx_exp2(lse_1[21] - lse_max_1[0]);
                                if (bound[0] > 21) {
                                    den_1[0] = den_1[0] + weight_85;
                                    acc0_1[0] = __fmaf_rn(weight_85, p0_1[21], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_85, p1_1[21], acc1_1[0]);
                                } else if (bound[1] > 21) {
                                    den_1[1] = den_1[1] + weight_85;
                                    acc0_1[1] = __fmaf_rn(weight_85, p0_1[21], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_85, p1_1[21], acc1_1[1]);
                                } else {
                                    if (bound[2] > 21) {
                                        den_1[2] = den_1[2] + weight_85;
                                        acc0_1[2] = __fmaf_rn(weight_85, p0_1[21], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_85, p1_1[21], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_85;
                                        acc0_1[3] = __fmaf_rn(weight_85, p0_1[21], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_85, p1_1[21], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_21 = num_splits > 22 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_21) {
                                float weight_86 = approx_exp2(lse_1[22] - lse_max_1[0]);
                                if (bound[0] > 22) {
                                    den_1[0] = den_1[0] + weight_86;
                                    acc0_1[0] = __fmaf_rn(weight_86, p0_1[22], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_86, p1_1[22], acc1_1[0]);
                                } else if (bound[1] > 22) {
                                    den_1[1] = den_1[1] + weight_86;
                                    acc0_1[1] = __fmaf_rn(weight_86, p0_1[22], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_86, p1_1[22], acc1_1[1]);
                                } else {
                                    if (bound[2] > 22) {
                                        den_1[2] = den_1[2] + weight_86;
                                        acc0_1[2] = __fmaf_rn(weight_86, p0_1[22], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_86, p1_1[22], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_86;
                                        acc0_1[3] = __fmaf_rn(weight_86, p0_1[22], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_86, p1_1[22], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_22 = num_splits > 23 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_22) {
                                float weight_87 = approx_exp2(lse_1[23] - lse_max_1[0]);
                                if (bound[0] > 23) {
                                    den_1[0] = den_1[0] + weight_87;
                                    acc0_1[0] = __fmaf_rn(weight_87, p0_1[23], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_87, p1_1[23], acc1_1[0]);
                                } else if (bound[1] > 23) {
                                    den_1[1] = den_1[1] + weight_87;
                                    acc0_1[1] = __fmaf_rn(weight_87, p0_1[23], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_87, p1_1[23], acc1_1[1]);
                                } else {
                                    if (bound[2] > 23) {
                                        den_1[2] = den_1[2] + weight_87;
                                        acc0_1[2] = __fmaf_rn(weight_87, p0_1[23], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_87, p1_1[23], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_87;
                                        acc0_1[3] = __fmaf_rn(weight_87, p0_1[23], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_87, p1_1[23], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_23 = num_splits > 24 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_23) {
                                float weight_88 = approx_exp2(lse_1[24] - lse_max_1[0]);
                                if (bound[0] > 24) {
                                    den_1[0] = den_1[0] + weight_88;
                                    acc0_1[0] = __fmaf_rn(weight_88, p0_1[24], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_88, p1_1[24], acc1_1[0]);
                                } else if (bound[1] > 24) {
                                    den_1[1] = den_1[1] + weight_88;
                                    acc0_1[1] = __fmaf_rn(weight_88, p0_1[24], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_88, p1_1[24], acc1_1[1]);
                                } else {
                                    if (bound[2] > 24) {
                                        den_1[2] = den_1[2] + weight_88;
                                        acc0_1[2] = __fmaf_rn(weight_88, p0_1[24], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_88, p1_1[24], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_88;
                                        acc0_1[3] = __fmaf_rn(weight_88, p0_1[24], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_88, p1_1[24], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_24 = num_splits > 25 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_24) {
                                float weight_89 = approx_exp2(lse_1[25] - lse_max_1[0]);
                                if (bound[0] > 25) {
                                    den_1[0] = den_1[0] + weight_89;
                                    acc0_1[0] = __fmaf_rn(weight_89, p0_1[25], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_89, p1_1[25], acc1_1[0]);
                                } else if (bound[1] > 25) {
                                    den_1[1] = den_1[1] + weight_89;
                                    acc0_1[1] = __fmaf_rn(weight_89, p0_1[25], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_89, p1_1[25], acc1_1[1]);
                                } else {
                                    if (bound[2] > 25) {
                                        den_1[2] = den_1[2] + weight_89;
                                        acc0_1[2] = __fmaf_rn(weight_89, p0_1[25], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_89, p1_1[25], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_89;
                                        acc0_1[3] = __fmaf_rn(weight_89, p0_1[25], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_89, p1_1[25], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_25 = num_splits > 26 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_25) {
                                float weight_90 = approx_exp2(lse_1[26] - lse_max_1[0]);
                                if (bound[0] > 26) {
                                    den_1[0] = den_1[0] + weight_90;
                                    acc0_1[0] = __fmaf_rn(weight_90, p0_1[26], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_90, p1_1[26], acc1_1[0]);
                                } else if (bound[1] > 26) {
                                    den_1[1] = den_1[1] + weight_90;
                                    acc0_1[1] = __fmaf_rn(weight_90, p0_1[26], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_90, p1_1[26], acc1_1[1]);
                                } else {
                                    if (bound[2] > 26) {
                                        den_1[2] = den_1[2] + weight_90;
                                        acc0_1[2] = __fmaf_rn(weight_90, p0_1[26], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_90, p1_1[26], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_90;
                                        acc0_1[3] = __fmaf_rn(weight_90, p0_1[26], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_90, p1_1[26], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_26 = num_splits > 27 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_26) {
                                float weight_91 = approx_exp2(lse_1[27] - lse_max_1[0]);
                                if (bound[0] > 27) {
                                    den_1[0] = den_1[0] + weight_91;
                                    acc0_1[0] = __fmaf_rn(weight_91, p0_1[27], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_91, p1_1[27], acc1_1[0]);
                                } else if (bound[1] > 27) {
                                    den_1[1] = den_1[1] + weight_91;
                                    acc0_1[1] = __fmaf_rn(weight_91, p0_1[27], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_91, p1_1[27], acc1_1[1]);
                                } else {
                                    if (bound[2] > 27) {
                                        den_1[2] = den_1[2] + weight_91;
                                        acc0_1[2] = __fmaf_rn(weight_91, p0_1[27], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_91, p1_1[27], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_91;
                                        acc0_1[3] = __fmaf_rn(weight_91, p0_1[27], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_91, p1_1[27], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_27 = num_splits > 28 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_27) {
                                float weight_92 = approx_exp2(lse_1[28] - lse_max_1[0]);
                                if (bound[0] > 28) {
                                    den_1[0] = den_1[0] + weight_92;
                                    acc0_1[0] = __fmaf_rn(weight_92, p0_1[28], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_92, p1_1[28], acc1_1[0]);
                                } else if (bound[1] > 28) {
                                    den_1[1] = den_1[1] + weight_92;
                                    acc0_1[1] = __fmaf_rn(weight_92, p0_1[28], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_92, p1_1[28], acc1_1[1]);
                                } else {
                                    if (bound[2] > 28) {
                                        den_1[2] = den_1[2] + weight_92;
                                        acc0_1[2] = __fmaf_rn(weight_92, p0_1[28], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_92, p1_1[28], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_92;
                                        acc0_1[3] = __fmaf_rn(weight_92, p0_1[28], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_92, p1_1[28], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_28 = num_splits > 29 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_28) {
                                float weight_93 = approx_exp2(lse_1[29] - lse_max_1[0]);
                                if (bound[0] > 29) {
                                    den_1[0] = den_1[0] + weight_93;
                                    acc0_1[0] = __fmaf_rn(weight_93, p0_1[29], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_93, p1_1[29], acc1_1[0]);
                                } else if (bound[1] > 29) {
                                    den_1[1] = den_1[1] + weight_93;
                                    acc0_1[1] = __fmaf_rn(weight_93, p0_1[29], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_93, p1_1[29], acc1_1[1]);
                                } else {
                                    if (bound[2] > 29) {
                                        den_1[2] = den_1[2] + weight_93;
                                        acc0_1[2] = __fmaf_rn(weight_93, p0_1[29], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_93, p1_1[29], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_93;
                                        acc0_1[3] = __fmaf_rn(weight_93, p0_1[29], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_93, p1_1[29], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_29 = num_splits > 30 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_29) {
                                float weight_94 = approx_exp2(lse_1[30] - lse_max_1[0]);
                                if (bound[0] > 30) {
                                    den_1[0] = den_1[0] + weight_94;
                                    acc0_1[0] = __fmaf_rn(weight_94, p0_1[30], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_94, p1_1[30], acc1_1[0]);
                                } else if (bound[1] > 30) {
                                    den_1[1] = den_1[1] + weight_94;
                                    acc0_1[1] = __fmaf_rn(weight_94, p0_1[30], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_94, p1_1[30], acc1_1[1]);
                                } else {
                                    if (bound[2] > 30) {
                                        den_1[2] = den_1[2] + weight_94;
                                        acc0_1[2] = __fmaf_rn(weight_94, p0_1[30], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_94, p1_1[30], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_94;
                                        acc0_1[3] = __fmaf_rn(weight_94, p0_1[30], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_94, p1_1[30], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_30 = num_splits > 31 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_30) {
                                float weight_95 = approx_exp2(lse_1[31] - lse_max_1[0]);
                                if (bound[0] > 31) {
                                    den_1[0] = den_1[0] + weight_95;
                                    acc0_1[0] = __fmaf_rn(weight_95, p0_1[31], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_95, p1_1[31], acc1_1[0]);
                                } else if (bound[1] > 31) {
                                    den_1[1] = den_1[1] + weight_95;
                                    acc0_1[1] = __fmaf_rn(weight_95, p0_1[31], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_95, p1_1[31], acc1_1[1]);
                                } else {
                                    if (bound[2] > 31) {
                                        den_1[2] = den_1[2] + weight_95;
                                        acc0_1[2] = __fmaf_rn(weight_95, p0_1[31], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_95, p1_1[31], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_95;
                                        acc0_1[3] = __fmaf_rn(weight_95, p0_1[31], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_95, p1_1[31], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_31 = num_splits > 32 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_31) {
                                float weight_96 = approx_exp2(lse_1[32] - lse_max_1[0]);
                                if (bound[0] > 32) {
                                    den_1[0] = den_1[0] + weight_96;
                                    acc0_1[0] = __fmaf_rn(weight_96, p0_1[32], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_96, p1_1[32], acc1_1[0]);
                                } else if (bound[1] > 32) {
                                    den_1[1] = den_1[1] + weight_96;
                                    acc0_1[1] = __fmaf_rn(weight_96, p0_1[32], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_96, p1_1[32], acc1_1[1]);
                                } else {
                                    if (bound[2] > 32) {
                                        den_1[2] = den_1[2] + weight_96;
                                        acc0_1[2] = __fmaf_rn(weight_96, p0_1[32], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_96, p1_1[32], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_96;
                                        acc0_1[3] = __fmaf_rn(weight_96, p0_1[32], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_96, p1_1[32], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_32 = num_splits > 33 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_32) {
                                float weight_97 = approx_exp2(lse_1[33] - lse_max_1[0]);
                                if (bound[0] > 33) {
                                    den_1[0] = den_1[0] + weight_97;
                                    acc0_1[0] = __fmaf_rn(weight_97, p0_1[33], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_97, p1_1[33], acc1_1[0]);
                                } else if (bound[1] > 33) {
                                    den_1[1] = den_1[1] + weight_97;
                                    acc0_1[1] = __fmaf_rn(weight_97, p0_1[33], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_97, p1_1[33], acc1_1[1]);
                                } else {
                                    if (bound[2] > 33) {
                                        den_1[2] = den_1[2] + weight_97;
                                        acc0_1[2] = __fmaf_rn(weight_97, p0_1[33], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_97, p1_1[33], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_97;
                                        acc0_1[3] = __fmaf_rn(weight_97, p0_1[33], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_97, p1_1[33], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_33 = num_splits > 34 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_33) {
                                float weight_98 = approx_exp2(lse_1[34] - lse_max_1[0]);
                                if (bound[0] > 34) {
                                    den_1[0] = den_1[0] + weight_98;
                                    acc0_1[0] = __fmaf_rn(weight_98, p0_1[34], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_98, p1_1[34], acc1_1[0]);
                                } else if (bound[1] > 34) {
                                    den_1[1] = den_1[1] + weight_98;
                                    acc0_1[1] = __fmaf_rn(weight_98, p0_1[34], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_98, p1_1[34], acc1_1[1]);
                                } else {
                                    if (bound[2] > 34) {
                                        den_1[2] = den_1[2] + weight_98;
                                        acc0_1[2] = __fmaf_rn(weight_98, p0_1[34], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_98, p1_1[34], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_98;
                                        acc0_1[3] = __fmaf_rn(weight_98, p0_1[34], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_98, p1_1[34], acc1_1[3]);
                                    }
                                }
                            }
                            bool live_34 = num_splits > 35 && lse_max_1[0] > -CUDART_INF_F;
                            if (live_34) {
                                float weight_99 = approx_exp2(lse_1[35] - lse_max_1[0]);
                                if (bound[0] > 35) {
                                    den_1[0] = den_1[0] + weight_99;
                                    acc0_1[0] = __fmaf_rn(weight_99, p0_1[35], acc0_1[0]);
                                    acc1_1[0] = __fmaf_rn(weight_99, p1_1[35], acc1_1[0]);
                                } else if (bound[1] > 35) {
                                    den_1[1] = den_1[1] + weight_99;
                                    acc0_1[1] = __fmaf_rn(weight_99, p0_1[35], acc0_1[1]);
                                    acc1_1[1] = __fmaf_rn(weight_99, p1_1[35], acc1_1[1]);
                                } else {
                                    if (bound[2] > 35) {
                                        den_1[2] = den_1[2] + weight_99;
                                        acc0_1[2] = __fmaf_rn(weight_99, p0_1[35], acc0_1[2]);
                                        acc1_1[2] = __fmaf_rn(weight_99, p1_1[35], acc1_1[2]);
                                    } else {
                                        den_1[3] = den_1[3] + weight_99;
                                        acc0_1[3] = __fmaf_rn(weight_99, p0_1[35], acc0_1[3]);
                                        acc1_1[3] = __fmaf_rn(weight_99, p1_1[35], acc1_1[3]);
                                    }
                                }
                            }
                            if (valid_1[0] != 0) {
                                float den_k_1 = den_1[0];
                                float acc0_k_1 = acc0_1[0];
                                float acc1_k_1 = acc1_1[0];
                                den_k_1 = den_k_1 + den_1[1];
                                acc0_k_1 = acc0_k_1 + acc0_1[1];
                                acc1_k_1 = acc1_k_1 + acc1_1[1];
                                den_k_1 = den_k_1 + den_1[2];
                                acc0_k_1 = acc0_k_1 + acc0_1[2];
                                acc1_k_1 = acc1_k_1 + acc1_1[2];
                                den_k_1 = den_k_1 + den_1[3];
                                acc0_k_1 = acc0_k_1 + acc0_1[3];
                                acc1_k_1 = acc1_k_1 + acc1_1[3];
                                float inv_3 = 0.0f;
                                if (den_k_1 > 0.0f) {
                                    float _rcp_258 = approx_rcp(den_k_1);
                                    inv_3 = _rcp_258;
                                }
                                float attn0_1 = (float)(__nv_bfloat16)(acc0_k_1 * inv_3);
                                float attn1_1 = (float)(__nv_bfloat16)(acc1_k_1 * inv_3);
                                float _exp2_260 = approx_exp2((-gate0_1[0]) * 1.442695f);
                                float _rcp_259 = approx_rcp(1.0f + _exp2_260);
                                float result0_1 = attn0_1 * _rcp_259;
                                float _exp2_262 = approx_exp2((-gate1_1[0]) * 1.442695f);
                                float _rcp_261 = approx_rcp(1.0f + _exp2_262);
                                float result1_1 = attn1_1 * _rcp_261;
                                __nv_bfloat162 _bit_cast_src_263 = __float22bfloat162_rn(make_float2(result0_1, result1_1));
                                word_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_263)[0];
                                long long out_offset_1 = row_i64 * out_stride_row + qo_head_3[0] * out_stride_head + dim_1[0];
                                *(reinterpret_cast<unsigned int*>(out + out_offset_1) + (0)) = word_1[0];
                            }
                        }
                    } else {
                        if (num_splits > 8) {
                            float lse_2[32];
                            float p0_2[32];
                            float p1_2[32];
                            float acc0_2[8];
                            float acc1_2[8];
                            float den_2[8];
                            int chunk_w_2 = (num_splits + 3) / 4;
                            if (num_splits <= 4) {
                                chunk_w_2 = num_splits;
                            }
                            int bound_1[4];
                            bound_1[0] = chunk_w_2;
                            bound_1[1] = chunk_w_2 * 2;
                            bound_1[2] = chunk_w_2 * 3;
                            bound_1[3] = chunk_w_2 * 4;
                            float lse_max_2[2];
                            int valid_2[2];
                            int tab_2[2];
                            long long dim_2[2];
                            long long qo_head_4[2];
                            long long row_head_2[2];
                            float gate0_2[2];
                            float gate1_2[2];
                            unsigned int word_2[1];
                            long long nqo_2 = (long long)num_qo_heads;
                            #pragma unroll 1
                            for (int p_1_2 = p_first; p_1_2 < end; p_1_2 += 256) {
                                int pk_2 = ((p_1_2) < (end - 1) ? (p_1_2) : (end - 1));
                                valid_2[0] = ((end > p_1_2) ? 1 : 0);
                                int head_4 = pk_2 >> 7;
                                dim_2[0] = (long long)((pk_2 & 127) * 2);
                                qo_head_4[0] = (long long)(first_head + head_4);
                                row_head_2[0] = row_i64 * nqo_2 + qo_head_4[0];
                                acc0_2[0] = 0.0f;
                                acc1_2[0] = 0.0f;
                                den_2[0] = 0.0f;
                                acc0_2[1] = 0.0f;
                                acc1_2[1] = 0.0f;
                                den_2[1] = 0.0f;
                                acc0_2[2] = 0.0f;
                                acc1_2[2] = 0.0f;
                                den_2[2] = 0.0f;
                                acc0_2[3] = 0.0f;
                                acc1_2[3] = 0.0f;
                                den_2[3] = 0.0f;
                                lse_max_2[0] = -CUDART_INF_F;
                                float _vec_load_114[2];
                                {
                                    uint32_t _bf16x2_bits_264;
                                    _bf16x2_bits_264 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2[0] * 256 + dim_2[0]) + 0);
                                    (&_vec_load_114[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_264) << 16);
                                    (&_vec_load_114[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_264) & 0xffff0000u);
                                }
                                gate0_2[0] = _vec_load_114[0];
                                gate1_2[0] = _vec_load_114[1];
                                int pk_0 = ((p_1_2 + 128) < (end - 1) ? (p_1_2 + 128) : (end - 1));
                                valid_2[1] = ((end > p_1_2 + 128) ? 1 : 0);
                                int head_1_1 = pk_0 >> 7;
                                dim_2[1] = (long long)((pk_0 & 127) * 2);
                                qo_head_4[1] = (long long)(first_head + head_1_1);
                                row_head_2[1] = row_i64 * nqo_2 + qo_head_4[1];
                                acc0_2[4] = 0.0f;
                                acc1_2[4] = 0.0f;
                                den_2[4] = 0.0f;
                                acc0_2[5] = 0.0f;
                                acc1_2[5] = 0.0f;
                                den_2[5] = 0.0f;
                                acc0_2[6] = 0.0f;
                                acc1_2[6] = 0.0f;
                                den_2[6] = 0.0f;
                                acc0_2[7] = 0.0f;
                                acc1_2[7] = 0.0f;
                                den_2[7] = 0.0f;
                                lse_max_2[1] = -CUDART_INF_F;
                                float _vec_load_115[2];
                                {
                                    uint32_t _bf16x2_bits_265;
                                    _bf16x2_bits_265 = *reinterpret_cast<const uint32_t*>(gate + (row_head_2[1] * 256 + dim_2[1]) + 0);
                                    (&_vec_load_115[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_265) << 16);
                                    (&_vec_load_115[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_265) & 0xffff0000u);
                                }
                                gate0_2[1] = _vec_load_115[0];
                                gate1_2[1] = _vec_load_115[1];
                                lse_2[0] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_100 = row_head_2[0];
                                    lse_2[0] = partial_lse[part_row_100];
                                    float _vec_load_116[2];
                                    {
                                        float2 _v2_266 = *reinterpret_cast<const float2*>(partial_out + (part_row_100 * 256 + dim_2[0]) + 0);
                                        _vec_load_116[0] = _v2_266.x;
                                        _vec_load_116[0 + 1] = _v2_266.y;
                                    }
                                    p0_2[0] = _vec_load_116[0];
                                    p1_2[0] = _vec_load_116[1];
                                }
                                lse_2[1] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_101 = split_pitch + row_head_2[0];
                                    lse_2[1] = partial_lse[part_row_101];
                                    float _vec_load_117[2];
                                    {
                                        float2 _v2_267 = *reinterpret_cast<const float2*>(partial_out + (part_row_101 * 256 + dim_2[0]) + 0);
                                        _vec_load_117[0] = _v2_267.x;
                                        _vec_load_117[0 + 1] = _v2_267.y;
                                    }
                                    p0_2[1] = _vec_load_117[0];
                                    p1_2[1] = _vec_load_117[1];
                                }
                                lse_2[2] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_102 = 2 * split_pitch + row_head_2[0];
                                    lse_2[2] = partial_lse[part_row_102];
                                    float _vec_load_118[2];
                                    {
                                        float2 _v2_268 = *reinterpret_cast<const float2*>(partial_out + (part_row_102 * 256 + dim_2[0]) + 0);
                                        _vec_load_118[0] = _v2_268.x;
                                        _vec_load_118[0 + 1] = _v2_268.y;
                                    }
                                    p0_2[2] = _vec_load_118[0];
                                    p1_2[2] = _vec_load_118[1];
                                }
                                lse_2[3] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_103 = 3 * split_pitch + row_head_2[0];
                                    lse_2[3] = partial_lse[part_row_103];
                                    float _vec_load_119[2];
                                    {
                                        float2 _v2_269 = *reinterpret_cast<const float2*>(partial_out + (part_row_103 * 256 + dim_2[0]) + 0);
                                        _vec_load_119[0] = _v2_269.x;
                                        _vec_load_119[0 + 1] = _v2_269.y;
                                    }
                                    p0_2[3] = _vec_load_119[0];
                                    p1_2[3] = _vec_load_119[1];
                                }
                                lse_2[4] = -CUDART_INF_F;
                                if (num_splits > 4) {
                                    long long part_row_104 = 4 * split_pitch + row_head_2[0];
                                    lse_2[4] = partial_lse[part_row_104];
                                    float _vec_load_120[2];
                                    {
                                        float2 _v2_270 = *reinterpret_cast<const float2*>(partial_out + (part_row_104 * 256 + dim_2[0]) + 0);
                                        _vec_load_120[0] = _v2_270.x;
                                        _vec_load_120[0 + 1] = _v2_270.y;
                                    }
                                    p0_2[4] = _vec_load_120[0];
                                    p1_2[4] = _vec_load_120[1];
                                }
                                lse_2[5] = -CUDART_INF_F;
                                if (num_splits > 5) {
                                    long long part_row_105 = 5 * split_pitch + row_head_2[0];
                                    lse_2[5] = partial_lse[part_row_105];
                                    float _vec_load_121[2];
                                    {
                                        float2 _v2_271 = *reinterpret_cast<const float2*>(partial_out + (part_row_105 * 256 + dim_2[0]) + 0);
                                        _vec_load_121[0] = _v2_271.x;
                                        _vec_load_121[0 + 1] = _v2_271.y;
                                    }
                                    p0_2[5] = _vec_load_121[0];
                                    p1_2[5] = _vec_load_121[1];
                                }
                                lse_2[6] = -CUDART_INF_F;
                                if (num_splits > 6) {
                                    long long part_row_106 = 6 * split_pitch + row_head_2[0];
                                    lse_2[6] = partial_lse[part_row_106];
                                    float _vec_load_122[2];
                                    {
                                        float2 _v2_272 = *reinterpret_cast<const float2*>(partial_out + (part_row_106 * 256 + dim_2[0]) + 0);
                                        _vec_load_122[0] = _v2_272.x;
                                        _vec_load_122[0 + 1] = _v2_272.y;
                                    }
                                    p0_2[6] = _vec_load_122[0];
                                    p1_2[6] = _vec_load_122[1];
                                }
                                lse_2[7] = -CUDART_INF_F;
                                if (num_splits > 7) {
                                    long long part_row_107 = 7 * split_pitch + row_head_2[0];
                                    lse_2[7] = partial_lse[part_row_107];
                                    float _vec_load_123[2];
                                    {
                                        float2 _v2_273 = *reinterpret_cast<const float2*>(partial_out + (part_row_107 * 256 + dim_2[0]) + 0);
                                        _vec_load_123[0] = _v2_273.x;
                                        _vec_load_123[0 + 1] = _v2_273.y;
                                    }
                                    p0_2[7] = _vec_load_123[0];
                                    p1_2[7] = _vec_load_123[1];
                                }
                                lse_2[8] = -CUDART_INF_F;
                                if (num_splits > 8) {
                                    long long part_row_108 = 8 * split_pitch + row_head_2[0];
                                    lse_2[8] = partial_lse[part_row_108];
                                    float _vec_load_124[2];
                                    {
                                        float2 _v2_274 = *reinterpret_cast<const float2*>(partial_out + (part_row_108 * 256 + dim_2[0]) + 0);
                                        _vec_load_124[0] = _v2_274.x;
                                        _vec_load_124[0 + 1] = _v2_274.y;
                                    }
                                    p0_2[8] = _vec_load_124[0];
                                    p1_2[8] = _vec_load_124[1];
                                }
                                lse_2[9] = -CUDART_INF_F;
                                if (num_splits > 9) {
                                    long long part_row_109 = 9 * split_pitch + row_head_2[0];
                                    lse_2[9] = partial_lse[part_row_109];
                                    float _vec_load_125[2];
                                    {
                                        float2 _v2_275 = *reinterpret_cast<const float2*>(partial_out + (part_row_109 * 256 + dim_2[0]) + 0);
                                        _vec_load_125[0] = _v2_275.x;
                                        _vec_load_125[0 + 1] = _v2_275.y;
                                    }
                                    p0_2[9] = _vec_load_125[0];
                                    p1_2[9] = _vec_load_125[1];
                                }
                                lse_2[10] = -CUDART_INF_F;
                                if (num_splits > 10) {
                                    long long part_row_110 = 10 * split_pitch + row_head_2[0];
                                    lse_2[10] = partial_lse[part_row_110];
                                    float _vec_load_126[2];
                                    {
                                        float2 _v2_276 = *reinterpret_cast<const float2*>(partial_out + (part_row_110 * 256 + dim_2[0]) + 0);
                                        _vec_load_126[0] = _v2_276.x;
                                        _vec_load_126[0 + 1] = _v2_276.y;
                                    }
                                    p0_2[10] = _vec_load_126[0];
                                    p1_2[10] = _vec_load_126[1];
                                }
                                lse_2[11] = -CUDART_INF_F;
                                if (num_splits > 11) {
                                    long long part_row_111 = 11 * split_pitch + row_head_2[0];
                                    lse_2[11] = partial_lse[part_row_111];
                                    float _vec_load_127[2];
                                    {
                                        float2 _v2_277 = *reinterpret_cast<const float2*>(partial_out + (part_row_111 * 256 + dim_2[0]) + 0);
                                        _vec_load_127[0] = _v2_277.x;
                                        _vec_load_127[0 + 1] = _v2_277.y;
                                    }
                                    p0_2[11] = _vec_load_127[0];
                                    p1_2[11] = _vec_load_127[1];
                                }
                                lse_2[12] = -CUDART_INF_F;
                                if (num_splits > 12) {
                                    long long part_row_112 = 12 * split_pitch + row_head_2[0];
                                    lse_2[12] = partial_lse[part_row_112];
                                    float _vec_load_128[2];
                                    {
                                        float2 _v2_278 = *reinterpret_cast<const float2*>(partial_out + (part_row_112 * 256 + dim_2[0]) + 0);
                                        _vec_load_128[0] = _v2_278.x;
                                        _vec_load_128[0 + 1] = _v2_278.y;
                                    }
                                    p0_2[12] = _vec_load_128[0];
                                    p1_2[12] = _vec_load_128[1];
                                }
                                lse_2[13] = -CUDART_INF_F;
                                if (num_splits > 13) {
                                    long long part_row_113 = 13 * split_pitch + row_head_2[0];
                                    lse_2[13] = partial_lse[part_row_113];
                                    float _vec_load_129[2];
                                    {
                                        float2 _v2_279 = *reinterpret_cast<const float2*>(partial_out + (part_row_113 * 256 + dim_2[0]) + 0);
                                        _vec_load_129[0] = _v2_279.x;
                                        _vec_load_129[0 + 1] = _v2_279.y;
                                    }
                                    p0_2[13] = _vec_load_129[0];
                                    p1_2[13] = _vec_load_129[1];
                                }
                                lse_2[14] = -CUDART_INF_F;
                                if (num_splits > 14) {
                                    long long part_row_114 = 14 * split_pitch + row_head_2[0];
                                    lse_2[14] = partial_lse[part_row_114];
                                    float _vec_load_130[2];
                                    {
                                        float2 _v2_280 = *reinterpret_cast<const float2*>(partial_out + (part_row_114 * 256 + dim_2[0]) + 0);
                                        _vec_load_130[0] = _v2_280.x;
                                        _vec_load_130[0 + 1] = _v2_280.y;
                                    }
                                    p0_2[14] = _vec_load_130[0];
                                    p1_2[14] = _vec_load_130[1];
                                }
                                lse_2[15] = -CUDART_INF_F;
                                if (num_splits > 15) {
                                    long long part_row_115 = 15 * split_pitch + row_head_2[0];
                                    lse_2[15] = partial_lse[part_row_115];
                                    float _vec_load_131[2];
                                    {
                                        float2 _v2_281 = *reinterpret_cast<const float2*>(partial_out + (part_row_115 * 256 + dim_2[0]) + 0);
                                        _vec_load_131[0] = _v2_281.x;
                                        _vec_load_131[0 + 1] = _v2_281.y;
                                    }
                                    p0_2[15] = _vec_load_131[0];
                                    p1_2[15] = _vec_load_131[1];
                                }
                                lse_2[16] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_116 = row_head_2[1];
                                    lse_2[16] = partial_lse[part_row_116];
                                    float _vec_load_132[2];
                                    {
                                        float2 _v2_282 = *reinterpret_cast<const float2*>(partial_out + (part_row_116 * 256 + dim_2[1]) + 0);
                                        _vec_load_132[0] = _v2_282.x;
                                        _vec_load_132[0 + 1] = _v2_282.y;
                                    }
                                    p0_2[16] = _vec_load_132[0];
                                    p1_2[16] = _vec_load_132[1];
                                }
                                lse_2[17] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_117 = split_pitch + row_head_2[1];
                                    lse_2[17] = partial_lse[part_row_117];
                                    float _vec_load_133[2];
                                    {
                                        float2 _v2_283 = *reinterpret_cast<const float2*>(partial_out + (part_row_117 * 256 + dim_2[1]) + 0);
                                        _vec_load_133[0] = _v2_283.x;
                                        _vec_load_133[0 + 1] = _v2_283.y;
                                    }
                                    p0_2[17] = _vec_load_133[0];
                                    p1_2[17] = _vec_load_133[1];
                                }
                                lse_2[18] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_118 = 2 * split_pitch + row_head_2[1];
                                    lse_2[18] = partial_lse[part_row_118];
                                    float _vec_load_134[2];
                                    {
                                        float2 _v2_284 = *reinterpret_cast<const float2*>(partial_out + (part_row_118 * 256 + dim_2[1]) + 0);
                                        _vec_load_134[0] = _v2_284.x;
                                        _vec_load_134[0 + 1] = _v2_284.y;
                                    }
                                    p0_2[18] = _vec_load_134[0];
                                    p1_2[18] = _vec_load_134[1];
                                }
                                lse_2[19] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_119 = 3 * split_pitch + row_head_2[1];
                                    lse_2[19] = partial_lse[part_row_119];
                                    float _vec_load_135[2];
                                    {
                                        float2 _v2_285 = *reinterpret_cast<const float2*>(partial_out + (part_row_119 * 256 + dim_2[1]) + 0);
                                        _vec_load_135[0] = _v2_285.x;
                                        _vec_load_135[0 + 1] = _v2_285.y;
                                    }
                                    p0_2[19] = _vec_load_135[0];
                                    p1_2[19] = _vec_load_135[1];
                                }
                                lse_2[20] = -CUDART_INF_F;
                                if (num_splits > 4) {
                                    long long part_row_120 = 4 * split_pitch + row_head_2[1];
                                    lse_2[20] = partial_lse[part_row_120];
                                    float _vec_load_136[2];
                                    {
                                        float2 _v2_286 = *reinterpret_cast<const float2*>(partial_out + (part_row_120 * 256 + dim_2[1]) + 0);
                                        _vec_load_136[0] = _v2_286.x;
                                        _vec_load_136[0 + 1] = _v2_286.y;
                                    }
                                    p0_2[20] = _vec_load_136[0];
                                    p1_2[20] = _vec_load_136[1];
                                }
                                lse_2[21] = -CUDART_INF_F;
                                if (num_splits > 5) {
                                    long long part_row_121 = 5 * split_pitch + row_head_2[1];
                                    lse_2[21] = partial_lse[part_row_121];
                                    float _vec_load_137[2];
                                    {
                                        float2 _v2_287 = *reinterpret_cast<const float2*>(partial_out + (part_row_121 * 256 + dim_2[1]) + 0);
                                        _vec_load_137[0] = _v2_287.x;
                                        _vec_load_137[0 + 1] = _v2_287.y;
                                    }
                                    p0_2[21] = _vec_load_137[0];
                                    p1_2[21] = _vec_load_137[1];
                                }
                                lse_2[22] = -CUDART_INF_F;
                                if (num_splits > 6) {
                                    long long part_row_122 = 6 * split_pitch + row_head_2[1];
                                    lse_2[22] = partial_lse[part_row_122];
                                    float _vec_load_138[2];
                                    {
                                        float2 _v2_288 = *reinterpret_cast<const float2*>(partial_out + (part_row_122 * 256 + dim_2[1]) + 0);
                                        _vec_load_138[0] = _v2_288.x;
                                        _vec_load_138[0 + 1] = _v2_288.y;
                                    }
                                    p0_2[22] = _vec_load_138[0];
                                    p1_2[22] = _vec_load_138[1];
                                }
                                lse_2[23] = -CUDART_INF_F;
                                if (num_splits > 7) {
                                    long long part_row_123 = 7 * split_pitch + row_head_2[1];
                                    lse_2[23] = partial_lse[part_row_123];
                                    float _vec_load_139[2];
                                    {
                                        float2 _v2_289 = *reinterpret_cast<const float2*>(partial_out + (part_row_123 * 256 + dim_2[1]) + 0);
                                        _vec_load_139[0] = _v2_289.x;
                                        _vec_load_139[0 + 1] = _v2_289.y;
                                    }
                                    p0_2[23] = _vec_load_139[0];
                                    p1_2[23] = _vec_load_139[1];
                                }
                                lse_2[24] = -CUDART_INF_F;
                                if (num_splits > 8) {
                                    long long part_row_124 = 8 * split_pitch + row_head_2[1];
                                    lse_2[24] = partial_lse[part_row_124];
                                    float _vec_load_140[2];
                                    {
                                        float2 _v2_290 = *reinterpret_cast<const float2*>(partial_out + (part_row_124 * 256 + dim_2[1]) + 0);
                                        _vec_load_140[0] = _v2_290.x;
                                        _vec_load_140[0 + 1] = _v2_290.y;
                                    }
                                    p0_2[24] = _vec_load_140[0];
                                    p1_2[24] = _vec_load_140[1];
                                }
                                lse_2[25] = -CUDART_INF_F;
                                if (num_splits > 9) {
                                    long long part_row_125 = 9 * split_pitch + row_head_2[1];
                                    lse_2[25] = partial_lse[part_row_125];
                                    float _vec_load_141[2];
                                    {
                                        float2 _v2_291 = *reinterpret_cast<const float2*>(partial_out + (part_row_125 * 256 + dim_2[1]) + 0);
                                        _vec_load_141[0] = _v2_291.x;
                                        _vec_load_141[0 + 1] = _v2_291.y;
                                    }
                                    p0_2[25] = _vec_load_141[0];
                                    p1_2[25] = _vec_load_141[1];
                                }
                                lse_2[26] = -CUDART_INF_F;
                                if (num_splits > 10) {
                                    long long part_row_126 = 10 * split_pitch + row_head_2[1];
                                    lse_2[26] = partial_lse[part_row_126];
                                    float _vec_load_142[2];
                                    {
                                        float2 _v2_292 = *reinterpret_cast<const float2*>(partial_out + (part_row_126 * 256 + dim_2[1]) + 0);
                                        _vec_load_142[0] = _v2_292.x;
                                        _vec_load_142[0 + 1] = _v2_292.y;
                                    }
                                    p0_2[26] = _vec_load_142[0];
                                    p1_2[26] = _vec_load_142[1];
                                }
                                lse_2[27] = -CUDART_INF_F;
                                if (num_splits > 11) {
                                    long long part_row_127 = 11 * split_pitch + row_head_2[1];
                                    lse_2[27] = partial_lse[part_row_127];
                                    float _vec_load_143[2];
                                    {
                                        float2 _v2_293 = *reinterpret_cast<const float2*>(partial_out + (part_row_127 * 256 + dim_2[1]) + 0);
                                        _vec_load_143[0] = _v2_293.x;
                                        _vec_load_143[0 + 1] = _v2_293.y;
                                    }
                                    p0_2[27] = _vec_load_143[0];
                                    p1_2[27] = _vec_load_143[1];
                                }
                                lse_2[28] = -CUDART_INF_F;
                                if (num_splits > 12) {
                                    long long part_row_128 = 12 * split_pitch + row_head_2[1];
                                    lse_2[28] = partial_lse[part_row_128];
                                    float _vec_load_144[2];
                                    {
                                        float2 _v2_294 = *reinterpret_cast<const float2*>(partial_out + (part_row_128 * 256 + dim_2[1]) + 0);
                                        _vec_load_144[0] = _v2_294.x;
                                        _vec_load_144[0 + 1] = _v2_294.y;
                                    }
                                    p0_2[28] = _vec_load_144[0];
                                    p1_2[28] = _vec_load_144[1];
                                }
                                lse_2[29] = -CUDART_INF_F;
                                if (num_splits > 13) {
                                    long long part_row_129 = 13 * split_pitch + row_head_2[1];
                                    lse_2[29] = partial_lse[part_row_129];
                                    float _vec_load_145[2];
                                    {
                                        float2 _v2_295 = *reinterpret_cast<const float2*>(partial_out + (part_row_129 * 256 + dim_2[1]) + 0);
                                        _vec_load_145[0] = _v2_295.x;
                                        _vec_load_145[0 + 1] = _v2_295.y;
                                    }
                                    p0_2[29] = _vec_load_145[0];
                                    p1_2[29] = _vec_load_145[1];
                                }
                                lse_2[30] = -CUDART_INF_F;
                                if (num_splits > 14) {
                                    long long part_row_130 = 14 * split_pitch + row_head_2[1];
                                    lse_2[30] = partial_lse[part_row_130];
                                    float _vec_load_146[2];
                                    {
                                        float2 _v2_296 = *reinterpret_cast<const float2*>(partial_out + (part_row_130 * 256 + dim_2[1]) + 0);
                                        _vec_load_146[0] = _v2_296.x;
                                        _vec_load_146[0 + 1] = _v2_296.y;
                                    }
                                    p0_2[30] = _vec_load_146[0];
                                    p1_2[30] = _vec_load_146[1];
                                }
                                lse_2[31] = -CUDART_INF_F;
                                if (num_splits > 15) {
                                    long long part_row_131 = 15 * split_pitch + row_head_2[1];
                                    lse_2[31] = partial_lse[part_row_131];
                                    float _vec_load_147[2];
                                    {
                                        float2 _v2_297 = *reinterpret_cast<const float2*>(partial_out + (part_row_131 * 256 + dim_2[1]) + 0);
                                        _vec_load_147[0] = _v2_297.x;
                                        _vec_load_147[0 + 1] = _v2_297.y;
                                    }
                                    p0_2[31] = _vec_load_147[0];
                                    p1_2[31] = _vec_load_147[1];
                                }
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[0]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[1]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[2]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[3]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[4]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[5]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[6]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[7]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[8]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[9]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[10]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[11]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[12]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[13]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[14]);
                                lse_max_2[0] = fmaxf(lse_max_2[0], lse_2[15]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[16]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[17]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[18]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[19]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[20]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[21]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[22]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[23]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[24]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[25]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[26]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[27]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[28]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[29]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[30]);
                                lse_max_2[1] = fmaxf(lse_max_2[1], lse_2[31]);
                                bool live_35 = num_splits > 0 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_35) {
                                    float weight_100 = approx_exp2(lse_2[0] - lse_max_2[0]);
                                    if (bound_1[0] > 0) {
                                        den_2[0] = den_2[0] + weight_100;
                                        acc0_2[0] = __fmaf_rn(weight_100, p0_2[0], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_100, p1_2[0], acc1_2[0]);
                                    } else if (bound_1[1] > 0) {
                                        den_2[1] = den_2[1] + weight_100;
                                        acc0_2[1] = __fmaf_rn(weight_100, p0_2[0], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_100, p1_2[0], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 0) {
                                            den_2[2] = den_2[2] + weight_100;
                                            acc0_2[2] = __fmaf_rn(weight_100, p0_2[0], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_100, p1_2[0], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_100;
                                            acc0_2[3] = __fmaf_rn(weight_100, p0_2[0], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_100, p1_2[0], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_2_5 = num_splits > 1 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_2_5) {
                                    float weight_101 = approx_exp2(lse_2[1] - lse_max_2[0]);
                                    if (bound_1[0] > 1) {
                                        den_2[0] = den_2[0] + weight_101;
                                        acc0_2[0] = __fmaf_rn(weight_101, p0_2[1], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_101, p1_2[1], acc1_2[0]);
                                    } else if (bound_1[1] > 1) {
                                        den_2[1] = den_2[1] + weight_101;
                                        acc0_2[1] = __fmaf_rn(weight_101, p0_2[1], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_101, p1_2[1], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 1) {
                                            den_2[2] = den_2[2] + weight_101;
                                            acc0_2[2] = __fmaf_rn(weight_101, p0_2[1], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_101, p1_2[1], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_101;
                                            acc0_2[3] = __fmaf_rn(weight_101, p0_2[1], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_101, p1_2[1], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_3_5 = num_splits > 2 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_3_5) {
                                    float weight_102 = approx_exp2(lse_2[2] - lse_max_2[0]);
                                    if (bound_1[0] > 2) {
                                        den_2[0] = den_2[0] + weight_102;
                                        acc0_2[0] = __fmaf_rn(weight_102, p0_2[2], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_102, p1_2[2], acc1_2[0]);
                                    } else if (bound_1[1] > 2) {
                                        den_2[1] = den_2[1] + weight_102;
                                        acc0_2[1] = __fmaf_rn(weight_102, p0_2[2], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_102, p1_2[2], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 2) {
                                            den_2[2] = den_2[2] + weight_102;
                                            acc0_2[2] = __fmaf_rn(weight_102, p0_2[2], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_102, p1_2[2], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_102;
                                            acc0_2[3] = __fmaf_rn(weight_102, p0_2[2], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_102, p1_2[2], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_4_5 = num_splits > 3 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_4_5) {
                                    float weight_103 = approx_exp2(lse_2[3] - lse_max_2[0]);
                                    if (bound_1[0] > 3) {
                                        den_2[0] = den_2[0] + weight_103;
                                        acc0_2[0] = __fmaf_rn(weight_103, p0_2[3], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_103, p1_2[3], acc1_2[0]);
                                    } else if (bound_1[1] > 3) {
                                        den_2[1] = den_2[1] + weight_103;
                                        acc0_2[1] = __fmaf_rn(weight_103, p0_2[3], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_103, p1_2[3], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 3) {
                                            den_2[2] = den_2[2] + weight_103;
                                            acc0_2[2] = __fmaf_rn(weight_103, p0_2[3], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_103, p1_2[3], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_103;
                                            acc0_2[3] = __fmaf_rn(weight_103, p0_2[3], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_103, p1_2[3], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_5_5 = num_splits > 4 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_5_5) {
                                    float weight_104 = approx_exp2(lse_2[4] - lse_max_2[0]);
                                    if (bound_1[0] > 4) {
                                        den_2[0] = den_2[0] + weight_104;
                                        acc0_2[0] = __fmaf_rn(weight_104, p0_2[4], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_104, p1_2[4], acc1_2[0]);
                                    } else if (bound_1[1] > 4) {
                                        den_2[1] = den_2[1] + weight_104;
                                        acc0_2[1] = __fmaf_rn(weight_104, p0_2[4], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_104, p1_2[4], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 4) {
                                            den_2[2] = den_2[2] + weight_104;
                                            acc0_2[2] = __fmaf_rn(weight_104, p0_2[4], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_104, p1_2[4], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_104;
                                            acc0_2[3] = __fmaf_rn(weight_104, p0_2[4], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_104, p1_2[4], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_6_5 = num_splits > 5 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_6_5) {
                                    float weight_105 = approx_exp2(lse_2[5] - lse_max_2[0]);
                                    if (bound_1[0] > 5) {
                                        den_2[0] = den_2[0] + weight_105;
                                        acc0_2[0] = __fmaf_rn(weight_105, p0_2[5], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_105, p1_2[5], acc1_2[0]);
                                    } else if (bound_1[1] > 5) {
                                        den_2[1] = den_2[1] + weight_105;
                                        acc0_2[1] = __fmaf_rn(weight_105, p0_2[5], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_105, p1_2[5], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 5) {
                                            den_2[2] = den_2[2] + weight_105;
                                            acc0_2[2] = __fmaf_rn(weight_105, p0_2[5], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_105, p1_2[5], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_105;
                                            acc0_2[3] = __fmaf_rn(weight_105, p0_2[5], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_105, p1_2[5], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_7_5 = num_splits > 6 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_7_5) {
                                    float weight_106 = approx_exp2(lse_2[6] - lse_max_2[0]);
                                    if (bound_1[0] > 6) {
                                        den_2[0] = den_2[0] + weight_106;
                                        acc0_2[0] = __fmaf_rn(weight_106, p0_2[6], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_106, p1_2[6], acc1_2[0]);
                                    } else if (bound_1[1] > 6) {
                                        den_2[1] = den_2[1] + weight_106;
                                        acc0_2[1] = __fmaf_rn(weight_106, p0_2[6], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_106, p1_2[6], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 6) {
                                            den_2[2] = den_2[2] + weight_106;
                                            acc0_2[2] = __fmaf_rn(weight_106, p0_2[6], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_106, p1_2[6], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_106;
                                            acc0_2[3] = __fmaf_rn(weight_106, p0_2[6], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_106, p1_2[6], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_8_5 = num_splits > 7 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_8_5) {
                                    float weight_107 = approx_exp2(lse_2[7] - lse_max_2[0]);
                                    if (bound_1[0] > 7) {
                                        den_2[0] = den_2[0] + weight_107;
                                        acc0_2[0] = __fmaf_rn(weight_107, p0_2[7], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_107, p1_2[7], acc1_2[0]);
                                    } else if (bound_1[1] > 7) {
                                        den_2[1] = den_2[1] + weight_107;
                                        acc0_2[1] = __fmaf_rn(weight_107, p0_2[7], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_107, p1_2[7], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 7) {
                                            den_2[2] = den_2[2] + weight_107;
                                            acc0_2[2] = __fmaf_rn(weight_107, p0_2[7], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_107, p1_2[7], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_107;
                                            acc0_2[3] = __fmaf_rn(weight_107, p0_2[7], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_107, p1_2[7], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_9_5 = num_splits > 8 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_9_5) {
                                    float weight_108 = approx_exp2(lse_2[8] - lse_max_2[0]);
                                    if (bound_1[0] > 8) {
                                        den_2[0] = den_2[0] + weight_108;
                                        acc0_2[0] = __fmaf_rn(weight_108, p0_2[8], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_108, p1_2[8], acc1_2[0]);
                                    } else if (bound_1[1] > 8) {
                                        den_2[1] = den_2[1] + weight_108;
                                        acc0_2[1] = __fmaf_rn(weight_108, p0_2[8], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_108, p1_2[8], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 8) {
                                            den_2[2] = den_2[2] + weight_108;
                                            acc0_2[2] = __fmaf_rn(weight_108, p0_2[8], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_108, p1_2[8], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_108;
                                            acc0_2[3] = __fmaf_rn(weight_108, p0_2[8], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_108, p1_2[8], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_10_5 = num_splits > 9 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_10_5) {
                                    float weight_109 = approx_exp2(lse_2[9] - lse_max_2[0]);
                                    if (bound_1[0] > 9) {
                                        den_2[0] = den_2[0] + weight_109;
                                        acc0_2[0] = __fmaf_rn(weight_109, p0_2[9], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_109, p1_2[9], acc1_2[0]);
                                    } else if (bound_1[1] > 9) {
                                        den_2[1] = den_2[1] + weight_109;
                                        acc0_2[1] = __fmaf_rn(weight_109, p0_2[9], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_109, p1_2[9], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 9) {
                                            den_2[2] = den_2[2] + weight_109;
                                            acc0_2[2] = __fmaf_rn(weight_109, p0_2[9], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_109, p1_2[9], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_109;
                                            acc0_2[3] = __fmaf_rn(weight_109, p0_2[9], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_109, p1_2[9], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_11_5 = num_splits > 10 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_11_5) {
                                    float weight_110 = approx_exp2(lse_2[10] - lse_max_2[0]);
                                    if (bound_1[0] > 10) {
                                        den_2[0] = den_2[0] + weight_110;
                                        acc0_2[0] = __fmaf_rn(weight_110, p0_2[10], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_110, p1_2[10], acc1_2[0]);
                                    } else if (bound_1[1] > 10) {
                                        den_2[1] = den_2[1] + weight_110;
                                        acc0_2[1] = __fmaf_rn(weight_110, p0_2[10], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_110, p1_2[10], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 10) {
                                            den_2[2] = den_2[2] + weight_110;
                                            acc0_2[2] = __fmaf_rn(weight_110, p0_2[10], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_110, p1_2[10], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_110;
                                            acc0_2[3] = __fmaf_rn(weight_110, p0_2[10], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_110, p1_2[10], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_12_5 = num_splits > 11 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_12_5) {
                                    float weight_111 = approx_exp2(lse_2[11] - lse_max_2[0]);
                                    if (bound_1[0] > 11) {
                                        den_2[0] = den_2[0] + weight_111;
                                        acc0_2[0] = __fmaf_rn(weight_111, p0_2[11], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_111, p1_2[11], acc1_2[0]);
                                    } else if (bound_1[1] > 11) {
                                        den_2[1] = den_2[1] + weight_111;
                                        acc0_2[1] = __fmaf_rn(weight_111, p0_2[11], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_111, p1_2[11], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 11) {
                                            den_2[2] = den_2[2] + weight_111;
                                            acc0_2[2] = __fmaf_rn(weight_111, p0_2[11], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_111, p1_2[11], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_111;
                                            acc0_2[3] = __fmaf_rn(weight_111, p0_2[11], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_111, p1_2[11], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_13_5 = num_splits > 12 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_13_5) {
                                    float weight_112 = approx_exp2(lse_2[12] - lse_max_2[0]);
                                    if (bound_1[0] > 12) {
                                        den_2[0] = den_2[0] + weight_112;
                                        acc0_2[0] = __fmaf_rn(weight_112, p0_2[12], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_112, p1_2[12], acc1_2[0]);
                                    } else if (bound_1[1] > 12) {
                                        den_2[1] = den_2[1] + weight_112;
                                        acc0_2[1] = __fmaf_rn(weight_112, p0_2[12], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_112, p1_2[12], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 12) {
                                            den_2[2] = den_2[2] + weight_112;
                                            acc0_2[2] = __fmaf_rn(weight_112, p0_2[12], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_112, p1_2[12], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_112;
                                            acc0_2[3] = __fmaf_rn(weight_112, p0_2[12], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_112, p1_2[12], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_14_5 = num_splits > 13 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_14_5) {
                                    float weight_113 = approx_exp2(lse_2[13] - lse_max_2[0]);
                                    if (bound_1[0] > 13) {
                                        den_2[0] = den_2[0] + weight_113;
                                        acc0_2[0] = __fmaf_rn(weight_113, p0_2[13], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_113, p1_2[13], acc1_2[0]);
                                    } else if (bound_1[1] > 13) {
                                        den_2[1] = den_2[1] + weight_113;
                                        acc0_2[1] = __fmaf_rn(weight_113, p0_2[13], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_113, p1_2[13], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 13) {
                                            den_2[2] = den_2[2] + weight_113;
                                            acc0_2[2] = __fmaf_rn(weight_113, p0_2[13], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_113, p1_2[13], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_113;
                                            acc0_2[3] = __fmaf_rn(weight_113, p0_2[13], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_113, p1_2[13], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_15_2 = num_splits > 14 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_15_2) {
                                    float weight_114 = approx_exp2(lse_2[14] - lse_max_2[0]);
                                    if (bound_1[0] > 14) {
                                        den_2[0] = den_2[0] + weight_114;
                                        acc0_2[0] = __fmaf_rn(weight_114, p0_2[14], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_114, p1_2[14], acc1_2[0]);
                                    } else if (bound_1[1] > 14) {
                                        den_2[1] = den_2[1] + weight_114;
                                        acc0_2[1] = __fmaf_rn(weight_114, p0_2[14], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_114, p1_2[14], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 14) {
                                            den_2[2] = den_2[2] + weight_114;
                                            acc0_2[2] = __fmaf_rn(weight_114, p0_2[14], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_114, p1_2[14], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_114;
                                            acc0_2[3] = __fmaf_rn(weight_114, p0_2[14], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_114, p1_2[14], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_16_2 = num_splits > 15 && lse_max_2[0] > -CUDART_INF_F;
                                if (live_16_2) {
                                    float weight_115 = approx_exp2(lse_2[15] - lse_max_2[0]);
                                    if (bound_1[0] > 15) {
                                        den_2[0] = den_2[0] + weight_115;
                                        acc0_2[0] = __fmaf_rn(weight_115, p0_2[15], acc0_2[0]);
                                        acc1_2[0] = __fmaf_rn(weight_115, p1_2[15], acc1_2[0]);
                                    } else if (bound_1[1] > 15) {
                                        den_2[1] = den_2[1] + weight_115;
                                        acc0_2[1] = __fmaf_rn(weight_115, p0_2[15], acc0_2[1]);
                                        acc1_2[1] = __fmaf_rn(weight_115, p1_2[15], acc1_2[1]);
                                    } else {
                                        if (bound_1[2] > 15) {
                                            den_2[2] = den_2[2] + weight_115;
                                            acc0_2[2] = __fmaf_rn(weight_115, p0_2[15], acc0_2[2]);
                                            acc1_2[2] = __fmaf_rn(weight_115, p1_2[15], acc1_2[2]);
                                        } else {
                                            den_2[3] = den_2[3] + weight_115;
                                            acc0_2[3] = __fmaf_rn(weight_115, p0_2[15], acc0_2[3]);
                                            acc1_2[3] = __fmaf_rn(weight_115, p1_2[15], acc1_2[3]);
                                        }
                                    }
                                }
                                bool live_17_2 = num_splits > 0 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_17_2) {
                                    float weight_116 = approx_exp2(lse_2[16] - lse_max_2[1]);
                                    if (bound_1[0] > 0) {
                                        den_2[4] = den_2[4] + weight_116;
                                        acc0_2[4] = __fmaf_rn(weight_116, p0_2[16], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_116, p1_2[16], acc1_2[4]);
                                    } else if (bound_1[1] > 0) {
                                        den_2[5] = den_2[5] + weight_116;
                                        acc0_2[5] = __fmaf_rn(weight_116, p0_2[16], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_116, p1_2[16], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 0) {
                                            den_2[6] = den_2[6] + weight_116;
                                            acc0_2[6] = __fmaf_rn(weight_116, p0_2[16], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_116, p1_2[16], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_116;
                                            acc0_2[7] = __fmaf_rn(weight_116, p0_2[16], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_116, p1_2[16], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_18_2 = num_splits > 1 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_18_2) {
                                    float weight_117 = approx_exp2(lse_2[17] - lse_max_2[1]);
                                    if (bound_1[0] > 1) {
                                        den_2[4] = den_2[4] + weight_117;
                                        acc0_2[4] = __fmaf_rn(weight_117, p0_2[17], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_117, p1_2[17], acc1_2[4]);
                                    } else if (bound_1[1] > 1) {
                                        den_2[5] = den_2[5] + weight_117;
                                        acc0_2[5] = __fmaf_rn(weight_117, p0_2[17], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_117, p1_2[17], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 1) {
                                            den_2[6] = den_2[6] + weight_117;
                                            acc0_2[6] = __fmaf_rn(weight_117, p0_2[17], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_117, p1_2[17], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_117;
                                            acc0_2[7] = __fmaf_rn(weight_117, p0_2[17], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_117, p1_2[17], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_19_1 = num_splits > 2 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_19_1) {
                                    float weight_118 = approx_exp2(lse_2[18] - lse_max_2[1]);
                                    if (bound_1[0] > 2) {
                                        den_2[4] = den_2[4] + weight_118;
                                        acc0_2[4] = __fmaf_rn(weight_118, p0_2[18], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_118, p1_2[18], acc1_2[4]);
                                    } else if (bound_1[1] > 2) {
                                        den_2[5] = den_2[5] + weight_118;
                                        acc0_2[5] = __fmaf_rn(weight_118, p0_2[18], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_118, p1_2[18], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 2) {
                                            den_2[6] = den_2[6] + weight_118;
                                            acc0_2[6] = __fmaf_rn(weight_118, p0_2[18], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_118, p1_2[18], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_118;
                                            acc0_2[7] = __fmaf_rn(weight_118, p0_2[18], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_118, p1_2[18], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_20_1 = num_splits > 3 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_20_1) {
                                    float weight_119 = approx_exp2(lse_2[19] - lse_max_2[1]);
                                    if (bound_1[0] > 3) {
                                        den_2[4] = den_2[4] + weight_119;
                                        acc0_2[4] = __fmaf_rn(weight_119, p0_2[19], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_119, p1_2[19], acc1_2[4]);
                                    } else if (bound_1[1] > 3) {
                                        den_2[5] = den_2[5] + weight_119;
                                        acc0_2[5] = __fmaf_rn(weight_119, p0_2[19], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_119, p1_2[19], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 3) {
                                            den_2[6] = den_2[6] + weight_119;
                                            acc0_2[6] = __fmaf_rn(weight_119, p0_2[19], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_119, p1_2[19], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_119;
                                            acc0_2[7] = __fmaf_rn(weight_119, p0_2[19], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_119, p1_2[19], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_21_1 = num_splits > 4 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_21_1) {
                                    float weight_120 = approx_exp2(lse_2[20] - lse_max_2[1]);
                                    if (bound_1[0] > 4) {
                                        den_2[4] = den_2[4] + weight_120;
                                        acc0_2[4] = __fmaf_rn(weight_120, p0_2[20], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_120, p1_2[20], acc1_2[4]);
                                    } else if (bound_1[1] > 4) {
                                        den_2[5] = den_2[5] + weight_120;
                                        acc0_2[5] = __fmaf_rn(weight_120, p0_2[20], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_120, p1_2[20], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 4) {
                                            den_2[6] = den_2[6] + weight_120;
                                            acc0_2[6] = __fmaf_rn(weight_120, p0_2[20], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_120, p1_2[20], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_120;
                                            acc0_2[7] = __fmaf_rn(weight_120, p0_2[20], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_120, p1_2[20], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_22_1 = num_splits > 5 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_22_1) {
                                    float weight_121 = approx_exp2(lse_2[21] - lse_max_2[1]);
                                    if (bound_1[0] > 5) {
                                        den_2[4] = den_2[4] + weight_121;
                                        acc0_2[4] = __fmaf_rn(weight_121, p0_2[21], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_121, p1_2[21], acc1_2[4]);
                                    } else if (bound_1[1] > 5) {
                                        den_2[5] = den_2[5] + weight_121;
                                        acc0_2[5] = __fmaf_rn(weight_121, p0_2[21], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_121, p1_2[21], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 5) {
                                            den_2[6] = den_2[6] + weight_121;
                                            acc0_2[6] = __fmaf_rn(weight_121, p0_2[21], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_121, p1_2[21], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_121;
                                            acc0_2[7] = __fmaf_rn(weight_121, p0_2[21], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_121, p1_2[21], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_23_1 = num_splits > 6 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_23_1) {
                                    float weight_122 = approx_exp2(lse_2[22] - lse_max_2[1]);
                                    if (bound_1[0] > 6) {
                                        den_2[4] = den_2[4] + weight_122;
                                        acc0_2[4] = __fmaf_rn(weight_122, p0_2[22], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_122, p1_2[22], acc1_2[4]);
                                    } else if (bound_1[1] > 6) {
                                        den_2[5] = den_2[5] + weight_122;
                                        acc0_2[5] = __fmaf_rn(weight_122, p0_2[22], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_122, p1_2[22], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 6) {
                                            den_2[6] = den_2[6] + weight_122;
                                            acc0_2[6] = __fmaf_rn(weight_122, p0_2[22], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_122, p1_2[22], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_122;
                                            acc0_2[7] = __fmaf_rn(weight_122, p0_2[22], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_122, p1_2[22], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_24_1 = num_splits > 7 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_24_1) {
                                    float weight_123 = approx_exp2(lse_2[23] - lse_max_2[1]);
                                    if (bound_1[0] > 7) {
                                        den_2[4] = den_2[4] + weight_123;
                                        acc0_2[4] = __fmaf_rn(weight_123, p0_2[23], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_123, p1_2[23], acc1_2[4]);
                                    } else if (bound_1[1] > 7) {
                                        den_2[5] = den_2[5] + weight_123;
                                        acc0_2[5] = __fmaf_rn(weight_123, p0_2[23], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_123, p1_2[23], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 7) {
                                            den_2[6] = den_2[6] + weight_123;
                                            acc0_2[6] = __fmaf_rn(weight_123, p0_2[23], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_123, p1_2[23], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_123;
                                            acc0_2[7] = __fmaf_rn(weight_123, p0_2[23], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_123, p1_2[23], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_25_1 = num_splits > 8 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_25_1) {
                                    float weight_124 = approx_exp2(lse_2[24] - lse_max_2[1]);
                                    if (bound_1[0] > 8) {
                                        den_2[4] = den_2[4] + weight_124;
                                        acc0_2[4] = __fmaf_rn(weight_124, p0_2[24], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_124, p1_2[24], acc1_2[4]);
                                    } else if (bound_1[1] > 8) {
                                        den_2[5] = den_2[5] + weight_124;
                                        acc0_2[5] = __fmaf_rn(weight_124, p0_2[24], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_124, p1_2[24], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 8) {
                                            den_2[6] = den_2[6] + weight_124;
                                            acc0_2[6] = __fmaf_rn(weight_124, p0_2[24], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_124, p1_2[24], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_124;
                                            acc0_2[7] = __fmaf_rn(weight_124, p0_2[24], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_124, p1_2[24], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_26_1 = num_splits > 9 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_26_1) {
                                    float weight_125 = approx_exp2(lse_2[25] - lse_max_2[1]);
                                    if (bound_1[0] > 9) {
                                        den_2[4] = den_2[4] + weight_125;
                                        acc0_2[4] = __fmaf_rn(weight_125, p0_2[25], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_125, p1_2[25], acc1_2[4]);
                                    } else if (bound_1[1] > 9) {
                                        den_2[5] = den_2[5] + weight_125;
                                        acc0_2[5] = __fmaf_rn(weight_125, p0_2[25], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_125, p1_2[25], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 9) {
                                            den_2[6] = den_2[6] + weight_125;
                                            acc0_2[6] = __fmaf_rn(weight_125, p0_2[25], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_125, p1_2[25], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_125;
                                            acc0_2[7] = __fmaf_rn(weight_125, p0_2[25], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_125, p1_2[25], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_27_1 = num_splits > 10 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_27_1) {
                                    float weight_126 = approx_exp2(lse_2[26] - lse_max_2[1]);
                                    if (bound_1[0] > 10) {
                                        den_2[4] = den_2[4] + weight_126;
                                        acc0_2[4] = __fmaf_rn(weight_126, p0_2[26], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_126, p1_2[26], acc1_2[4]);
                                    } else if (bound_1[1] > 10) {
                                        den_2[5] = den_2[5] + weight_126;
                                        acc0_2[5] = __fmaf_rn(weight_126, p0_2[26], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_126, p1_2[26], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 10) {
                                            den_2[6] = den_2[6] + weight_126;
                                            acc0_2[6] = __fmaf_rn(weight_126, p0_2[26], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_126, p1_2[26], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_126;
                                            acc0_2[7] = __fmaf_rn(weight_126, p0_2[26], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_126, p1_2[26], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_28_1 = num_splits > 11 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_28_1) {
                                    float weight_127 = approx_exp2(lse_2[27] - lse_max_2[1]);
                                    if (bound_1[0] > 11) {
                                        den_2[4] = den_2[4] + weight_127;
                                        acc0_2[4] = __fmaf_rn(weight_127, p0_2[27], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_127, p1_2[27], acc1_2[4]);
                                    } else if (bound_1[1] > 11) {
                                        den_2[5] = den_2[5] + weight_127;
                                        acc0_2[5] = __fmaf_rn(weight_127, p0_2[27], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_127, p1_2[27], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 11) {
                                            den_2[6] = den_2[6] + weight_127;
                                            acc0_2[6] = __fmaf_rn(weight_127, p0_2[27], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_127, p1_2[27], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_127;
                                            acc0_2[7] = __fmaf_rn(weight_127, p0_2[27], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_127, p1_2[27], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_29_1 = num_splits > 12 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_29_1) {
                                    float weight_128 = approx_exp2(lse_2[28] - lse_max_2[1]);
                                    if (bound_1[0] > 12) {
                                        den_2[4] = den_2[4] + weight_128;
                                        acc0_2[4] = __fmaf_rn(weight_128, p0_2[28], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_128, p1_2[28], acc1_2[4]);
                                    } else if (bound_1[1] > 12) {
                                        den_2[5] = den_2[5] + weight_128;
                                        acc0_2[5] = __fmaf_rn(weight_128, p0_2[28], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_128, p1_2[28], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 12) {
                                            den_2[6] = den_2[6] + weight_128;
                                            acc0_2[6] = __fmaf_rn(weight_128, p0_2[28], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_128, p1_2[28], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_128;
                                            acc0_2[7] = __fmaf_rn(weight_128, p0_2[28], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_128, p1_2[28], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_30_1 = num_splits > 13 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_30_1) {
                                    float weight_129 = approx_exp2(lse_2[29] - lse_max_2[1]);
                                    if (bound_1[0] > 13) {
                                        den_2[4] = den_2[4] + weight_129;
                                        acc0_2[4] = __fmaf_rn(weight_129, p0_2[29], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_129, p1_2[29], acc1_2[4]);
                                    } else if (bound_1[1] > 13) {
                                        den_2[5] = den_2[5] + weight_129;
                                        acc0_2[5] = __fmaf_rn(weight_129, p0_2[29], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_129, p1_2[29], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 13) {
                                            den_2[6] = den_2[6] + weight_129;
                                            acc0_2[6] = __fmaf_rn(weight_129, p0_2[29], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_129, p1_2[29], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_129;
                                            acc0_2[7] = __fmaf_rn(weight_129, p0_2[29], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_129, p1_2[29], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_31_1 = num_splits > 14 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_31_1) {
                                    float weight_130 = approx_exp2(lse_2[30] - lse_max_2[1]);
                                    if (bound_1[0] > 14) {
                                        den_2[4] = den_2[4] + weight_130;
                                        acc0_2[4] = __fmaf_rn(weight_130, p0_2[30], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_130, p1_2[30], acc1_2[4]);
                                    } else if (bound_1[1] > 14) {
                                        den_2[5] = den_2[5] + weight_130;
                                        acc0_2[5] = __fmaf_rn(weight_130, p0_2[30], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_130, p1_2[30], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 14) {
                                            den_2[6] = den_2[6] + weight_130;
                                            acc0_2[6] = __fmaf_rn(weight_130, p0_2[30], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_130, p1_2[30], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_130;
                                            acc0_2[7] = __fmaf_rn(weight_130, p0_2[30], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_130, p1_2[30], acc1_2[7]);
                                        }
                                    }
                                }
                                bool live_32_1 = num_splits > 15 && lse_max_2[1] > -CUDART_INF_F;
                                if (live_32_1) {
                                    float weight_131 = approx_exp2(lse_2[31] - lse_max_2[1]);
                                    if (bound_1[0] > 15) {
                                        den_2[4] = den_2[4] + weight_131;
                                        acc0_2[4] = __fmaf_rn(weight_131, p0_2[31], acc0_2[4]);
                                        acc1_2[4] = __fmaf_rn(weight_131, p1_2[31], acc1_2[4]);
                                    } else if (bound_1[1] > 15) {
                                        den_2[5] = den_2[5] + weight_131;
                                        acc0_2[5] = __fmaf_rn(weight_131, p0_2[31], acc0_2[5]);
                                        acc1_2[5] = __fmaf_rn(weight_131, p1_2[31], acc1_2[5]);
                                    } else {
                                        if (bound_1[2] > 15) {
                                            den_2[6] = den_2[6] + weight_131;
                                            acc0_2[6] = __fmaf_rn(weight_131, p0_2[31], acc0_2[6]);
                                            acc1_2[6] = __fmaf_rn(weight_131, p1_2[31], acc1_2[6]);
                                        } else {
                                            den_2[7] = den_2[7] + weight_131;
                                            acc0_2[7] = __fmaf_rn(weight_131, p0_2[31], acc0_2[7]);
                                            acc1_2[7] = __fmaf_rn(weight_131, p1_2[31], acc1_2[7]);
                                        }
                                    }
                                }
                                if (valid_2[0] != 0) {
                                    float den_k_2 = den_2[0];
                                    float acc0_k_2 = acc0_2[0];
                                    float acc1_k_2 = acc1_2[0];
                                    den_k_2 = den_k_2 + den_2[1];
                                    acc0_k_2 = acc0_k_2 + acc0_2[1];
                                    acc1_k_2 = acc1_k_2 + acc1_2[1];
                                    den_k_2 = den_k_2 + den_2[2];
                                    acc0_k_2 = acc0_k_2 + acc0_2[2];
                                    acc1_k_2 = acc1_k_2 + acc1_2[2];
                                    den_k_2 = den_k_2 + den_2[3];
                                    acc0_k_2 = acc0_k_2 + acc0_2[3];
                                    acc1_k_2 = acc1_k_2 + acc1_2[3];
                                    float inv_4 = 0.0f;
                                    if (den_k_2 > 0.0f) {
                                        float _rcp_298 = approx_rcp(den_k_2);
                                        inv_4 = _rcp_298;
                                    }
                                    float attn0_2 = (float)(__nv_bfloat16)(acc0_k_2 * inv_4);
                                    float attn1_2 = (float)(__nv_bfloat16)(acc1_k_2 * inv_4);
                                    float _exp2_300 = approx_exp2((-gate0_2[0]) * 1.442695f);
                                    float _rcp_299 = approx_rcp(1.0f + _exp2_300);
                                    float result0_2 = attn0_2 * _rcp_299;
                                    float _exp2_302 = approx_exp2((-gate1_2[0]) * 1.442695f);
                                    float _rcp_301 = approx_rcp(1.0f + _exp2_302);
                                    float result1_2 = attn1_2 * _rcp_301;
                                    __nv_bfloat162 _bit_cast_src_303 = __float22bfloat162_rn(make_float2(result0_2, result1_2));
                                    word_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_303)[0];
                                    long long out_offset_2 = row_i64 * out_stride_row + qo_head_4[0] * out_stride_head + dim_2[0];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_2) + (0)) = word_2[0];
                                }
                                if (valid_2[1] != 0) {
                                    float den_k_3 = den_2[4];
                                    float acc0_k_3 = acc0_2[4];
                                    float acc1_k_3 = acc1_2[4];
                                    den_k_3 = den_k_3 + den_2[5];
                                    acc0_k_3 = acc0_k_3 + acc0_2[5];
                                    acc1_k_3 = acc1_k_3 + acc1_2[5];
                                    den_k_3 = den_k_3 + den_2[6];
                                    acc0_k_3 = acc0_k_3 + acc0_2[6];
                                    acc1_k_3 = acc1_k_3 + acc1_2[6];
                                    den_k_3 = den_k_3 + den_2[7];
                                    acc0_k_3 = acc0_k_3 + acc0_2[7];
                                    acc1_k_3 = acc1_k_3 + acc1_2[7];
                                    float inv_5 = 0.0f;
                                    if (den_k_3 > 0.0f) {
                                        float _rcp_304 = approx_rcp(den_k_3);
                                        inv_5 = _rcp_304;
                                    }
                                    float attn0_3 = (float)(__nv_bfloat16)(acc0_k_3 * inv_5);
                                    float attn1_3 = (float)(__nv_bfloat16)(acc1_k_3 * inv_5);
                                    float _exp2_306 = approx_exp2((-gate0_2[1]) * 1.442695f);
                                    float _rcp_305 = approx_rcp(1.0f + _exp2_306);
                                    float result0_3 = attn0_3 * _rcp_305;
                                    float _exp2_308 = approx_exp2((-gate1_2[1]) * 1.442695f);
                                    float _rcp_307 = approx_rcp(1.0f + _exp2_308);
                                    float result1_3 = attn1_3 * _rcp_307;
                                    __nv_bfloat162 _bit_cast_src_309 = __float22bfloat162_rn(make_float2(result0_3, result1_3));
                                    word_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_309)[0];
                                    long long out_offset_3 = row_i64 * out_stride_row + qo_head_4[1] * out_stride_head + dim_2[1];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_3) + (0)) = word_2[0];
                                }
                            }
                        } else if (num_splits > 4) {
                            float lse_3[16];
                            float p0_3[16];
                            float p1_3[16];
                            float acc0_3[8];
                            float acc1_3[8];
                            float den_3[8];
                            int chunk_w_3 = (num_splits + 3) / 4;
                            if (num_splits <= 4) {
                                chunk_w_3 = num_splits;
                            }
                            int bound_2[4];
                            bound_2[0] = chunk_w_3;
                            bound_2[1] = chunk_w_3 * 2;
                            bound_2[2] = chunk_w_3 * 3;
                            bound_2[3] = chunk_w_3 * 4;
                            float lse_max_3[2];
                            int valid_3[2];
                            int tab_3[2];
                            long long dim_3[2];
                            long long qo_head_5[2];
                            long long row_head_3[2];
                            float gate0_3[2];
                            float gate1_3[2];
                            unsigned int word_3[1];
                            long long nqo_3 = (long long)num_qo_heads;
                            #pragma unroll 1
                            for (int p_1_3 = p_first; p_1_3 < end; p_1_3 += 256) {
                                int pk_3 = ((p_1_3) < (end - 1) ? (p_1_3) : (end - 1));
                                valid_3[0] = ((end > p_1_3) ? 1 : 0);
                                int head_5 = pk_3 >> 7;
                                dim_3[0] = (long long)((pk_3 & 127) * 2);
                                qo_head_5[0] = (long long)(first_head + head_5);
                                row_head_3[0] = row_i64 * nqo_3 + qo_head_5[0];
                                acc0_3[0] = 0.0f;
                                acc1_3[0] = 0.0f;
                                den_3[0] = 0.0f;
                                acc0_3[1] = 0.0f;
                                acc1_3[1] = 0.0f;
                                den_3[1] = 0.0f;
                                acc0_3[2] = 0.0f;
                                acc1_3[2] = 0.0f;
                                den_3[2] = 0.0f;
                                acc0_3[3] = 0.0f;
                                acc1_3[3] = 0.0f;
                                den_3[3] = 0.0f;
                                lse_max_3[0] = -CUDART_INF_F;
                                float _vec_load_148[2];
                                {
                                    uint32_t _bf16x2_bits_310;
                                    _bf16x2_bits_310 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[0] * 256 + dim_3[0]) + 0);
                                    (&_vec_load_148[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_310) << 16);
                                    (&_vec_load_148[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_310) & 0xffff0000u);
                                }
                                gate0_3[0] = _vec_load_148[0];
                                gate1_3[0] = _vec_load_148[1];
                                int pk_0_1 = ((p_1_3 + 128) < (end - 1) ? (p_1_3 + 128) : (end - 1));
                                valid_3[1] = ((end > p_1_3 + 128) ? 1 : 0);
                                int head_1_2 = pk_0_1 >> 7;
                                dim_3[1] = (long long)((pk_0_1 & 127) * 2);
                                qo_head_5[1] = (long long)(first_head + head_1_2);
                                row_head_3[1] = row_i64 * nqo_3 + qo_head_5[1];
                                acc0_3[4] = 0.0f;
                                acc1_3[4] = 0.0f;
                                den_3[4] = 0.0f;
                                acc0_3[5] = 0.0f;
                                acc1_3[5] = 0.0f;
                                den_3[5] = 0.0f;
                                acc0_3[6] = 0.0f;
                                acc1_3[6] = 0.0f;
                                den_3[6] = 0.0f;
                                acc0_3[7] = 0.0f;
                                acc1_3[7] = 0.0f;
                                den_3[7] = 0.0f;
                                lse_max_3[1] = -CUDART_INF_F;
                                float _vec_load_149[2];
                                {
                                    uint32_t _bf16x2_bits_311;
                                    _bf16x2_bits_311 = *reinterpret_cast<const uint32_t*>(gate + (row_head_3[1] * 256 + dim_3[1]) + 0);
                                    (&_vec_load_149[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_311) << 16);
                                    (&_vec_load_149[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_311) & 0xffff0000u);
                                }
                                gate0_3[1] = _vec_load_149[0];
                                gate1_3[1] = _vec_load_149[1];
                                lse_3[0] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_132 = row_head_3[0];
                                    lse_3[0] = partial_lse[part_row_132];
                                    float _vec_load_150[2];
                                    {
                                        float2 _v2_312 = *reinterpret_cast<const float2*>(partial_out + (part_row_132 * 256 + dim_3[0]) + 0);
                                        _vec_load_150[0] = _v2_312.x;
                                        _vec_load_150[0 + 1] = _v2_312.y;
                                    }
                                    p0_3[0] = _vec_load_150[0];
                                    p1_3[0] = _vec_load_150[1];
                                }
                                lse_3[1] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_133 = split_pitch + row_head_3[0];
                                    lse_3[1] = partial_lse[part_row_133];
                                    float _vec_load_151[2];
                                    {
                                        float2 _v2_313 = *reinterpret_cast<const float2*>(partial_out + (part_row_133 * 256 + dim_3[0]) + 0);
                                        _vec_load_151[0] = _v2_313.x;
                                        _vec_load_151[0 + 1] = _v2_313.y;
                                    }
                                    p0_3[1] = _vec_load_151[0];
                                    p1_3[1] = _vec_load_151[1];
                                }
                                lse_3[2] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_134 = 2 * split_pitch + row_head_3[0];
                                    lse_3[2] = partial_lse[part_row_134];
                                    float _vec_load_152[2];
                                    {
                                        float2 _v2_314 = *reinterpret_cast<const float2*>(partial_out + (part_row_134 * 256 + dim_3[0]) + 0);
                                        _vec_load_152[0] = _v2_314.x;
                                        _vec_load_152[0 + 1] = _v2_314.y;
                                    }
                                    p0_3[2] = _vec_load_152[0];
                                    p1_3[2] = _vec_load_152[1];
                                }
                                lse_3[3] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_135 = 3 * split_pitch + row_head_3[0];
                                    lse_3[3] = partial_lse[part_row_135];
                                    float _vec_load_153[2];
                                    {
                                        float2 _v2_315 = *reinterpret_cast<const float2*>(partial_out + (part_row_135 * 256 + dim_3[0]) + 0);
                                        _vec_load_153[0] = _v2_315.x;
                                        _vec_load_153[0 + 1] = _v2_315.y;
                                    }
                                    p0_3[3] = _vec_load_153[0];
                                    p1_3[3] = _vec_load_153[1];
                                }
                                lse_3[4] = -CUDART_INF_F;
                                if (num_splits > 4) {
                                    long long part_row_136 = 4 * split_pitch + row_head_3[0];
                                    lse_3[4] = partial_lse[part_row_136];
                                    float _vec_load_154[2];
                                    {
                                        float2 _v2_316 = *reinterpret_cast<const float2*>(partial_out + (part_row_136 * 256 + dim_3[0]) + 0);
                                        _vec_load_154[0] = _v2_316.x;
                                        _vec_load_154[0 + 1] = _v2_316.y;
                                    }
                                    p0_3[4] = _vec_load_154[0];
                                    p1_3[4] = _vec_load_154[1];
                                }
                                lse_3[5] = -CUDART_INF_F;
                                if (num_splits > 5) {
                                    long long part_row_137 = 5 * split_pitch + row_head_3[0];
                                    lse_3[5] = partial_lse[part_row_137];
                                    float _vec_load_155[2];
                                    {
                                        float2 _v2_317 = *reinterpret_cast<const float2*>(partial_out + (part_row_137 * 256 + dim_3[0]) + 0);
                                        _vec_load_155[0] = _v2_317.x;
                                        _vec_load_155[0 + 1] = _v2_317.y;
                                    }
                                    p0_3[5] = _vec_load_155[0];
                                    p1_3[5] = _vec_load_155[1];
                                }
                                lse_3[6] = -CUDART_INF_F;
                                if (num_splits > 6) {
                                    long long part_row_138 = 6 * split_pitch + row_head_3[0];
                                    lse_3[6] = partial_lse[part_row_138];
                                    float _vec_load_156[2];
                                    {
                                        float2 _v2_318 = *reinterpret_cast<const float2*>(partial_out + (part_row_138 * 256 + dim_3[0]) + 0);
                                        _vec_load_156[0] = _v2_318.x;
                                        _vec_load_156[0 + 1] = _v2_318.y;
                                    }
                                    p0_3[6] = _vec_load_156[0];
                                    p1_3[6] = _vec_load_156[1];
                                }
                                lse_3[7] = -CUDART_INF_F;
                                if (num_splits > 7) {
                                    long long part_row_139 = 7 * split_pitch + row_head_3[0];
                                    lse_3[7] = partial_lse[part_row_139];
                                    float _vec_load_157[2];
                                    {
                                        float2 _v2_319 = *reinterpret_cast<const float2*>(partial_out + (part_row_139 * 256 + dim_3[0]) + 0);
                                        _vec_load_157[0] = _v2_319.x;
                                        _vec_load_157[0 + 1] = _v2_319.y;
                                    }
                                    p0_3[7] = _vec_load_157[0];
                                    p1_3[7] = _vec_load_157[1];
                                }
                                lse_3[8] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_140 = row_head_3[1];
                                    lse_3[8] = partial_lse[part_row_140];
                                    float _vec_load_158[2];
                                    {
                                        float2 _v2_320 = *reinterpret_cast<const float2*>(partial_out + (part_row_140 * 256 + dim_3[1]) + 0);
                                        _vec_load_158[0] = _v2_320.x;
                                        _vec_load_158[0 + 1] = _v2_320.y;
                                    }
                                    p0_3[8] = _vec_load_158[0];
                                    p1_3[8] = _vec_load_158[1];
                                }
                                lse_3[9] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_141 = split_pitch + row_head_3[1];
                                    lse_3[9] = partial_lse[part_row_141];
                                    float _vec_load_159[2];
                                    {
                                        float2 _v2_321 = *reinterpret_cast<const float2*>(partial_out + (part_row_141 * 256 + dim_3[1]) + 0);
                                        _vec_load_159[0] = _v2_321.x;
                                        _vec_load_159[0 + 1] = _v2_321.y;
                                    }
                                    p0_3[9] = _vec_load_159[0];
                                    p1_3[9] = _vec_load_159[1];
                                }
                                lse_3[10] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_142 = 2 * split_pitch + row_head_3[1];
                                    lse_3[10] = partial_lse[part_row_142];
                                    float _vec_load_160[2];
                                    {
                                        float2 _v2_322 = *reinterpret_cast<const float2*>(partial_out + (part_row_142 * 256 + dim_3[1]) + 0);
                                        _vec_load_160[0] = _v2_322.x;
                                        _vec_load_160[0 + 1] = _v2_322.y;
                                    }
                                    p0_3[10] = _vec_load_160[0];
                                    p1_3[10] = _vec_load_160[1];
                                }
                                lse_3[11] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_143 = 3 * split_pitch + row_head_3[1];
                                    lse_3[11] = partial_lse[part_row_143];
                                    float _vec_load_161[2];
                                    {
                                        float2 _v2_323 = *reinterpret_cast<const float2*>(partial_out + (part_row_143 * 256 + dim_3[1]) + 0);
                                        _vec_load_161[0] = _v2_323.x;
                                        _vec_load_161[0 + 1] = _v2_323.y;
                                    }
                                    p0_3[11] = _vec_load_161[0];
                                    p1_3[11] = _vec_load_161[1];
                                }
                                lse_3[12] = -CUDART_INF_F;
                                if (num_splits > 4) {
                                    long long part_row_144 = 4 * split_pitch + row_head_3[1];
                                    lse_3[12] = partial_lse[part_row_144];
                                    float _vec_load_162[2];
                                    {
                                        float2 _v2_324 = *reinterpret_cast<const float2*>(partial_out + (part_row_144 * 256 + dim_3[1]) + 0);
                                        _vec_load_162[0] = _v2_324.x;
                                        _vec_load_162[0 + 1] = _v2_324.y;
                                    }
                                    p0_3[12] = _vec_load_162[0];
                                    p1_3[12] = _vec_load_162[1];
                                }
                                lse_3[13] = -CUDART_INF_F;
                                if (num_splits > 5) {
                                    long long part_row_145 = 5 * split_pitch + row_head_3[1];
                                    lse_3[13] = partial_lse[part_row_145];
                                    float _vec_load_163[2];
                                    {
                                        float2 _v2_325 = *reinterpret_cast<const float2*>(partial_out + (part_row_145 * 256 + dim_3[1]) + 0);
                                        _vec_load_163[0] = _v2_325.x;
                                        _vec_load_163[0 + 1] = _v2_325.y;
                                    }
                                    p0_3[13] = _vec_load_163[0];
                                    p1_3[13] = _vec_load_163[1];
                                }
                                lse_3[14] = -CUDART_INF_F;
                                if (num_splits > 6) {
                                    long long part_row_146 = 6 * split_pitch + row_head_3[1];
                                    lse_3[14] = partial_lse[part_row_146];
                                    float _vec_load_164[2];
                                    {
                                        float2 _v2_326 = *reinterpret_cast<const float2*>(partial_out + (part_row_146 * 256 + dim_3[1]) + 0);
                                        _vec_load_164[0] = _v2_326.x;
                                        _vec_load_164[0 + 1] = _v2_326.y;
                                    }
                                    p0_3[14] = _vec_load_164[0];
                                    p1_3[14] = _vec_load_164[1];
                                }
                                lse_3[15] = -CUDART_INF_F;
                                if (num_splits > 7) {
                                    long long part_row_147 = 7 * split_pitch + row_head_3[1];
                                    lse_3[15] = partial_lse[part_row_147];
                                    float _vec_load_165[2];
                                    {
                                        float2 _v2_327 = *reinterpret_cast<const float2*>(partial_out + (part_row_147 * 256 + dim_3[1]) + 0);
                                        _vec_load_165[0] = _v2_327.x;
                                        _vec_load_165[0 + 1] = _v2_327.y;
                                    }
                                    p0_3[15] = _vec_load_165[0];
                                    p1_3[15] = _vec_load_165[1];
                                }
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[0]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[1]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[2]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[3]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[4]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[5]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[6]);
                                lse_max_3[0] = fmaxf(lse_max_3[0], lse_3[7]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[8]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[9]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[10]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[11]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[12]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[13]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[14]);
                                lse_max_3[1] = fmaxf(lse_max_3[1], lse_3[15]);
                                bool live_36 = num_splits > 0 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_36) {
                                    float weight_132 = approx_exp2(lse_3[0] - lse_max_3[0]);
                                    if (bound_2[0] > 0) {
                                        den_3[0] = den_3[0] + weight_132;
                                        acc0_3[0] = __fmaf_rn(weight_132, p0_3[0], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_132, p1_3[0], acc1_3[0]);
                                    } else if (bound_2[1] > 0) {
                                        den_3[1] = den_3[1] + weight_132;
                                        acc0_3[1] = __fmaf_rn(weight_132, p0_3[0], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_132, p1_3[0], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 0) {
                                            den_3[2] = den_3[2] + weight_132;
                                            acc0_3[2] = __fmaf_rn(weight_132, p0_3[0], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_132, p1_3[0], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_132;
                                            acc0_3[3] = __fmaf_rn(weight_132, p0_3[0], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_132, p1_3[0], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_2_6 = num_splits > 1 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_2_6) {
                                    float weight_133 = approx_exp2(lse_3[1] - lse_max_3[0]);
                                    if (bound_2[0] > 1) {
                                        den_3[0] = den_3[0] + weight_133;
                                        acc0_3[0] = __fmaf_rn(weight_133, p0_3[1], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_133, p1_3[1], acc1_3[0]);
                                    } else if (bound_2[1] > 1) {
                                        den_3[1] = den_3[1] + weight_133;
                                        acc0_3[1] = __fmaf_rn(weight_133, p0_3[1], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_133, p1_3[1], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 1) {
                                            den_3[2] = den_3[2] + weight_133;
                                            acc0_3[2] = __fmaf_rn(weight_133, p0_3[1], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_133, p1_3[1], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_133;
                                            acc0_3[3] = __fmaf_rn(weight_133, p0_3[1], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_133, p1_3[1], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_3_6 = num_splits > 2 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_3_6) {
                                    float weight_134 = approx_exp2(lse_3[2] - lse_max_3[0]);
                                    if (bound_2[0] > 2) {
                                        den_3[0] = den_3[0] + weight_134;
                                        acc0_3[0] = __fmaf_rn(weight_134, p0_3[2], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_134, p1_3[2], acc1_3[0]);
                                    } else if (bound_2[1] > 2) {
                                        den_3[1] = den_3[1] + weight_134;
                                        acc0_3[1] = __fmaf_rn(weight_134, p0_3[2], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_134, p1_3[2], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 2) {
                                            den_3[2] = den_3[2] + weight_134;
                                            acc0_3[2] = __fmaf_rn(weight_134, p0_3[2], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_134, p1_3[2], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_134;
                                            acc0_3[3] = __fmaf_rn(weight_134, p0_3[2], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_134, p1_3[2], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_4_6 = num_splits > 3 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_4_6) {
                                    float weight_135 = approx_exp2(lse_3[3] - lse_max_3[0]);
                                    if (bound_2[0] > 3) {
                                        den_3[0] = den_3[0] + weight_135;
                                        acc0_3[0] = __fmaf_rn(weight_135, p0_3[3], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_135, p1_3[3], acc1_3[0]);
                                    } else if (bound_2[1] > 3) {
                                        den_3[1] = den_3[1] + weight_135;
                                        acc0_3[1] = __fmaf_rn(weight_135, p0_3[3], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_135, p1_3[3], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 3) {
                                            den_3[2] = den_3[2] + weight_135;
                                            acc0_3[2] = __fmaf_rn(weight_135, p0_3[3], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_135, p1_3[3], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_135;
                                            acc0_3[3] = __fmaf_rn(weight_135, p0_3[3], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_135, p1_3[3], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_5_6 = num_splits > 4 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_5_6) {
                                    float weight_136 = approx_exp2(lse_3[4] - lse_max_3[0]);
                                    if (bound_2[0] > 4) {
                                        den_3[0] = den_3[0] + weight_136;
                                        acc0_3[0] = __fmaf_rn(weight_136, p0_3[4], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_136, p1_3[4], acc1_3[0]);
                                    } else if (bound_2[1] > 4) {
                                        den_3[1] = den_3[1] + weight_136;
                                        acc0_3[1] = __fmaf_rn(weight_136, p0_3[4], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_136, p1_3[4], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 4) {
                                            den_3[2] = den_3[2] + weight_136;
                                            acc0_3[2] = __fmaf_rn(weight_136, p0_3[4], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_136, p1_3[4], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_136;
                                            acc0_3[3] = __fmaf_rn(weight_136, p0_3[4], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_136, p1_3[4], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_6_6 = num_splits > 5 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_6_6) {
                                    float weight_137 = approx_exp2(lse_3[5] - lse_max_3[0]);
                                    if (bound_2[0] > 5) {
                                        den_3[0] = den_3[0] + weight_137;
                                        acc0_3[0] = __fmaf_rn(weight_137, p0_3[5], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_137, p1_3[5], acc1_3[0]);
                                    } else if (bound_2[1] > 5) {
                                        den_3[1] = den_3[1] + weight_137;
                                        acc0_3[1] = __fmaf_rn(weight_137, p0_3[5], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_137, p1_3[5], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 5) {
                                            den_3[2] = den_3[2] + weight_137;
                                            acc0_3[2] = __fmaf_rn(weight_137, p0_3[5], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_137, p1_3[5], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_137;
                                            acc0_3[3] = __fmaf_rn(weight_137, p0_3[5], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_137, p1_3[5], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_7_6 = num_splits > 6 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_7_6) {
                                    float weight_138 = approx_exp2(lse_3[6] - lse_max_3[0]);
                                    if (bound_2[0] > 6) {
                                        den_3[0] = den_3[0] + weight_138;
                                        acc0_3[0] = __fmaf_rn(weight_138, p0_3[6], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_138, p1_3[6], acc1_3[0]);
                                    } else if (bound_2[1] > 6) {
                                        den_3[1] = den_3[1] + weight_138;
                                        acc0_3[1] = __fmaf_rn(weight_138, p0_3[6], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_138, p1_3[6], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 6) {
                                            den_3[2] = den_3[2] + weight_138;
                                            acc0_3[2] = __fmaf_rn(weight_138, p0_3[6], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_138, p1_3[6], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_138;
                                            acc0_3[3] = __fmaf_rn(weight_138, p0_3[6], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_138, p1_3[6], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_8_6 = num_splits > 7 && lse_max_3[0] > -CUDART_INF_F;
                                if (live_8_6) {
                                    float weight_139 = approx_exp2(lse_3[7] - lse_max_3[0]);
                                    if (bound_2[0] > 7) {
                                        den_3[0] = den_3[0] + weight_139;
                                        acc0_3[0] = __fmaf_rn(weight_139, p0_3[7], acc0_3[0]);
                                        acc1_3[0] = __fmaf_rn(weight_139, p1_3[7], acc1_3[0]);
                                    } else if (bound_2[1] > 7) {
                                        den_3[1] = den_3[1] + weight_139;
                                        acc0_3[1] = __fmaf_rn(weight_139, p0_3[7], acc0_3[1]);
                                        acc1_3[1] = __fmaf_rn(weight_139, p1_3[7], acc1_3[1]);
                                    } else {
                                        if (bound_2[2] > 7) {
                                            den_3[2] = den_3[2] + weight_139;
                                            acc0_3[2] = __fmaf_rn(weight_139, p0_3[7], acc0_3[2]);
                                            acc1_3[2] = __fmaf_rn(weight_139, p1_3[7], acc1_3[2]);
                                        } else {
                                            den_3[3] = den_3[3] + weight_139;
                                            acc0_3[3] = __fmaf_rn(weight_139, p0_3[7], acc0_3[3]);
                                            acc1_3[3] = __fmaf_rn(weight_139, p1_3[7], acc1_3[3]);
                                        }
                                    }
                                }
                                bool live_9_6 = num_splits > 0 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_9_6) {
                                    float weight_140 = approx_exp2(lse_3[8] - lse_max_3[1]);
                                    if (bound_2[0] > 0) {
                                        den_3[4] = den_3[4] + weight_140;
                                        acc0_3[4] = __fmaf_rn(weight_140, p0_3[8], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_140, p1_3[8], acc1_3[4]);
                                    } else if (bound_2[1] > 0) {
                                        den_3[5] = den_3[5] + weight_140;
                                        acc0_3[5] = __fmaf_rn(weight_140, p0_3[8], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_140, p1_3[8], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 0) {
                                            den_3[6] = den_3[6] + weight_140;
                                            acc0_3[6] = __fmaf_rn(weight_140, p0_3[8], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_140, p1_3[8], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_140;
                                            acc0_3[7] = __fmaf_rn(weight_140, p0_3[8], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_140, p1_3[8], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_10_6 = num_splits > 1 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_10_6) {
                                    float weight_141 = approx_exp2(lse_3[9] - lse_max_3[1]);
                                    if (bound_2[0] > 1) {
                                        den_3[4] = den_3[4] + weight_141;
                                        acc0_3[4] = __fmaf_rn(weight_141, p0_3[9], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_141, p1_3[9], acc1_3[4]);
                                    } else if (bound_2[1] > 1) {
                                        den_3[5] = den_3[5] + weight_141;
                                        acc0_3[5] = __fmaf_rn(weight_141, p0_3[9], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_141, p1_3[9], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 1) {
                                            den_3[6] = den_3[6] + weight_141;
                                            acc0_3[6] = __fmaf_rn(weight_141, p0_3[9], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_141, p1_3[9], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_141;
                                            acc0_3[7] = __fmaf_rn(weight_141, p0_3[9], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_141, p1_3[9], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_11_6 = num_splits > 2 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_11_6) {
                                    float weight_142 = approx_exp2(lse_3[10] - lse_max_3[1]);
                                    if (bound_2[0] > 2) {
                                        den_3[4] = den_3[4] + weight_142;
                                        acc0_3[4] = __fmaf_rn(weight_142, p0_3[10], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_142, p1_3[10], acc1_3[4]);
                                    } else if (bound_2[1] > 2) {
                                        den_3[5] = den_3[5] + weight_142;
                                        acc0_3[5] = __fmaf_rn(weight_142, p0_3[10], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_142, p1_3[10], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 2) {
                                            den_3[6] = den_3[6] + weight_142;
                                            acc0_3[6] = __fmaf_rn(weight_142, p0_3[10], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_142, p1_3[10], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_142;
                                            acc0_3[7] = __fmaf_rn(weight_142, p0_3[10], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_142, p1_3[10], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_12_6 = num_splits > 3 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_12_6) {
                                    float weight_143 = approx_exp2(lse_3[11] - lse_max_3[1]);
                                    if (bound_2[0] > 3) {
                                        den_3[4] = den_3[4] + weight_143;
                                        acc0_3[4] = __fmaf_rn(weight_143, p0_3[11], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_143, p1_3[11], acc1_3[4]);
                                    } else if (bound_2[1] > 3) {
                                        den_3[5] = den_3[5] + weight_143;
                                        acc0_3[5] = __fmaf_rn(weight_143, p0_3[11], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_143, p1_3[11], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 3) {
                                            den_3[6] = den_3[6] + weight_143;
                                            acc0_3[6] = __fmaf_rn(weight_143, p0_3[11], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_143, p1_3[11], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_143;
                                            acc0_3[7] = __fmaf_rn(weight_143, p0_3[11], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_143, p1_3[11], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_13_6 = num_splits > 4 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_13_6) {
                                    float weight_144 = approx_exp2(lse_3[12] - lse_max_3[1]);
                                    if (bound_2[0] > 4) {
                                        den_3[4] = den_3[4] + weight_144;
                                        acc0_3[4] = __fmaf_rn(weight_144, p0_3[12], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_144, p1_3[12], acc1_3[4]);
                                    } else if (bound_2[1] > 4) {
                                        den_3[5] = den_3[5] + weight_144;
                                        acc0_3[5] = __fmaf_rn(weight_144, p0_3[12], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_144, p1_3[12], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 4) {
                                            den_3[6] = den_3[6] + weight_144;
                                            acc0_3[6] = __fmaf_rn(weight_144, p0_3[12], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_144, p1_3[12], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_144;
                                            acc0_3[7] = __fmaf_rn(weight_144, p0_3[12], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_144, p1_3[12], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_14_6 = num_splits > 5 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_14_6) {
                                    float weight_145 = approx_exp2(lse_3[13] - lse_max_3[1]);
                                    if (bound_2[0] > 5) {
                                        den_3[4] = den_3[4] + weight_145;
                                        acc0_3[4] = __fmaf_rn(weight_145, p0_3[13], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_145, p1_3[13], acc1_3[4]);
                                    } else if (bound_2[1] > 5) {
                                        den_3[5] = den_3[5] + weight_145;
                                        acc0_3[5] = __fmaf_rn(weight_145, p0_3[13], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_145, p1_3[13], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 5) {
                                            den_3[6] = den_3[6] + weight_145;
                                            acc0_3[6] = __fmaf_rn(weight_145, p0_3[13], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_145, p1_3[13], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_145;
                                            acc0_3[7] = __fmaf_rn(weight_145, p0_3[13], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_145, p1_3[13], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_15_3 = num_splits > 6 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_15_3) {
                                    float weight_146 = approx_exp2(lse_3[14] - lse_max_3[1]);
                                    if (bound_2[0] > 6) {
                                        den_3[4] = den_3[4] + weight_146;
                                        acc0_3[4] = __fmaf_rn(weight_146, p0_3[14], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_146, p1_3[14], acc1_3[4]);
                                    } else if (bound_2[1] > 6) {
                                        den_3[5] = den_3[5] + weight_146;
                                        acc0_3[5] = __fmaf_rn(weight_146, p0_3[14], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_146, p1_3[14], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 6) {
                                            den_3[6] = den_3[6] + weight_146;
                                            acc0_3[6] = __fmaf_rn(weight_146, p0_3[14], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_146, p1_3[14], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_146;
                                            acc0_3[7] = __fmaf_rn(weight_146, p0_3[14], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_146, p1_3[14], acc1_3[7]);
                                        }
                                    }
                                }
                                bool live_16_3 = num_splits > 7 && lse_max_3[1] > -CUDART_INF_F;
                                if (live_16_3) {
                                    float weight_147 = approx_exp2(lse_3[15] - lse_max_3[1]);
                                    if (bound_2[0] > 7) {
                                        den_3[4] = den_3[4] + weight_147;
                                        acc0_3[4] = __fmaf_rn(weight_147, p0_3[15], acc0_3[4]);
                                        acc1_3[4] = __fmaf_rn(weight_147, p1_3[15], acc1_3[4]);
                                    } else if (bound_2[1] > 7) {
                                        den_3[5] = den_3[5] + weight_147;
                                        acc0_3[5] = __fmaf_rn(weight_147, p0_3[15], acc0_3[5]);
                                        acc1_3[5] = __fmaf_rn(weight_147, p1_3[15], acc1_3[5]);
                                    } else {
                                        if (bound_2[2] > 7) {
                                            den_3[6] = den_3[6] + weight_147;
                                            acc0_3[6] = __fmaf_rn(weight_147, p0_3[15], acc0_3[6]);
                                            acc1_3[6] = __fmaf_rn(weight_147, p1_3[15], acc1_3[6]);
                                        } else {
                                            den_3[7] = den_3[7] + weight_147;
                                            acc0_3[7] = __fmaf_rn(weight_147, p0_3[15], acc0_3[7]);
                                            acc1_3[7] = __fmaf_rn(weight_147, p1_3[15], acc1_3[7]);
                                        }
                                    }
                                }
                                if (valid_3[0] != 0) {
                                    float den_k_4 = den_3[0];
                                    float acc0_k_4 = acc0_3[0];
                                    float acc1_k_4 = acc1_3[0];
                                    den_k_4 = den_k_4 + den_3[1];
                                    acc0_k_4 = acc0_k_4 + acc0_3[1];
                                    acc1_k_4 = acc1_k_4 + acc1_3[1];
                                    den_k_4 = den_k_4 + den_3[2];
                                    acc0_k_4 = acc0_k_4 + acc0_3[2];
                                    acc1_k_4 = acc1_k_4 + acc1_3[2];
                                    den_k_4 = den_k_4 + den_3[3];
                                    acc0_k_4 = acc0_k_4 + acc0_3[3];
                                    acc1_k_4 = acc1_k_4 + acc1_3[3];
                                    float inv_6 = 0.0f;
                                    if (den_k_4 > 0.0f) {
                                        float _rcp_328 = approx_rcp(den_k_4);
                                        inv_6 = _rcp_328;
                                    }
                                    float attn0_4 = (float)(__nv_bfloat16)(acc0_k_4 * inv_6);
                                    float attn1_4 = (float)(__nv_bfloat16)(acc1_k_4 * inv_6);
                                    float _exp2_330 = approx_exp2((-gate0_3[0]) * 1.442695f);
                                    float _rcp_329 = approx_rcp(1.0f + _exp2_330);
                                    float result0_4 = attn0_4 * _rcp_329;
                                    float _exp2_332 = approx_exp2((-gate1_3[0]) * 1.442695f);
                                    float _rcp_331 = approx_rcp(1.0f + _exp2_332);
                                    float result1_4 = attn1_4 * _rcp_331;
                                    __nv_bfloat162 _bit_cast_src_333 = __float22bfloat162_rn(make_float2(result0_4, result1_4));
                                    word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_333)[0];
                                    long long out_offset_4 = row_i64 * out_stride_row + qo_head_5[0] * out_stride_head + dim_3[0];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_4) + (0)) = word_3[0];
                                }
                                if (valid_3[1] != 0) {
                                    float den_k_5 = den_3[4];
                                    float acc0_k_5 = acc0_3[4];
                                    float acc1_k_5 = acc1_3[4];
                                    den_k_5 = den_k_5 + den_3[5];
                                    acc0_k_5 = acc0_k_5 + acc0_3[5];
                                    acc1_k_5 = acc1_k_5 + acc1_3[5];
                                    den_k_5 = den_k_5 + den_3[6];
                                    acc0_k_5 = acc0_k_5 + acc0_3[6];
                                    acc1_k_5 = acc1_k_5 + acc1_3[6];
                                    den_k_5 = den_k_5 + den_3[7];
                                    acc0_k_5 = acc0_k_5 + acc0_3[7];
                                    acc1_k_5 = acc1_k_5 + acc1_3[7];
                                    float inv_7 = 0.0f;
                                    if (den_k_5 > 0.0f) {
                                        float _rcp_334 = approx_rcp(den_k_5);
                                        inv_7 = _rcp_334;
                                    }
                                    float attn0_5 = (float)(__nv_bfloat16)(acc0_k_5 * inv_7);
                                    float attn1_5 = (float)(__nv_bfloat16)(acc1_k_5 * inv_7);
                                    float _exp2_336 = approx_exp2((-gate0_3[1]) * 1.442695f);
                                    float _rcp_335 = approx_rcp(1.0f + _exp2_336);
                                    float result0_5 = attn0_5 * _rcp_335;
                                    float _exp2_338 = approx_exp2((-gate1_3[1]) * 1.442695f);
                                    float _rcp_337 = approx_rcp(1.0f + _exp2_338);
                                    float result1_5 = attn1_5 * _rcp_337;
                                    __nv_bfloat162 _bit_cast_src_339 = __float22bfloat162_rn(make_float2(result0_5, result1_5));
                                    word_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_339)[0];
                                    long long out_offset_5 = row_i64 * out_stride_row + qo_head_5[1] * out_stride_head + dim_3[1];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_5) + (0)) = word_3[0];
                                }
                            }
                        } else {
                            float lse_4[16];
                            float p0_4[16];
                            float p1_4[16];
                            float acc0_4[16];
                            float acc1_4[16];
                            float den_4[16];
                            int chunk_w_4 = (num_splits + 3) / 4;
                            if (num_splits <= 4) {
                                chunk_w_4 = num_splits;
                            }
                            int bound_3[4];
                            bound_3[0] = chunk_w_4;
                            bound_3[1] = chunk_w_4 * 2;
                            bound_3[2] = chunk_w_4 * 3;
                            bound_3[3] = chunk_w_4 * 4;
                            float lse_max_4[4];
                            int valid_4[4];
                            int tab_4[4];
                            long long dim_4[4];
                            long long qo_head_6[4];
                            long long row_head_4[4];
                            float gate0_4[4];
                            float gate1_4[4];
                            unsigned int word_4[1];
                            long long nqo_4 = (long long)num_qo_heads;
                            #pragma unroll 1
                            for (int p_1_4 = p_first; p_1_4 < end; p_1_4 += 512) {
                                int pk_4 = ((p_1_4) < (end - 1) ? (p_1_4) : (end - 1));
                                valid_4[0] = ((end > p_1_4) ? 1 : 0);
                                int head_6 = pk_4 >> 7;
                                dim_4[0] = (long long)((pk_4 & 127) * 2);
                                qo_head_6[0] = (long long)(first_head + head_6);
                                row_head_4[0] = row_i64 * nqo_4 + qo_head_6[0];
                                acc0_4[0] = 0.0f;
                                acc1_4[0] = 0.0f;
                                den_4[0] = 0.0f;
                                acc0_4[1] = 0.0f;
                                acc1_4[1] = 0.0f;
                                den_4[1] = 0.0f;
                                acc0_4[2] = 0.0f;
                                acc1_4[2] = 0.0f;
                                den_4[2] = 0.0f;
                                acc0_4[3] = 0.0f;
                                acc1_4[3] = 0.0f;
                                den_4[3] = 0.0f;
                                lse_max_4[0] = -CUDART_INF_F;
                                float _vec_load_166[2];
                                {
                                    uint32_t _bf16x2_bits_340;
                                    _bf16x2_bits_340 = *reinterpret_cast<const uint32_t*>(gate + (row_head_4[0] * 256 + dim_4[0]) + 0);
                                    (&_vec_load_166[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_340) << 16);
                                    (&_vec_load_166[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_340) & 0xffff0000u);
                                }
                                gate0_4[0] = _vec_load_166[0];
                                gate1_4[0] = _vec_load_166[1];
                                int pk_0_2 = ((p_1_4 + 128) < (end - 1) ? (p_1_4 + 128) : (end - 1));
                                valid_4[1] = ((end > p_1_4 + 128) ? 1 : 0);
                                int head_1_3 = pk_0_2 >> 7;
                                dim_4[1] = (long long)((pk_0_2 & 127) * 2);
                                qo_head_6[1] = (long long)(first_head + head_1_3);
                                row_head_4[1] = row_i64 * nqo_4 + qo_head_6[1];
                                acc0_4[4] = 0.0f;
                                acc1_4[4] = 0.0f;
                                den_4[4] = 0.0f;
                                acc0_4[5] = 0.0f;
                                acc1_4[5] = 0.0f;
                                den_4[5] = 0.0f;
                                acc0_4[6] = 0.0f;
                                acc1_4[6] = 0.0f;
                                den_4[6] = 0.0f;
                                acc0_4[7] = 0.0f;
                                acc1_4[7] = 0.0f;
                                den_4[7] = 0.0f;
                                lse_max_4[1] = -CUDART_INF_F;
                                float _vec_load_167[2];
                                {
                                    uint32_t _bf16x2_bits_341;
                                    _bf16x2_bits_341 = *reinterpret_cast<const uint32_t*>(gate + (row_head_4[1] * 256 + dim_4[1]) + 0);
                                    (&_vec_load_167[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_341) << 16);
                                    (&_vec_load_167[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_341) & 0xffff0000u);
                                }
                                gate0_4[1] = _vec_load_167[0];
                                gate1_4[1] = _vec_load_167[1];
                                int pk_2_1 = ((p_1_4 + 256) < (end - 1) ? (p_1_4 + 256) : (end - 1));
                                valid_4[2] = ((end > p_1_4 + 256) ? 1 : 0);
                                int head_3_1 = pk_2_1 >> 7;
                                dim_4[2] = (long long)((pk_2_1 & 127) * 2);
                                qo_head_6[2] = (long long)(first_head + head_3_1);
                                row_head_4[2] = row_i64 * nqo_4 + qo_head_6[2];
                                acc0_4[8] = 0.0f;
                                acc1_4[8] = 0.0f;
                                den_4[8] = 0.0f;
                                acc0_4[9] = 0.0f;
                                acc1_4[9] = 0.0f;
                                den_4[9] = 0.0f;
                                acc0_4[10] = 0.0f;
                                acc1_4[10] = 0.0f;
                                den_4[10] = 0.0f;
                                acc0_4[11] = 0.0f;
                                acc1_4[11] = 0.0f;
                                den_4[11] = 0.0f;
                                lse_max_4[2] = -CUDART_INF_F;
                                float _vec_load_168[2];
                                {
                                    uint32_t _bf16x2_bits_342;
                                    _bf16x2_bits_342 = *reinterpret_cast<const uint32_t*>(gate + (row_head_4[2] * 256 + dim_4[2]) + 0);
                                    (&_vec_load_168[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_342) << 16);
                                    (&_vec_load_168[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_342) & 0xffff0000u);
                                }
                                gate0_4[2] = _vec_load_168[0];
                                gate1_4[2] = _vec_load_168[1];
                                int pk_4_1 = ((p_1_4 + 384) < (end - 1) ? (p_1_4 + 384) : (end - 1));
                                valid_4[3] = ((end > p_1_4 + 384) ? 1 : 0);
                                int head_5_1 = pk_4_1 >> 7;
                                dim_4[3] = (long long)((pk_4_1 & 127) * 2);
                                qo_head_6[3] = (long long)(first_head + head_5_1);
                                row_head_4[3] = row_i64 * nqo_4 + qo_head_6[3];
                                acc0_4[12] = 0.0f;
                                acc1_4[12] = 0.0f;
                                den_4[12] = 0.0f;
                                acc0_4[13] = 0.0f;
                                acc1_4[13] = 0.0f;
                                den_4[13] = 0.0f;
                                acc0_4[14] = 0.0f;
                                acc1_4[14] = 0.0f;
                                den_4[14] = 0.0f;
                                acc0_4[15] = 0.0f;
                                acc1_4[15] = 0.0f;
                                den_4[15] = 0.0f;
                                lse_max_4[3] = -CUDART_INF_F;
                                float _vec_load_169[2];
                                {
                                    uint32_t _bf16x2_bits_343;
                                    _bf16x2_bits_343 = *reinterpret_cast<const uint32_t*>(gate + (row_head_4[3] * 256 + dim_4[3]) + 0);
                                    (&_vec_load_169[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_343) << 16);
                                    (&_vec_load_169[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_343) & 0xffff0000u);
                                }
                                gate0_4[3] = _vec_load_169[0];
                                gate1_4[3] = _vec_load_169[1];
                                lse_4[0] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_148 = row_head_4[0];
                                    lse_4[0] = partial_lse[part_row_148];
                                    float _vec_load_170[2];
                                    {
                                        float2 _v2_344 = *reinterpret_cast<const float2*>(partial_out + (part_row_148 * 256 + dim_4[0]) + 0);
                                        _vec_load_170[0] = _v2_344.x;
                                        _vec_load_170[0 + 1] = _v2_344.y;
                                    }
                                    p0_4[0] = _vec_load_170[0];
                                    p1_4[0] = _vec_load_170[1];
                                }
                                lse_4[1] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_149 = split_pitch + row_head_4[0];
                                    lse_4[1] = partial_lse[part_row_149];
                                    float _vec_load_171[2];
                                    {
                                        float2 _v2_345 = *reinterpret_cast<const float2*>(partial_out + (part_row_149 * 256 + dim_4[0]) + 0);
                                        _vec_load_171[0] = _v2_345.x;
                                        _vec_load_171[0 + 1] = _v2_345.y;
                                    }
                                    p0_4[1] = _vec_load_171[0];
                                    p1_4[1] = _vec_load_171[1];
                                }
                                lse_4[2] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_150 = 2 * split_pitch + row_head_4[0];
                                    lse_4[2] = partial_lse[part_row_150];
                                    float _vec_load_172[2];
                                    {
                                        float2 _v2_346 = *reinterpret_cast<const float2*>(partial_out + (part_row_150 * 256 + dim_4[0]) + 0);
                                        _vec_load_172[0] = _v2_346.x;
                                        _vec_load_172[0 + 1] = _v2_346.y;
                                    }
                                    p0_4[2] = _vec_load_172[0];
                                    p1_4[2] = _vec_load_172[1];
                                }
                                lse_4[3] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_151 = 3 * split_pitch + row_head_4[0];
                                    lse_4[3] = partial_lse[part_row_151];
                                    float _vec_load_173[2];
                                    {
                                        float2 _v2_347 = *reinterpret_cast<const float2*>(partial_out + (part_row_151 * 256 + dim_4[0]) + 0);
                                        _vec_load_173[0] = _v2_347.x;
                                        _vec_load_173[0 + 1] = _v2_347.y;
                                    }
                                    p0_4[3] = _vec_load_173[0];
                                    p1_4[3] = _vec_load_173[1];
                                }
                                lse_4[4] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_152 = row_head_4[1];
                                    lse_4[4] = partial_lse[part_row_152];
                                    float _vec_load_174[2];
                                    {
                                        float2 _v2_348 = *reinterpret_cast<const float2*>(partial_out + (part_row_152 * 256 + dim_4[1]) + 0);
                                        _vec_load_174[0] = _v2_348.x;
                                        _vec_load_174[0 + 1] = _v2_348.y;
                                    }
                                    p0_4[4] = _vec_load_174[0];
                                    p1_4[4] = _vec_load_174[1];
                                }
                                lse_4[5] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_153 = split_pitch + row_head_4[1];
                                    lse_4[5] = partial_lse[part_row_153];
                                    float _vec_load_175[2];
                                    {
                                        float2 _v2_349 = *reinterpret_cast<const float2*>(partial_out + (part_row_153 * 256 + dim_4[1]) + 0);
                                        _vec_load_175[0] = _v2_349.x;
                                        _vec_load_175[0 + 1] = _v2_349.y;
                                    }
                                    p0_4[5] = _vec_load_175[0];
                                    p1_4[5] = _vec_load_175[1];
                                }
                                lse_4[6] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_154 = 2 * split_pitch + row_head_4[1];
                                    lse_4[6] = partial_lse[part_row_154];
                                    float _vec_load_176[2];
                                    {
                                        float2 _v2_350 = *reinterpret_cast<const float2*>(partial_out + (part_row_154 * 256 + dim_4[1]) + 0);
                                        _vec_load_176[0] = _v2_350.x;
                                        _vec_load_176[0 + 1] = _v2_350.y;
                                    }
                                    p0_4[6] = _vec_load_176[0];
                                    p1_4[6] = _vec_load_176[1];
                                }
                                lse_4[7] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_155 = 3 * split_pitch + row_head_4[1];
                                    lse_4[7] = partial_lse[part_row_155];
                                    float _vec_load_177[2];
                                    {
                                        float2 _v2_351 = *reinterpret_cast<const float2*>(partial_out + (part_row_155 * 256 + dim_4[1]) + 0);
                                        _vec_load_177[0] = _v2_351.x;
                                        _vec_load_177[0 + 1] = _v2_351.y;
                                    }
                                    p0_4[7] = _vec_load_177[0];
                                    p1_4[7] = _vec_load_177[1];
                                }
                                lse_4[8] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_156 = row_head_4[2];
                                    lse_4[8] = partial_lse[part_row_156];
                                    float _vec_load_178[2];
                                    {
                                        float2 _v2_352 = *reinterpret_cast<const float2*>(partial_out + (part_row_156 * 256 + dim_4[2]) + 0);
                                        _vec_load_178[0] = _v2_352.x;
                                        _vec_load_178[0 + 1] = _v2_352.y;
                                    }
                                    p0_4[8] = _vec_load_178[0];
                                    p1_4[8] = _vec_load_178[1];
                                }
                                lse_4[9] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_157 = split_pitch + row_head_4[2];
                                    lse_4[9] = partial_lse[part_row_157];
                                    float _vec_load_179[2];
                                    {
                                        float2 _v2_353 = *reinterpret_cast<const float2*>(partial_out + (part_row_157 * 256 + dim_4[2]) + 0);
                                        _vec_load_179[0] = _v2_353.x;
                                        _vec_load_179[0 + 1] = _v2_353.y;
                                    }
                                    p0_4[9] = _vec_load_179[0];
                                    p1_4[9] = _vec_load_179[1];
                                }
                                lse_4[10] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_158 = 2 * split_pitch + row_head_4[2];
                                    lse_4[10] = partial_lse[part_row_158];
                                    float _vec_load_180[2];
                                    {
                                        float2 _v2_354 = *reinterpret_cast<const float2*>(partial_out + (part_row_158 * 256 + dim_4[2]) + 0);
                                        _vec_load_180[0] = _v2_354.x;
                                        _vec_load_180[0 + 1] = _v2_354.y;
                                    }
                                    p0_4[10] = _vec_load_180[0];
                                    p1_4[10] = _vec_load_180[1];
                                }
                                lse_4[11] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_159 = 3 * split_pitch + row_head_4[2];
                                    lse_4[11] = partial_lse[part_row_159];
                                    float _vec_load_181[2];
                                    {
                                        float2 _v2_355 = *reinterpret_cast<const float2*>(partial_out + (part_row_159 * 256 + dim_4[2]) + 0);
                                        _vec_load_181[0] = _v2_355.x;
                                        _vec_load_181[0 + 1] = _v2_355.y;
                                    }
                                    p0_4[11] = _vec_load_181[0];
                                    p1_4[11] = _vec_load_181[1];
                                }
                                lse_4[12] = -CUDART_INF_F;
                                if (num_splits > 0) {
                                    long long part_row_160 = row_head_4[3];
                                    lse_4[12] = partial_lse[part_row_160];
                                    float _vec_load_182[2];
                                    {
                                        float2 _v2_356 = *reinterpret_cast<const float2*>(partial_out + (part_row_160 * 256 + dim_4[3]) + 0);
                                        _vec_load_182[0] = _v2_356.x;
                                        _vec_load_182[0 + 1] = _v2_356.y;
                                    }
                                    p0_4[12] = _vec_load_182[0];
                                    p1_4[12] = _vec_load_182[1];
                                }
                                lse_4[13] = -CUDART_INF_F;
                                if (num_splits > 1) {
                                    long long part_row_161 = split_pitch + row_head_4[3];
                                    lse_4[13] = partial_lse[part_row_161];
                                    float _vec_load_183[2];
                                    {
                                        float2 _v2_357 = *reinterpret_cast<const float2*>(partial_out + (part_row_161 * 256 + dim_4[3]) + 0);
                                        _vec_load_183[0] = _v2_357.x;
                                        _vec_load_183[0 + 1] = _v2_357.y;
                                    }
                                    p0_4[13] = _vec_load_183[0];
                                    p1_4[13] = _vec_load_183[1];
                                }
                                lse_4[14] = -CUDART_INF_F;
                                if (num_splits > 2) {
                                    long long part_row_162 = 2 * split_pitch + row_head_4[3];
                                    lse_4[14] = partial_lse[part_row_162];
                                    float _vec_load_184[2];
                                    {
                                        float2 _v2_358 = *reinterpret_cast<const float2*>(partial_out + (part_row_162 * 256 + dim_4[3]) + 0);
                                        _vec_load_184[0] = _v2_358.x;
                                        _vec_load_184[0 + 1] = _v2_358.y;
                                    }
                                    p0_4[14] = _vec_load_184[0];
                                    p1_4[14] = _vec_load_184[1];
                                }
                                lse_4[15] = -CUDART_INF_F;
                                if (num_splits > 3) {
                                    long long part_row_163 = 3 * split_pitch + row_head_4[3];
                                    lse_4[15] = partial_lse[part_row_163];
                                    float _vec_load_185[2];
                                    {
                                        float2 _v2_359 = *reinterpret_cast<const float2*>(partial_out + (part_row_163 * 256 + dim_4[3]) + 0);
                                        _vec_load_185[0] = _v2_359.x;
                                        _vec_load_185[0 + 1] = _v2_359.y;
                                    }
                                    p0_4[15] = _vec_load_185[0];
                                    p1_4[15] = _vec_load_185[1];
                                }
                                lse_max_4[0] = fmaxf(lse_max_4[0], lse_4[0]);
                                lse_max_4[0] = fmaxf(lse_max_4[0], lse_4[1]);
                                lse_max_4[0] = fmaxf(lse_max_4[0], lse_4[2]);
                                lse_max_4[0] = fmaxf(lse_max_4[0], lse_4[3]);
                                lse_max_4[1] = fmaxf(lse_max_4[1], lse_4[4]);
                                lse_max_4[1] = fmaxf(lse_max_4[1], lse_4[5]);
                                lse_max_4[1] = fmaxf(lse_max_4[1], lse_4[6]);
                                lse_max_4[1] = fmaxf(lse_max_4[1], lse_4[7]);
                                lse_max_4[2] = fmaxf(lse_max_4[2], lse_4[8]);
                                lse_max_4[2] = fmaxf(lse_max_4[2], lse_4[9]);
                                lse_max_4[2] = fmaxf(lse_max_4[2], lse_4[10]);
                                lse_max_4[2] = fmaxf(lse_max_4[2], lse_4[11]);
                                lse_max_4[3] = fmaxf(lse_max_4[3], lse_4[12]);
                                lse_max_4[3] = fmaxf(lse_max_4[3], lse_4[13]);
                                lse_max_4[3] = fmaxf(lse_max_4[3], lse_4[14]);
                                lse_max_4[3] = fmaxf(lse_max_4[3], lse_4[15]);
                                bool live_37 = num_splits > 0 && lse_max_4[0] > -CUDART_INF_F;
                                if (live_37) {
                                    float weight_148 = approx_exp2(lse_4[0] - lse_max_4[0]);
                                    if (bound_3[0] > 0) {
                                        den_4[0] = den_4[0] + weight_148;
                                        acc0_4[0] = __fmaf_rn(weight_148, p0_4[0], acc0_4[0]);
                                        acc1_4[0] = __fmaf_rn(weight_148, p1_4[0], acc1_4[0]);
                                    } else if (bound_3[1] > 0) {
                                        den_4[1] = den_4[1] + weight_148;
                                        acc0_4[1] = __fmaf_rn(weight_148, p0_4[0], acc0_4[1]);
                                        acc1_4[1] = __fmaf_rn(weight_148, p1_4[0], acc1_4[1]);
                                    } else {
                                        if (bound_3[2] > 0) {
                                            den_4[2] = den_4[2] + weight_148;
                                            acc0_4[2] = __fmaf_rn(weight_148, p0_4[0], acc0_4[2]);
                                            acc1_4[2] = __fmaf_rn(weight_148, p1_4[0], acc1_4[2]);
                                        } else {
                                            den_4[3] = den_4[3] + weight_148;
                                            acc0_4[3] = __fmaf_rn(weight_148, p0_4[0], acc0_4[3]);
                                            acc1_4[3] = __fmaf_rn(weight_148, p1_4[0], acc1_4[3]);
                                        }
                                    }
                                }
                                bool live_6_7 = num_splits > 1 && lse_max_4[0] > -CUDART_INF_F;
                                if (live_6_7) {
                                    float weight_149 = approx_exp2(lse_4[1] - lse_max_4[0]);
                                    if (bound_3[0] > 1) {
                                        den_4[0] = den_4[0] + weight_149;
                                        acc0_4[0] = __fmaf_rn(weight_149, p0_4[1], acc0_4[0]);
                                        acc1_4[0] = __fmaf_rn(weight_149, p1_4[1], acc1_4[0]);
                                    } else if (bound_3[1] > 1) {
                                        den_4[1] = den_4[1] + weight_149;
                                        acc0_4[1] = __fmaf_rn(weight_149, p0_4[1], acc0_4[1]);
                                        acc1_4[1] = __fmaf_rn(weight_149, p1_4[1], acc1_4[1]);
                                    } else {
                                        if (bound_3[2] > 1) {
                                            den_4[2] = den_4[2] + weight_149;
                                            acc0_4[2] = __fmaf_rn(weight_149, p0_4[1], acc0_4[2]);
                                            acc1_4[2] = __fmaf_rn(weight_149, p1_4[1], acc1_4[2]);
                                        } else {
                                            den_4[3] = den_4[3] + weight_149;
                                            acc0_4[3] = __fmaf_rn(weight_149, p0_4[1], acc0_4[3]);
                                            acc1_4[3] = __fmaf_rn(weight_149, p1_4[1], acc1_4[3]);
                                        }
                                    }
                                }
                                bool live_7_7 = num_splits > 2 && lse_max_4[0] > -CUDART_INF_F;
                                if (live_7_7) {
                                    float weight_150 = approx_exp2(lse_4[2] - lse_max_4[0]);
                                    if (bound_3[0] > 2) {
                                        den_4[0] = den_4[0] + weight_150;
                                        acc0_4[0] = __fmaf_rn(weight_150, p0_4[2], acc0_4[0]);
                                        acc1_4[0] = __fmaf_rn(weight_150, p1_4[2], acc1_4[0]);
                                    } else if (bound_3[1] > 2) {
                                        den_4[1] = den_4[1] + weight_150;
                                        acc0_4[1] = __fmaf_rn(weight_150, p0_4[2], acc0_4[1]);
                                        acc1_4[1] = __fmaf_rn(weight_150, p1_4[2], acc1_4[1]);
                                    } else {
                                        if (bound_3[2] > 2) {
                                            den_4[2] = den_4[2] + weight_150;
                                            acc0_4[2] = __fmaf_rn(weight_150, p0_4[2], acc0_4[2]);
                                            acc1_4[2] = __fmaf_rn(weight_150, p1_4[2], acc1_4[2]);
                                        } else {
                                            den_4[3] = den_4[3] + weight_150;
                                            acc0_4[3] = __fmaf_rn(weight_150, p0_4[2], acc0_4[3]);
                                            acc1_4[3] = __fmaf_rn(weight_150, p1_4[2], acc1_4[3]);
                                        }
                                    }
                                }
                                bool live_8_7 = num_splits > 3 && lse_max_4[0] > -CUDART_INF_F;
                                if (live_8_7) {
                                    float weight_151 = approx_exp2(lse_4[3] - lse_max_4[0]);
                                    if (bound_3[0] > 3) {
                                        den_4[0] = den_4[0] + weight_151;
                                        acc0_4[0] = __fmaf_rn(weight_151, p0_4[3], acc0_4[0]);
                                        acc1_4[0] = __fmaf_rn(weight_151, p1_4[3], acc1_4[0]);
                                    } else if (bound_3[1] > 3) {
                                        den_4[1] = den_4[1] + weight_151;
                                        acc0_4[1] = __fmaf_rn(weight_151, p0_4[3], acc0_4[1]);
                                        acc1_4[1] = __fmaf_rn(weight_151, p1_4[3], acc1_4[1]);
                                    } else {
                                        if (bound_3[2] > 3) {
                                            den_4[2] = den_4[2] + weight_151;
                                            acc0_4[2] = __fmaf_rn(weight_151, p0_4[3], acc0_4[2]);
                                            acc1_4[2] = __fmaf_rn(weight_151, p1_4[3], acc1_4[2]);
                                        } else {
                                            den_4[3] = den_4[3] + weight_151;
                                            acc0_4[3] = __fmaf_rn(weight_151, p0_4[3], acc0_4[3]);
                                            acc1_4[3] = __fmaf_rn(weight_151, p1_4[3], acc1_4[3]);
                                        }
                                    }
                                }
                                bool live_9_7 = num_splits > 0 && lse_max_4[1] > -CUDART_INF_F;
                                if (live_9_7) {
                                    float weight_152 = approx_exp2(lse_4[4] - lse_max_4[1]);
                                    if (bound_3[0] > 0) {
                                        den_4[4] = den_4[4] + weight_152;
                                        acc0_4[4] = __fmaf_rn(weight_152, p0_4[4], acc0_4[4]);
                                        acc1_4[4] = __fmaf_rn(weight_152, p1_4[4], acc1_4[4]);
                                    } else if (bound_3[1] > 0) {
                                        den_4[5] = den_4[5] + weight_152;
                                        acc0_4[5] = __fmaf_rn(weight_152, p0_4[4], acc0_4[5]);
                                        acc1_4[5] = __fmaf_rn(weight_152, p1_4[4], acc1_4[5]);
                                    } else {
                                        if (bound_3[2] > 0) {
                                            den_4[6] = den_4[6] + weight_152;
                                            acc0_4[6] = __fmaf_rn(weight_152, p0_4[4], acc0_4[6]);
                                            acc1_4[6] = __fmaf_rn(weight_152, p1_4[4], acc1_4[6]);
                                        } else {
                                            den_4[7] = den_4[7] + weight_152;
                                            acc0_4[7] = __fmaf_rn(weight_152, p0_4[4], acc0_4[7]);
                                            acc1_4[7] = __fmaf_rn(weight_152, p1_4[4], acc1_4[7]);
                                        }
                                    }
                                }
                                bool live_10_7 = num_splits > 1 && lse_max_4[1] > -CUDART_INF_F;
                                if (live_10_7) {
                                    float weight_153 = approx_exp2(lse_4[5] - lse_max_4[1]);
                                    if (bound_3[0] > 1) {
                                        den_4[4] = den_4[4] + weight_153;
                                        acc0_4[4] = __fmaf_rn(weight_153, p0_4[5], acc0_4[4]);
                                        acc1_4[4] = __fmaf_rn(weight_153, p1_4[5], acc1_4[4]);
                                    } else if (bound_3[1] > 1) {
                                        den_4[5] = den_4[5] + weight_153;
                                        acc0_4[5] = __fmaf_rn(weight_153, p0_4[5], acc0_4[5]);
                                        acc1_4[5] = __fmaf_rn(weight_153, p1_4[5], acc1_4[5]);
                                    } else {
                                        if (bound_3[2] > 1) {
                                            den_4[6] = den_4[6] + weight_153;
                                            acc0_4[6] = __fmaf_rn(weight_153, p0_4[5], acc0_4[6]);
                                            acc1_4[6] = __fmaf_rn(weight_153, p1_4[5], acc1_4[6]);
                                        } else {
                                            den_4[7] = den_4[7] + weight_153;
                                            acc0_4[7] = __fmaf_rn(weight_153, p0_4[5], acc0_4[7]);
                                            acc1_4[7] = __fmaf_rn(weight_153, p1_4[5], acc1_4[7]);
                                        }
                                    }
                                }
                                bool live_11_7 = num_splits > 2 && lse_max_4[1] > -CUDART_INF_F;
                                if (live_11_7) {
                                    float weight_154 = approx_exp2(lse_4[6] - lse_max_4[1]);
                                    if (bound_3[0] > 2) {
                                        den_4[4] = den_4[4] + weight_154;
                                        acc0_4[4] = __fmaf_rn(weight_154, p0_4[6], acc0_4[4]);
                                        acc1_4[4] = __fmaf_rn(weight_154, p1_4[6], acc1_4[4]);
                                    } else if (bound_3[1] > 2) {
                                        den_4[5] = den_4[5] + weight_154;
                                        acc0_4[5] = __fmaf_rn(weight_154, p0_4[6], acc0_4[5]);
                                        acc1_4[5] = __fmaf_rn(weight_154, p1_4[6], acc1_4[5]);
                                    } else {
                                        if (bound_3[2] > 2) {
                                            den_4[6] = den_4[6] + weight_154;
                                            acc0_4[6] = __fmaf_rn(weight_154, p0_4[6], acc0_4[6]);
                                            acc1_4[6] = __fmaf_rn(weight_154, p1_4[6], acc1_4[6]);
                                        } else {
                                            den_4[7] = den_4[7] + weight_154;
                                            acc0_4[7] = __fmaf_rn(weight_154, p0_4[6], acc0_4[7]);
                                            acc1_4[7] = __fmaf_rn(weight_154, p1_4[6], acc1_4[7]);
                                        }
                                    }
                                }
                                bool live_12_7 = num_splits > 3 && lse_max_4[1] > -CUDART_INF_F;
                                if (live_12_7) {
                                    float weight_155 = approx_exp2(lse_4[7] - lse_max_4[1]);
                                    if (bound_3[0] > 3) {
                                        den_4[4] = den_4[4] + weight_155;
                                        acc0_4[4] = __fmaf_rn(weight_155, p0_4[7], acc0_4[4]);
                                        acc1_4[4] = __fmaf_rn(weight_155, p1_4[7], acc1_4[4]);
                                    } else if (bound_3[1] > 3) {
                                        den_4[5] = den_4[5] + weight_155;
                                        acc0_4[5] = __fmaf_rn(weight_155, p0_4[7], acc0_4[5]);
                                        acc1_4[5] = __fmaf_rn(weight_155, p1_4[7], acc1_4[5]);
                                    } else {
                                        if (bound_3[2] > 3) {
                                            den_4[6] = den_4[6] + weight_155;
                                            acc0_4[6] = __fmaf_rn(weight_155, p0_4[7], acc0_4[6]);
                                            acc1_4[6] = __fmaf_rn(weight_155, p1_4[7], acc1_4[6]);
                                        } else {
                                            den_4[7] = den_4[7] + weight_155;
                                            acc0_4[7] = __fmaf_rn(weight_155, p0_4[7], acc0_4[7]);
                                            acc1_4[7] = __fmaf_rn(weight_155, p1_4[7], acc1_4[7]);
                                        }
                                    }
                                }
                                bool live_13_7 = num_splits > 0 && lse_max_4[2] > -CUDART_INF_F;
                                if (live_13_7) {
                                    float weight_156 = approx_exp2(lse_4[8] - lse_max_4[2]);
                                    if (bound_3[0] > 0) {
                                        den_4[8] = den_4[8] + weight_156;
                                        acc0_4[8] = __fmaf_rn(weight_156, p0_4[8], acc0_4[8]);
                                        acc1_4[8] = __fmaf_rn(weight_156, p1_4[8], acc1_4[8]);
                                    } else if (bound_3[1] > 0) {
                                        den_4[9] = den_4[9] + weight_156;
                                        acc0_4[9] = __fmaf_rn(weight_156, p0_4[8], acc0_4[9]);
                                        acc1_4[9] = __fmaf_rn(weight_156, p1_4[8], acc1_4[9]);
                                    } else {
                                        if (bound_3[2] > 0) {
                                            den_4[10] = den_4[10] + weight_156;
                                            acc0_4[10] = __fmaf_rn(weight_156, p0_4[8], acc0_4[10]);
                                            acc1_4[10] = __fmaf_rn(weight_156, p1_4[8], acc1_4[10]);
                                        } else {
                                            den_4[11] = den_4[11] + weight_156;
                                            acc0_4[11] = __fmaf_rn(weight_156, p0_4[8], acc0_4[11]);
                                            acc1_4[11] = __fmaf_rn(weight_156, p1_4[8], acc1_4[11]);
                                        }
                                    }
                                }
                                bool live_14_7 = num_splits > 1 && lse_max_4[2] > -CUDART_INF_F;
                                if (live_14_7) {
                                    float weight_157 = approx_exp2(lse_4[9] - lse_max_4[2]);
                                    if (bound_3[0] > 1) {
                                        den_4[8] = den_4[8] + weight_157;
                                        acc0_4[8] = __fmaf_rn(weight_157, p0_4[9], acc0_4[8]);
                                        acc1_4[8] = __fmaf_rn(weight_157, p1_4[9], acc1_4[8]);
                                    } else if (bound_3[1] > 1) {
                                        den_4[9] = den_4[9] + weight_157;
                                        acc0_4[9] = __fmaf_rn(weight_157, p0_4[9], acc0_4[9]);
                                        acc1_4[9] = __fmaf_rn(weight_157, p1_4[9], acc1_4[9]);
                                    } else {
                                        if (bound_3[2] > 1) {
                                            den_4[10] = den_4[10] + weight_157;
                                            acc0_4[10] = __fmaf_rn(weight_157, p0_4[9], acc0_4[10]);
                                            acc1_4[10] = __fmaf_rn(weight_157, p1_4[9], acc1_4[10]);
                                        } else {
                                            den_4[11] = den_4[11] + weight_157;
                                            acc0_4[11] = __fmaf_rn(weight_157, p0_4[9], acc0_4[11]);
                                            acc1_4[11] = __fmaf_rn(weight_157, p1_4[9], acc1_4[11]);
                                        }
                                    }
                                }
                                bool live_15_4 = num_splits > 2 && lse_max_4[2] > -CUDART_INF_F;
                                if (live_15_4) {
                                    float weight_158 = approx_exp2(lse_4[10] - lse_max_4[2]);
                                    if (bound_3[0] > 2) {
                                        den_4[8] = den_4[8] + weight_158;
                                        acc0_4[8] = __fmaf_rn(weight_158, p0_4[10], acc0_4[8]);
                                        acc1_4[8] = __fmaf_rn(weight_158, p1_4[10], acc1_4[8]);
                                    } else if (bound_3[1] > 2) {
                                        den_4[9] = den_4[9] + weight_158;
                                        acc0_4[9] = __fmaf_rn(weight_158, p0_4[10], acc0_4[9]);
                                        acc1_4[9] = __fmaf_rn(weight_158, p1_4[10], acc1_4[9]);
                                    } else {
                                        if (bound_3[2] > 2) {
                                            den_4[10] = den_4[10] + weight_158;
                                            acc0_4[10] = __fmaf_rn(weight_158, p0_4[10], acc0_4[10]);
                                            acc1_4[10] = __fmaf_rn(weight_158, p1_4[10], acc1_4[10]);
                                        } else {
                                            den_4[11] = den_4[11] + weight_158;
                                            acc0_4[11] = __fmaf_rn(weight_158, p0_4[10], acc0_4[11]);
                                            acc1_4[11] = __fmaf_rn(weight_158, p1_4[10], acc1_4[11]);
                                        }
                                    }
                                }
                                bool live_16_4 = num_splits > 3 && lse_max_4[2] > -CUDART_INF_F;
                                if (live_16_4) {
                                    float weight_159 = approx_exp2(lse_4[11] - lse_max_4[2]);
                                    if (bound_3[0] > 3) {
                                        den_4[8] = den_4[8] + weight_159;
                                        acc0_4[8] = __fmaf_rn(weight_159, p0_4[11], acc0_4[8]);
                                        acc1_4[8] = __fmaf_rn(weight_159, p1_4[11], acc1_4[8]);
                                    } else if (bound_3[1] > 3) {
                                        den_4[9] = den_4[9] + weight_159;
                                        acc0_4[9] = __fmaf_rn(weight_159, p0_4[11], acc0_4[9]);
                                        acc1_4[9] = __fmaf_rn(weight_159, p1_4[11], acc1_4[9]);
                                    } else {
                                        if (bound_3[2] > 3) {
                                            den_4[10] = den_4[10] + weight_159;
                                            acc0_4[10] = __fmaf_rn(weight_159, p0_4[11], acc0_4[10]);
                                            acc1_4[10] = __fmaf_rn(weight_159, p1_4[11], acc1_4[10]);
                                        } else {
                                            den_4[11] = den_4[11] + weight_159;
                                            acc0_4[11] = __fmaf_rn(weight_159, p0_4[11], acc0_4[11]);
                                            acc1_4[11] = __fmaf_rn(weight_159, p1_4[11], acc1_4[11]);
                                        }
                                    }
                                }
                                bool live_17_3 = num_splits > 0 && lse_max_4[3] > -CUDART_INF_F;
                                if (live_17_3) {
                                    float weight_160 = approx_exp2(lse_4[12] - lse_max_4[3]);
                                    if (bound_3[0] > 0) {
                                        den_4[12] = den_4[12] + weight_160;
                                        acc0_4[12] = __fmaf_rn(weight_160, p0_4[12], acc0_4[12]);
                                        acc1_4[12] = __fmaf_rn(weight_160, p1_4[12], acc1_4[12]);
                                    } else if (bound_3[1] > 0) {
                                        den_4[13] = den_4[13] + weight_160;
                                        acc0_4[13] = __fmaf_rn(weight_160, p0_4[12], acc0_4[13]);
                                        acc1_4[13] = __fmaf_rn(weight_160, p1_4[12], acc1_4[13]);
                                    } else {
                                        if (bound_3[2] > 0) {
                                            den_4[14] = den_4[14] + weight_160;
                                            acc0_4[14] = __fmaf_rn(weight_160, p0_4[12], acc0_4[14]);
                                            acc1_4[14] = __fmaf_rn(weight_160, p1_4[12], acc1_4[14]);
                                        } else {
                                            den_4[15] = den_4[15] + weight_160;
                                            acc0_4[15] = __fmaf_rn(weight_160, p0_4[12], acc0_4[15]);
                                            acc1_4[15] = __fmaf_rn(weight_160, p1_4[12], acc1_4[15]);
                                        }
                                    }
                                }
                                bool live_18_3 = num_splits > 1 && lse_max_4[3] > -CUDART_INF_F;
                                if (live_18_3) {
                                    float weight_161 = approx_exp2(lse_4[13] - lse_max_4[3]);
                                    if (bound_3[0] > 1) {
                                        den_4[12] = den_4[12] + weight_161;
                                        acc0_4[12] = __fmaf_rn(weight_161, p0_4[13], acc0_4[12]);
                                        acc1_4[12] = __fmaf_rn(weight_161, p1_4[13], acc1_4[12]);
                                    } else if (bound_3[1] > 1) {
                                        den_4[13] = den_4[13] + weight_161;
                                        acc0_4[13] = __fmaf_rn(weight_161, p0_4[13], acc0_4[13]);
                                        acc1_4[13] = __fmaf_rn(weight_161, p1_4[13], acc1_4[13]);
                                    } else {
                                        if (bound_3[2] > 1) {
                                            den_4[14] = den_4[14] + weight_161;
                                            acc0_4[14] = __fmaf_rn(weight_161, p0_4[13], acc0_4[14]);
                                            acc1_4[14] = __fmaf_rn(weight_161, p1_4[13], acc1_4[14]);
                                        } else {
                                            den_4[15] = den_4[15] + weight_161;
                                            acc0_4[15] = __fmaf_rn(weight_161, p0_4[13], acc0_4[15]);
                                            acc1_4[15] = __fmaf_rn(weight_161, p1_4[13], acc1_4[15]);
                                        }
                                    }
                                }
                                bool live_19_2 = num_splits > 2 && lse_max_4[3] > -CUDART_INF_F;
                                if (live_19_2) {
                                    float weight_162 = approx_exp2(lse_4[14] - lse_max_4[3]);
                                    if (bound_3[0] > 2) {
                                        den_4[12] = den_4[12] + weight_162;
                                        acc0_4[12] = __fmaf_rn(weight_162, p0_4[14], acc0_4[12]);
                                        acc1_4[12] = __fmaf_rn(weight_162, p1_4[14], acc1_4[12]);
                                    } else if (bound_3[1] > 2) {
                                        den_4[13] = den_4[13] + weight_162;
                                        acc0_4[13] = __fmaf_rn(weight_162, p0_4[14], acc0_4[13]);
                                        acc1_4[13] = __fmaf_rn(weight_162, p1_4[14], acc1_4[13]);
                                    } else {
                                        if (bound_3[2] > 2) {
                                            den_4[14] = den_4[14] + weight_162;
                                            acc0_4[14] = __fmaf_rn(weight_162, p0_4[14], acc0_4[14]);
                                            acc1_4[14] = __fmaf_rn(weight_162, p1_4[14], acc1_4[14]);
                                        } else {
                                            den_4[15] = den_4[15] + weight_162;
                                            acc0_4[15] = __fmaf_rn(weight_162, p0_4[14], acc0_4[15]);
                                            acc1_4[15] = __fmaf_rn(weight_162, p1_4[14], acc1_4[15]);
                                        }
                                    }
                                }
                                bool live_20_2 = num_splits > 3 && lse_max_4[3] > -CUDART_INF_F;
                                if (live_20_2) {
                                    float weight_163 = approx_exp2(lse_4[15] - lse_max_4[3]);
                                    if (bound_3[0] > 3) {
                                        den_4[12] = den_4[12] + weight_163;
                                        acc0_4[12] = __fmaf_rn(weight_163, p0_4[15], acc0_4[12]);
                                        acc1_4[12] = __fmaf_rn(weight_163, p1_4[15], acc1_4[12]);
                                    } else if (bound_3[1] > 3) {
                                        den_4[13] = den_4[13] + weight_163;
                                        acc0_4[13] = __fmaf_rn(weight_163, p0_4[15], acc0_4[13]);
                                        acc1_4[13] = __fmaf_rn(weight_163, p1_4[15], acc1_4[13]);
                                    } else {
                                        if (bound_3[2] > 3) {
                                            den_4[14] = den_4[14] + weight_163;
                                            acc0_4[14] = __fmaf_rn(weight_163, p0_4[15], acc0_4[14]);
                                            acc1_4[14] = __fmaf_rn(weight_163, p1_4[15], acc1_4[14]);
                                        } else {
                                            den_4[15] = den_4[15] + weight_163;
                                            acc0_4[15] = __fmaf_rn(weight_163, p0_4[15], acc0_4[15]);
                                            acc1_4[15] = __fmaf_rn(weight_163, p1_4[15], acc1_4[15]);
                                        }
                                    }
                                }
                                if (valid_4[0] != 0) {
                                    float den_k_6 = den_4[0];
                                    float acc0_k_6 = acc0_4[0];
                                    float acc1_k_6 = acc1_4[0];
                                    den_k_6 = den_k_6 + den_4[1];
                                    acc0_k_6 = acc0_k_6 + acc0_4[1];
                                    acc1_k_6 = acc1_k_6 + acc1_4[1];
                                    den_k_6 = den_k_6 + den_4[2];
                                    acc0_k_6 = acc0_k_6 + acc0_4[2];
                                    acc1_k_6 = acc1_k_6 + acc1_4[2];
                                    den_k_6 = den_k_6 + den_4[3];
                                    acc0_k_6 = acc0_k_6 + acc0_4[3];
                                    acc1_k_6 = acc1_k_6 + acc1_4[3];
                                    float inv_8 = 0.0f;
                                    if (den_k_6 > 0.0f) {
                                        float _rcp_360 = approx_rcp(den_k_6);
                                        inv_8 = _rcp_360;
                                    }
                                    float attn0_6 = (float)(__nv_bfloat16)(acc0_k_6 * inv_8);
                                    float attn1_6 = (float)(__nv_bfloat16)(acc1_k_6 * inv_8);
                                    float _exp2_362 = approx_exp2((-gate0_4[0]) * 1.442695f);
                                    float _rcp_361 = approx_rcp(1.0f + _exp2_362);
                                    float result0_6 = attn0_6 * _rcp_361;
                                    float _exp2_364 = approx_exp2((-gate1_4[0]) * 1.442695f);
                                    float _rcp_363 = approx_rcp(1.0f + _exp2_364);
                                    float result1_6 = attn1_6 * _rcp_363;
                                    __nv_bfloat162 _bit_cast_src_365 = __float22bfloat162_rn(make_float2(result0_6, result1_6));
                                    word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_365)[0];
                                    long long out_offset_6 = row_i64 * out_stride_row + qo_head_6[0] * out_stride_head + dim_4[0];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_6) + (0)) = word_4[0];
                                }
                                if (valid_4[1] != 0) {
                                    float den_k_7 = den_4[4];
                                    float acc0_k_7 = acc0_4[4];
                                    float acc1_k_7 = acc1_4[4];
                                    den_k_7 = den_k_7 + den_4[5];
                                    acc0_k_7 = acc0_k_7 + acc0_4[5];
                                    acc1_k_7 = acc1_k_7 + acc1_4[5];
                                    den_k_7 = den_k_7 + den_4[6];
                                    acc0_k_7 = acc0_k_7 + acc0_4[6];
                                    acc1_k_7 = acc1_k_7 + acc1_4[6];
                                    den_k_7 = den_k_7 + den_4[7];
                                    acc0_k_7 = acc0_k_7 + acc0_4[7];
                                    acc1_k_7 = acc1_k_7 + acc1_4[7];
                                    float inv_9 = 0.0f;
                                    if (den_k_7 > 0.0f) {
                                        float _rcp_366 = approx_rcp(den_k_7);
                                        inv_9 = _rcp_366;
                                    }
                                    float attn0_7 = (float)(__nv_bfloat16)(acc0_k_7 * inv_9);
                                    float attn1_7 = (float)(__nv_bfloat16)(acc1_k_7 * inv_9);
                                    float _exp2_368 = approx_exp2((-gate0_4[1]) * 1.442695f);
                                    float _rcp_367 = approx_rcp(1.0f + _exp2_368);
                                    float result0_7 = attn0_7 * _rcp_367;
                                    float _exp2_370 = approx_exp2((-gate1_4[1]) * 1.442695f);
                                    float _rcp_369 = approx_rcp(1.0f + _exp2_370);
                                    float result1_7 = attn1_7 * _rcp_369;
                                    __nv_bfloat162 _bit_cast_src_371 = __float22bfloat162_rn(make_float2(result0_7, result1_7));
                                    word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_371)[0];
                                    long long out_offset_7 = row_i64 * out_stride_row + qo_head_6[1] * out_stride_head + dim_4[1];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_7) + (0)) = word_4[0];
                                }
                                if (valid_4[2] != 0) {
                                    float den_k_8 = den_4[8];
                                    float acc0_k_8 = acc0_4[8];
                                    float acc1_k_8 = acc1_4[8];
                                    den_k_8 = den_k_8 + den_4[9];
                                    acc0_k_8 = acc0_k_8 + acc0_4[9];
                                    acc1_k_8 = acc1_k_8 + acc1_4[9];
                                    den_k_8 = den_k_8 + den_4[10];
                                    acc0_k_8 = acc0_k_8 + acc0_4[10];
                                    acc1_k_8 = acc1_k_8 + acc1_4[10];
                                    den_k_8 = den_k_8 + den_4[11];
                                    acc0_k_8 = acc0_k_8 + acc0_4[11];
                                    acc1_k_8 = acc1_k_8 + acc1_4[11];
                                    float inv_10 = 0.0f;
                                    if (den_k_8 > 0.0f) {
                                        float _rcp_372 = approx_rcp(den_k_8);
                                        inv_10 = _rcp_372;
                                    }
                                    float attn0_8 = (float)(__nv_bfloat16)(acc0_k_8 * inv_10);
                                    float attn1_8 = (float)(__nv_bfloat16)(acc1_k_8 * inv_10);
                                    float _exp2_374 = approx_exp2((-gate0_4[2]) * 1.442695f);
                                    float _rcp_373 = approx_rcp(1.0f + _exp2_374);
                                    float result0_8 = attn0_8 * _rcp_373;
                                    float _exp2_376 = approx_exp2((-gate1_4[2]) * 1.442695f);
                                    float _rcp_375 = approx_rcp(1.0f + _exp2_376);
                                    float result1_8 = attn1_8 * _rcp_375;
                                    __nv_bfloat162 _bit_cast_src_377 = __float22bfloat162_rn(make_float2(result0_8, result1_8));
                                    word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_377)[0];
                                    long long out_offset_8 = row_i64 * out_stride_row + qo_head_6[2] * out_stride_head + dim_4[2];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_8) + (0)) = word_4[0];
                                }
                                if (valid_4[3] != 0) {
                                    float den_k_9 = den_4[12];
                                    float acc0_k_9 = acc0_4[12];
                                    float acc1_k_9 = acc1_4[12];
                                    den_k_9 = den_k_9 + den_4[13];
                                    acc0_k_9 = acc0_k_9 + acc0_4[13];
                                    acc1_k_9 = acc1_k_9 + acc1_4[13];
                                    den_k_9 = den_k_9 + den_4[14];
                                    acc0_k_9 = acc0_k_9 + acc0_4[14];
                                    acc1_k_9 = acc1_k_9 + acc1_4[14];
                                    den_k_9 = den_k_9 + den_4[15];
                                    acc0_k_9 = acc0_k_9 + acc0_4[15];
                                    acc1_k_9 = acc1_k_9 + acc1_4[15];
                                    float inv_11 = 0.0f;
                                    if (den_k_9 > 0.0f) {
                                        float _rcp_378 = approx_rcp(den_k_9);
                                        inv_11 = _rcp_378;
                                    }
                                    float attn0_9 = (float)(__nv_bfloat16)(acc0_k_9 * inv_11);
                                    float attn1_9 = (float)(__nv_bfloat16)(acc1_k_9 * inv_11);
                                    float _exp2_380 = approx_exp2((-gate0_4[3]) * 1.442695f);
                                    float _rcp_379 = approx_rcp(1.0f + _exp2_380);
                                    float result0_9 = attn0_9 * _rcp_379;
                                    float _exp2_382 = approx_exp2((-gate1_4[3]) * 1.442695f);
                                    float _rcp_381 = approx_rcp(1.0f + _exp2_382);
                                    float result1_9 = attn1_9 * _rcp_381;
                                    __nv_bfloat162 _bit_cast_src_383 = __float22bfloat162_rn(make_float2(result0_9, result1_9));
                                    word_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_383)[0];
                                    long long out_offset_9 = row_i64 * out_stride_row + qo_head_6[3] * out_stride_head + dim_4[3];
                                    *(reinterpret_cast<unsigned int*>(out + out_offset_9) + (0)) = word_4[0];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: transform_k ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // transform_k_main
            int tid_0_1 = tid;
            int qd = (tid_0_1 >> 5) - 4;
            int lane_2 = tid_0_1 & 31;
            int tk = 16 * qd + (lane_2 & 15);
            int hh = lane_2 >> 4;
            unsigned int raw_addr = raw_u8_addr;
            unsigned int k_lane0 = taddr + 64 + (unsigned int)(32 * qd << 16);
            #pragma unroll 1
            for (int n_1 = 0; n_1 < n_tiles; n_1++) {
                int st = n_1 % 4;
                int par = n_1 & 1;
                int _stage_raw_full_k = (int)((unsigned int)n_1 % 4);
                int _parity_raw_full_k = (int)((unsigned int)n_1 / 4 % 2);
                mbarrier_wait(raw_full_k_addr + (_stage_raw_full_k) * 8, _parity_raw_full_k);
                int _stage_k_empty = (int)((unsigned int)n_1 % 2);
                int _parity_k_empty = (int)((unsigned int)n_1 / 2 % 2) ^ 1;
                mbarrier_wait(k_empty_addr + (_stage_k_empty) * 8, _parity_k_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int row_addr = raw_addr + (unsigned int)(st * 18432 + tk * 144);
                unsigned int raw_1[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3]))
                    : "r"(row_addr + (unsigned int)(16 * (4 * hh))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4) + 3]))
                    : "r"(row_addr + (unsigned int)(16 * (4 * hh + 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(8) + 3]))
                    : "r"(row_addr + (unsigned int)(16 * (4 * hh + 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(12) + 3]))
                    : "r"(row_addr + (unsigned int)(16 * (4 * hh + 3))));
                unsigned int sw[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw[(0) + 1]))
                    : "r"(row_addr + (unsigned int)(128 + 8 * hh)));
                unsigned int sfa[2];
                unsigned int sfb[2];
                uint32_t _prmt_0;
                asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_0) : "r"(sw[0]), "r"(sw[0]));
                uint32_t _bf16x2_fma_0;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_0) : "r"(_prmt_0 & 2147516416u | (_prmt_0 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfa[0] = _bf16x2_fma_0;
                uint32_t _bf16x2_fma_1;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_1) : "r"(_prmt_0 << 8 & 2147516416u | (_prmt_0 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfa[1] = _bf16x2_fma_1;
                uint32_t _prmt_1;
                asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_1) : "r"(sw[1]), "r"(sw[1]));
                uint32_t _bf16x2_fma_2;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_2) : "r"(_prmt_1 & 2147516416u | (_prmt_1 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfb[0] = _bf16x2_fma_2;
                uint32_t _bf16x2_fma_3;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_3) : "r"(_prmt_1 << 8 & 2147516416u | (_prmt_1 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfb[1] = _bf16x2_fma_3;
                unsigned int sc[8];
                uint32_t _prmt_0_1;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_0_1) : "r"(sfa[0]), "r"(0));
                sc[0] = _prmt_0_1;
                uint32_t _prmt_1_1;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_1_1) : "r"(sfb[0]), "r"(0));
                sc[4] = _prmt_1_1;
                uint32_t _prmt_2;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_2) : "r"(sfa[0]), "r"(0));
                sc[1] = _prmt_2;
                uint32_t _prmt_3;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_3) : "r"(sfb[0]), "r"(0));
                sc[5] = _prmt_3;
                uint32_t _prmt_4;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_4) : "r"(sfa[1]), "r"(0));
                sc[2] = _prmt_4;
                uint32_t _prmt_5;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_5) : "r"(sfb[1]), "r"(0));
                sc[6] = _prmt_5;
                uint32_t _prmt_6;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_6) : "r"(sfa[1]), "r"(0));
                sc[3] = _prmt_6;
                uint32_t _prmt_7;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_7) : "r"(sfb[1]), "r"(0));
                sc[7] = _prmt_7;
                unsigned int kdst = k_lane0 + (unsigned int)(16 * par << 16);
                unsigned int w16[16];
                unsigned int d[4];
                uint32_t _fp4_bf16x8_0[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw_1[0])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw_1[0])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_0[0], hi = (uint16_t)(_fp4_bf16x8_0[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_0[1], hi = (uint16_t)(_fp4_bf16x8_0[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_0[2], hi = (uint16_t)(_fp4_bf16x8_0[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_0[3], hi = (uint16_t)(_fp4_bf16x8_0[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d[0] = _fp4_bf16x8_0[0];
                d[1] = _fp4_bf16x8_0[1];
                d[2] = _fp4_bf16x8_0[2];
                d[3] = _fp4_bf16x8_0[3];
                uint32_t _bf16x2_mul_0;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(d[0]), "r"(sc[0]));
                w16[0] = _bf16x2_mul_0;
                uint32_t _bf16x2_mul_1;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(d[1]), "r"(sc[0]));
                w16[1] = _bf16x2_mul_1;
                uint32_t _bf16x2_mul_2;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(d[2]), "r"(sc[0]));
                w16[2] = _bf16x2_mul_2;
                uint32_t _bf16x2_mul_3;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(d[3]), "r"(sc[0]));
                w16[3] = _bf16x2_mul_3;
                unsigned int d_0[4];
                uint32_t _fp4_bf16x8_1[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw_1[1])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw_1[1])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_1[0], hi = (uint16_t)(_fp4_bf16x8_1[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_1[1], hi = (uint16_t)(_fp4_bf16x8_1[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_1[2], hi = (uint16_t)(_fp4_bf16x8_1[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_1[3], hi = (uint16_t)(_fp4_bf16x8_1[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_0[0] = _fp4_bf16x8_1[0];
                d_0[1] = _fp4_bf16x8_1[1];
                d_0[2] = _fp4_bf16x8_1[2];
                d_0[3] = _fp4_bf16x8_1[3];
                uint32_t _bf16x2_mul_4;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(d_0[0]), "r"(sc[0]));
                w16[4] = _bf16x2_mul_4;
                uint32_t _bf16x2_mul_5;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(d_0[1]), "r"(sc[0]));
                w16[5] = _bf16x2_mul_5;
                uint32_t _bf16x2_mul_6;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(d_0[2]), "r"(sc[0]));
                w16[6] = _bf16x2_mul_6;
                uint32_t _bf16x2_mul_7;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(d_0[3]), "r"(sc[0]));
                w16[7] = _bf16x2_mul_7;
                unsigned int d_1[4];
                uint32_t _fp4_bf16x8_2[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(raw_1[2])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(raw_1[2])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_2[0], hi = (uint16_t)(_fp4_bf16x8_2[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_2[1], hi = (uint16_t)(_fp4_bf16x8_2[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_2[2], hi = (uint16_t)(_fp4_bf16x8_2[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_2[3], hi = (uint16_t)(_fp4_bf16x8_2[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_1[0] = _fp4_bf16x8_2[0];
                d_1[1] = _fp4_bf16x8_2[1];
                d_1[2] = _fp4_bf16x8_2[2];
                d_1[3] = _fp4_bf16x8_2[3];
                uint32_t _bf16x2_mul_8;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(d_1[0]), "r"(sc[1]));
                w16[8] = _bf16x2_mul_8;
                uint32_t _bf16x2_mul_9;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(d_1[1]), "r"(sc[1]));
                w16[9] = _bf16x2_mul_9;
                uint32_t _bf16x2_mul_10;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(d_1[2]), "r"(sc[1]));
                w16[10] = _bf16x2_mul_10;
                uint32_t _bf16x2_mul_11;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(d_1[3]), "r"(sc[1]));
                w16[11] = _bf16x2_mul_11;
                unsigned int d_2[4];
                uint32_t _fp4_bf16x8_3[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(raw_1[3])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(raw_1[3])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_3[0], hi = (uint16_t)(_fp4_bf16x8_3[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_3[1], hi = (uint16_t)(_fp4_bf16x8_3[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_3[2], hi = (uint16_t)(_fp4_bf16x8_3[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_3[3], hi = (uint16_t)(_fp4_bf16x8_3[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_2[0] = _fp4_bf16x8_3[0];
                d_2[1] = _fp4_bf16x8_3[1];
                d_2[2] = _fp4_bf16x8_3[2];
                d_2[3] = _fp4_bf16x8_3[3];
                uint32_t _bf16x2_mul_12;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(d_2[0]), "r"(sc[1]));
                w16[12] = _bf16x2_mul_12;
                uint32_t _bf16x2_mul_13;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(d_2[1]), "r"(sc[1]));
                w16[13] = _bf16x2_mul_13;
                uint32_t _bf16x2_mul_14;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(d_2[2]), "r"(sc[1]));
                w16[14] = _bf16x2_mul_14;
                uint32_t _bf16x2_mul_15;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(d_2[3]), "r"(sc[1]));
                w16[15] = _bf16x2_mul_15;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(kdst), "r"(*reinterpret_cast<const uint32_t*>(&w16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16[15])));
                unsigned int w16_3[16];
                unsigned int d_4[4];
                uint32_t _fp4_bf16x8_4[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(raw_1[4])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(raw_1[4])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_4[0], hi = (uint16_t)(_fp4_bf16x8_4[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_4[1], hi = (uint16_t)(_fp4_bf16x8_4[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_4[2], hi = (uint16_t)(_fp4_bf16x8_4[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_4[3], hi = (uint16_t)(_fp4_bf16x8_4[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_4[0] = _fp4_bf16x8_4[0];
                d_4[1] = _fp4_bf16x8_4[1];
                d_4[2] = _fp4_bf16x8_4[2];
                d_4[3] = _fp4_bf16x8_4[3];
                uint32_t _bf16x2_mul_16;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(d_4[0]), "r"(sc[2]));
                w16_3[0] = _bf16x2_mul_16;
                uint32_t _bf16x2_mul_17;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(d_4[1]), "r"(sc[2]));
                w16_3[1] = _bf16x2_mul_17;
                uint32_t _bf16x2_mul_18;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(d_4[2]), "r"(sc[2]));
                w16_3[2] = _bf16x2_mul_18;
                uint32_t _bf16x2_mul_19;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(d_4[3]), "r"(sc[2]));
                w16_3[3] = _bf16x2_mul_19;
                unsigned int d_5[4];
                uint32_t _fp4_bf16x8_5[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(raw_1[5])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(raw_1[5])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_5[0], hi = (uint16_t)(_fp4_bf16x8_5[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_5[1], hi = (uint16_t)(_fp4_bf16x8_5[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_5[2], hi = (uint16_t)(_fp4_bf16x8_5[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_5[3], hi = (uint16_t)(_fp4_bf16x8_5[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_5[0] = _fp4_bf16x8_5[0];
                d_5[1] = _fp4_bf16x8_5[1];
                d_5[2] = _fp4_bf16x8_5[2];
                d_5[3] = _fp4_bf16x8_5[3];
                uint32_t _bf16x2_mul_20;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(d_5[0]), "r"(sc[2]));
                w16_3[4] = _bf16x2_mul_20;
                uint32_t _bf16x2_mul_21;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(d_5[1]), "r"(sc[2]));
                w16_3[5] = _bf16x2_mul_21;
                uint32_t _bf16x2_mul_22;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(d_5[2]), "r"(sc[2]));
                w16_3[6] = _bf16x2_mul_22;
                uint32_t _bf16x2_mul_23;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(d_5[3]), "r"(sc[2]));
                w16_3[7] = _bf16x2_mul_23;
                unsigned int d_6[4];
                uint32_t _fp4_bf16x8_6[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(raw_1[6])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(raw_1[6])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_6[0], hi = (uint16_t)(_fp4_bf16x8_6[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_6[1], hi = (uint16_t)(_fp4_bf16x8_6[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_6[2], hi = (uint16_t)(_fp4_bf16x8_6[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_6[3], hi = (uint16_t)(_fp4_bf16x8_6[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_6[0] = _fp4_bf16x8_6[0];
                d_6[1] = _fp4_bf16x8_6[1];
                d_6[2] = _fp4_bf16x8_6[2];
                d_6[3] = _fp4_bf16x8_6[3];
                uint32_t _bf16x2_mul_24;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(d_6[0]), "r"(sc[3]));
                w16_3[8] = _bf16x2_mul_24;
                uint32_t _bf16x2_mul_25;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(d_6[1]), "r"(sc[3]));
                w16_3[9] = _bf16x2_mul_25;
                uint32_t _bf16x2_mul_26;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(d_6[2]), "r"(sc[3]));
                w16_3[10] = _bf16x2_mul_26;
                uint32_t _bf16x2_mul_27;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(d_6[3]), "r"(sc[3]));
                w16_3[11] = _bf16x2_mul_27;
                unsigned int d_7[4];
                uint32_t _fp4_bf16x8_7[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(raw_1[7])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(raw_1[7])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_7[0], hi = (uint16_t)(_fp4_bf16x8_7[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_7[1], hi = (uint16_t)(_fp4_bf16x8_7[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_7[2], hi = (uint16_t)(_fp4_bf16x8_7[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_7[3], hi = (uint16_t)(_fp4_bf16x8_7[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_7[0] = _fp4_bf16x8_7[0];
                d_7[1] = _fp4_bf16x8_7[1];
                d_7[2] = _fp4_bf16x8_7[2];
                d_7[3] = _fp4_bf16x8_7[3];
                uint32_t _bf16x2_mul_28;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(d_7[0]), "r"(sc[3]));
                w16_3[12] = _bf16x2_mul_28;
                uint32_t _bf16x2_mul_29;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(d_7[1]), "r"(sc[3]));
                w16_3[13] = _bf16x2_mul_29;
                uint32_t _bf16x2_mul_30;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(d_7[2]), "r"(sc[3]));
                w16_3[14] = _bf16x2_mul_30;
                uint32_t _bf16x2_mul_31;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(d_7[3]), "r"(sc[3]));
                w16_3[15] = _bf16x2_mul_31;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(kdst + 16), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_3[15])));
                unsigned int w16_8[16];
                unsigned int d_9[4];
                uint32_t _fp4_bf16x8_8[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(raw_1[8])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(raw_1[8])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_8[0], hi = (uint16_t)(_fp4_bf16x8_8[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_8[1], hi = (uint16_t)(_fp4_bf16x8_8[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_8[2], hi = (uint16_t)(_fp4_bf16x8_8[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_8[3], hi = (uint16_t)(_fp4_bf16x8_8[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_9[0] = _fp4_bf16x8_8[0];
                d_9[1] = _fp4_bf16x8_8[1];
                d_9[2] = _fp4_bf16x8_8[2];
                d_9[3] = _fp4_bf16x8_8[3];
                uint32_t _bf16x2_mul_32;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(d_9[0]), "r"(sc[4]));
                w16_8[0] = _bf16x2_mul_32;
                uint32_t _bf16x2_mul_33;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(d_9[1]), "r"(sc[4]));
                w16_8[1] = _bf16x2_mul_33;
                uint32_t _bf16x2_mul_34;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(d_9[2]), "r"(sc[4]));
                w16_8[2] = _bf16x2_mul_34;
                uint32_t _bf16x2_mul_35;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(d_9[3]), "r"(sc[4]));
                w16_8[3] = _bf16x2_mul_35;
                unsigned int d_10[4];
                uint32_t _fp4_bf16x8_9[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(raw_1[9])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(raw_1[9])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_9[0], hi = (uint16_t)(_fp4_bf16x8_9[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_9[1], hi = (uint16_t)(_fp4_bf16x8_9[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_9[2], hi = (uint16_t)(_fp4_bf16x8_9[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_9[3], hi = (uint16_t)(_fp4_bf16x8_9[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_10[0] = _fp4_bf16x8_9[0];
                d_10[1] = _fp4_bf16x8_9[1];
                d_10[2] = _fp4_bf16x8_9[2];
                d_10[3] = _fp4_bf16x8_9[3];
                uint32_t _bf16x2_mul_36;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(d_10[0]), "r"(sc[4]));
                w16_8[4] = _bf16x2_mul_36;
                uint32_t _bf16x2_mul_37;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(d_10[1]), "r"(sc[4]));
                w16_8[5] = _bf16x2_mul_37;
                uint32_t _bf16x2_mul_38;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(d_10[2]), "r"(sc[4]));
                w16_8[6] = _bf16x2_mul_38;
                uint32_t _bf16x2_mul_39;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(d_10[3]), "r"(sc[4]));
                w16_8[7] = _bf16x2_mul_39;
                unsigned int d_11[4];
                uint32_t _fp4_bf16x8_10[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(raw_1[10])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(raw_1[10])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_10[0], hi = (uint16_t)(_fp4_bf16x8_10[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_10[1], hi = (uint16_t)(_fp4_bf16x8_10[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_10[2], hi = (uint16_t)(_fp4_bf16x8_10[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_10[3], hi = (uint16_t)(_fp4_bf16x8_10[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_11[0] = _fp4_bf16x8_10[0];
                d_11[1] = _fp4_bf16x8_10[1];
                d_11[2] = _fp4_bf16x8_10[2];
                d_11[3] = _fp4_bf16x8_10[3];
                uint32_t _bf16x2_mul_40;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(d_11[0]), "r"(sc[5]));
                w16_8[8] = _bf16x2_mul_40;
                uint32_t _bf16x2_mul_41;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(d_11[1]), "r"(sc[5]));
                w16_8[9] = _bf16x2_mul_41;
                uint32_t _bf16x2_mul_42;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(d_11[2]), "r"(sc[5]));
                w16_8[10] = _bf16x2_mul_42;
                uint32_t _bf16x2_mul_43;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(d_11[3]), "r"(sc[5]));
                w16_8[11] = _bf16x2_mul_43;
                unsigned int d_12[4];
                uint32_t _fp4_bf16x8_11[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(raw_1[11])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(raw_1[11])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_11[0], hi = (uint16_t)(_fp4_bf16x8_11[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_11[1], hi = (uint16_t)(_fp4_bf16x8_11[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_11[2], hi = (uint16_t)(_fp4_bf16x8_11[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_11[3], hi = (uint16_t)(_fp4_bf16x8_11[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_12[0] = _fp4_bf16x8_11[0];
                d_12[1] = _fp4_bf16x8_11[1];
                d_12[2] = _fp4_bf16x8_11[2];
                d_12[3] = _fp4_bf16x8_11[3];
                uint32_t _bf16x2_mul_44;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(d_12[0]), "r"(sc[5]));
                w16_8[12] = _bf16x2_mul_44;
                uint32_t _bf16x2_mul_45;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(d_12[1]), "r"(sc[5]));
                w16_8[13] = _bf16x2_mul_45;
                uint32_t _bf16x2_mul_46;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(d_12[2]), "r"(sc[5]));
                w16_8[14] = _bf16x2_mul_46;
                uint32_t _bf16x2_mul_47;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(d_12[3]), "r"(sc[5]));
                w16_8[15] = _bf16x2_mul_47;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(kdst + 32), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_8[15])));
                unsigned int w16_13[16];
                unsigned int d_14[4];
                uint32_t _fp4_bf16x8_12[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(raw_1[12])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(raw_1[12])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_12[0], hi = (uint16_t)(_fp4_bf16x8_12[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_12[1], hi = (uint16_t)(_fp4_bf16x8_12[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_12[2], hi = (uint16_t)(_fp4_bf16x8_12[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_12[3], hi = (uint16_t)(_fp4_bf16x8_12[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_14[0] = _fp4_bf16x8_12[0];
                d_14[1] = _fp4_bf16x8_12[1];
                d_14[2] = _fp4_bf16x8_12[2];
                d_14[3] = _fp4_bf16x8_12[3];
                uint32_t _bf16x2_mul_48;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(d_14[0]), "r"(sc[6]));
                w16_13[0] = _bf16x2_mul_48;
                uint32_t _bf16x2_mul_49;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(d_14[1]), "r"(sc[6]));
                w16_13[1] = _bf16x2_mul_49;
                uint32_t _bf16x2_mul_50;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(d_14[2]), "r"(sc[6]));
                w16_13[2] = _bf16x2_mul_50;
                uint32_t _bf16x2_mul_51;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(d_14[3]), "r"(sc[6]));
                w16_13[3] = _bf16x2_mul_51;
                unsigned int d_15[4];
                uint32_t _fp4_bf16x8_13[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(raw_1[13])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(raw_1[13])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_13[0], hi = (uint16_t)(_fp4_bf16x8_13[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_13[1], hi = (uint16_t)(_fp4_bf16x8_13[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_13[2], hi = (uint16_t)(_fp4_bf16x8_13[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_13[3], hi = (uint16_t)(_fp4_bf16x8_13[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_15[0] = _fp4_bf16x8_13[0];
                d_15[1] = _fp4_bf16x8_13[1];
                d_15[2] = _fp4_bf16x8_13[2];
                d_15[3] = _fp4_bf16x8_13[3];
                uint32_t _bf16x2_mul_52;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(d_15[0]), "r"(sc[6]));
                w16_13[4] = _bf16x2_mul_52;
                uint32_t _bf16x2_mul_53;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(d_15[1]), "r"(sc[6]));
                w16_13[5] = _bf16x2_mul_53;
                uint32_t _bf16x2_mul_54;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(d_15[2]), "r"(sc[6]));
                w16_13[6] = _bf16x2_mul_54;
                uint32_t _bf16x2_mul_55;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(d_15[3]), "r"(sc[6]));
                w16_13[7] = _bf16x2_mul_55;
                unsigned int d_16[4];
                uint32_t _fp4_bf16x8_14[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(raw_1[14])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(raw_1[14])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_14[0], hi = (uint16_t)(_fp4_bf16x8_14[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_14[1], hi = (uint16_t)(_fp4_bf16x8_14[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_14[2], hi = (uint16_t)(_fp4_bf16x8_14[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_14[3], hi = (uint16_t)(_fp4_bf16x8_14[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_16[0] = _fp4_bf16x8_14[0];
                d_16[1] = _fp4_bf16x8_14[1];
                d_16[2] = _fp4_bf16x8_14[2];
                d_16[3] = _fp4_bf16x8_14[3];
                uint32_t _bf16x2_mul_56;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(d_16[0]), "r"(sc[7]));
                w16_13[8] = _bf16x2_mul_56;
                uint32_t _bf16x2_mul_57;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(d_16[1]), "r"(sc[7]));
                w16_13[9] = _bf16x2_mul_57;
                uint32_t _bf16x2_mul_58;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(d_16[2]), "r"(sc[7]));
                w16_13[10] = _bf16x2_mul_58;
                uint32_t _bf16x2_mul_59;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(d_16[3]), "r"(sc[7]));
                w16_13[11] = _bf16x2_mul_59;
                unsigned int d_17[4];
                uint32_t _fp4_bf16x8_15[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(raw_1[15])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(raw_1[15])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_15[0], hi = (uint16_t)(_fp4_bf16x8_15[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_15[1], hi = (uint16_t)(_fp4_bf16x8_15[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_15[2], hi = (uint16_t)(_fp4_bf16x8_15[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_15[3], hi = (uint16_t)(_fp4_bf16x8_15[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_17[0] = _fp4_bf16x8_15[0];
                d_17[1] = _fp4_bf16x8_15[1];
                d_17[2] = _fp4_bf16x8_15[2];
                d_17[3] = _fp4_bf16x8_15[3];
                uint32_t _bf16x2_mul_60;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(d_17[0]), "r"(sc[7]));
                w16_13[12] = _bf16x2_mul_60;
                uint32_t _bf16x2_mul_61;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(d_17[1]), "r"(sc[7]));
                w16_13[13] = _bf16x2_mul_61;
                uint32_t _bf16x2_mul_62;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(d_17[2]), "r"(sc[7]));
                w16_13[14] = _bf16x2_mul_62;
                uint32_t _bf16x2_mul_63;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(d_17[3]), "r"(sc[7]));
                w16_13[15] = _bf16x2_mul_63;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(kdst + 48), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_13[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(k_full_addr + ((int)((unsigned int)n_1 % 2)) * 8);
                    mbarrier_arrive(raw_empty_addr + ((int)((unsigned int)n_1 % 4)) * 8);
                }
            }
        }
    }
    // ---- Role: transform_v ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // transform_v_main
            int tid_0_2 = tid;
            int qv = (tid_0_2 >> 5) - 8;
            int lane_3 = tid_0_2 & 31;
            int tv = 16 * qv + (lane_3 & 15);
            int hh_1 = lane_3 >> 4;
            int tm7 = tv & 7;
            unsigned int raw_addr_1 = raw_u8_addr;
            #pragma unroll 1
            for (int n_2 = 0; n_2 < n_tiles; n_2++) {
                int st_1 = n_2 % 4;
                int vs = n_2 % 2;
                int _stage_raw_full_v = (int)((unsigned int)n_2 % 4);
                int _parity_raw_full_v = (int)((unsigned int)n_2 / 4 % 2);
                mbarrier_wait(raw_full_v_addr + (_stage_raw_full_v) * 8, _parity_raw_full_v);
                int _stage_v_empty = (int)((unsigned int)n_2 % 2);
                int _parity_v_empty = (int)((unsigned int)n_2 / 2 % 2) ^ 1;
                mbarrier_wait(v_empty_addr + (_stage_v_empty) * 8, _parity_v_empty);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned int row_addr_1 = raw_addr_1 + (unsigned int)(st_1 * 18432 + 9216 + tv * 144);
                unsigned int raw_3[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(0) + 3]))
                    : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(4) + 3]))
                    : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1 + 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(8) + 3]))
                    : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1 + 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(12) + 3]))
                    : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1 + 3))));
                unsigned int sw_1[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 1]))
                    : "r"(row_addr_1 + (unsigned int)(128 + 8 * hh_1)));
                unsigned int sfa_1[2];
                unsigned int sfb_1[2];
                uint32_t _prmt_2_1;
                asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_2_1) : "r"(sw_1[0]), "r"(sw_1[0]));
                uint32_t _bf16x2_fma_4;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_4) : "r"(_prmt_2_1 & 2147516416u | (_prmt_2_1 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfa_1[0] = _bf16x2_fma_4;
                uint32_t _bf16x2_fma_5;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_5) : "r"(_prmt_2_1 << 8 & 2147516416u | (_prmt_2_1 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfa_1[1] = _bf16x2_fma_5;
                uint32_t _prmt_3_1;
                asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(_prmt_3_1) : "r"(sw_1[1]), "r"(sw_1[1]));
                uint32_t _bf16x2_fma_6;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_6) : "r"(_prmt_3_1 & 2147516416u | (_prmt_3_1 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfb_1[0] = _bf16x2_fma_6;
                uint32_t _bf16x2_fma_7;
                asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_7) : "r"(_prmt_3_1 << 8 & 2147516416u | (_prmt_3_1 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                sfb_1[1] = _bf16x2_fma_7;
                unsigned int sc_1[8];
                uint32_t _prmt_0_2;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_0_2) : "r"(sfa_1[0]), "r"(0));
                sc_1[0] = _prmt_0_2;
                uint32_t _prmt_1_2;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_1_2) : "r"(sfb_1[0]), "r"(0));
                sc_1[4] = _prmt_1_2;
                uint32_t _prmt_2_2;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_2_2) : "r"(sfa_1[0]), "r"(0));
                sc_1[1] = _prmt_2_2;
                uint32_t _prmt_3_2;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_3_2) : "r"(sfb_1[0]), "r"(0));
                sc_1[5] = _prmt_3_2;
                uint32_t _prmt_4_1;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_4_1) : "r"(sfa_1[1]), "r"(0));
                sc_1[2] = _prmt_4_1;
                uint32_t _prmt_5_1;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_5_1) : "r"(sfb_1[1]), "r"(0));
                sc_1[6] = _prmt_5_1;
                uint32_t _prmt_6_1;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_6_1) : "r"(sfa_1[1]), "r"(0));
                sc_1[3] = _prmt_6_1;
                uint32_t _prmt_7_1;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_7_1) : "r"(sfb_1[1]), "r"(0));
                sc_1[7] = _prmt_7_1;
                unsigned int vbase = smem_vt_addr + (unsigned int)((2 * vs + hh_1) * 16384) + (unsigned int)(128 * tv);
                unsigned int plane = vbase;
                unsigned int d_3[4];
                uint32_t _fp4_bf16x8_16[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw_3[0])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw_3[0])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_16[0], hi = (uint16_t)(_fp4_bf16x8_16[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_16[1], hi = (uint16_t)(_fp4_bf16x8_16[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_16[2], hi = (uint16_t)(_fp4_bf16x8_16[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_16[3], hi = (uint16_t)(_fp4_bf16x8_16[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_3[0] = _fp4_bf16x8_16[0];
                d_3[1] = _fp4_bf16x8_16[1];
                d_3[2] = _fp4_bf16x8_16[2];
                d_3[3] = _fp4_bf16x8_16[3];
                unsigned int w4[4];
                uint32_t _bf16x2_mul_64;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(d_3[0]), "r"(sc_1[0]));
                w4[0] = _bf16x2_mul_64;
                uint32_t _bf16x2_mul_65;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(d_3[1]), "r"(sc_1[0]));
                w4[1] = _bf16x2_mul_65;
                uint32_t _bf16x2_mul_66;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(d_3[2]), "r"(sc_1[0]));
                w4[2] = _bf16x2_mul_66;
                uint32_t _bf16x2_mul_67;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(d_3[3]), "r"(sc_1[0]));
                w4[3] = _bf16x2_mul_67;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane + (unsigned int)((0 ^ tm7) << 4)), "r"(w4[0]), "r"(w4[1]), "r"(w4[2]), "r"(w4[3]));
                unsigned int d_0_1[4];
                uint32_t _fp4_bf16x8_17[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(raw_3[1])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(raw_3[1])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_17[0], hi = (uint16_t)(_fp4_bf16x8_17[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_17[1], hi = (uint16_t)(_fp4_bf16x8_17[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_17[2], hi = (uint16_t)(_fp4_bf16x8_17[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_17[3], hi = (uint16_t)(_fp4_bf16x8_17[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_0_1[0] = _fp4_bf16x8_17[0];
                d_0_1[1] = _fp4_bf16x8_17[1];
                d_0_1[2] = _fp4_bf16x8_17[2];
                d_0_1[3] = _fp4_bf16x8_17[3];
                unsigned int w4_1[4];
                uint32_t _bf16x2_mul_68;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(d_0_1[0]), "r"(sc_1[0]));
                w4_1[0] = _bf16x2_mul_68;
                uint32_t _bf16x2_mul_69;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(d_0_1[1]), "r"(sc_1[0]));
                w4_1[1] = _bf16x2_mul_69;
                uint32_t _bf16x2_mul_70;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(d_0_1[2]), "r"(sc_1[0]));
                w4_1[2] = _bf16x2_mul_70;
                uint32_t _bf16x2_mul_71;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(d_0_1[3]), "r"(sc_1[0]));
                w4_1[3] = _bf16x2_mul_71;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane + (unsigned int)((1 ^ tm7) << 4)), "r"(w4_1[0]), "r"(w4_1[1]), "r"(w4_1[2]), "r"(w4_1[3]));
                unsigned int d_2_1[4];
                uint32_t _fp4_bf16x8_18[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(raw_3[2])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(raw_3[2])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_18[0], hi = (uint16_t)(_fp4_bf16x8_18[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_18[1], hi = (uint16_t)(_fp4_bf16x8_18[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_18[2], hi = (uint16_t)(_fp4_bf16x8_18[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_18[3], hi = (uint16_t)(_fp4_bf16x8_18[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_2_1[0] = _fp4_bf16x8_18[0];
                d_2_1[1] = _fp4_bf16x8_18[1];
                d_2_1[2] = _fp4_bf16x8_18[2];
                d_2_1[3] = _fp4_bf16x8_18[3];
                unsigned int w4_3[4];
                uint32_t _bf16x2_mul_72;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(d_2_1[0]), "r"(sc_1[1]));
                w4_3[0] = _bf16x2_mul_72;
                uint32_t _bf16x2_mul_73;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(d_2_1[1]), "r"(sc_1[1]));
                w4_3[1] = _bf16x2_mul_73;
                uint32_t _bf16x2_mul_74;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(d_2_1[2]), "r"(sc_1[1]));
                w4_3[2] = _bf16x2_mul_74;
                uint32_t _bf16x2_mul_75;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(d_2_1[3]), "r"(sc_1[1]));
                w4_3[3] = _bf16x2_mul_75;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane + (unsigned int)((2 ^ tm7) << 4)), "r"(w4_3[0]), "r"(w4_3[1]), "r"(w4_3[2]), "r"(w4_3[3]));
                unsigned int d_4_1[4];
                uint32_t _fp4_bf16x8_19[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(raw_3[3])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(raw_3[3])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_19[0], hi = (uint16_t)(_fp4_bf16x8_19[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_19[1], hi = (uint16_t)(_fp4_bf16x8_19[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_19[2], hi = (uint16_t)(_fp4_bf16x8_19[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_19[3], hi = (uint16_t)(_fp4_bf16x8_19[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_4_1[0] = _fp4_bf16x8_19[0];
                d_4_1[1] = _fp4_bf16x8_19[1];
                d_4_1[2] = _fp4_bf16x8_19[2];
                d_4_1[3] = _fp4_bf16x8_19[3];
                unsigned int w4_5[4];
                uint32_t _bf16x2_mul_76;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(d_4_1[0]), "r"(sc_1[1]));
                w4_5[0] = _bf16x2_mul_76;
                uint32_t _bf16x2_mul_77;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(d_4_1[1]), "r"(sc_1[1]));
                w4_5[1] = _bf16x2_mul_77;
                uint32_t _bf16x2_mul_78;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(d_4_1[2]), "r"(sc_1[1]));
                w4_5[2] = _bf16x2_mul_78;
                uint32_t _bf16x2_mul_79;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(d_4_1[3]), "r"(sc_1[1]));
                w4_5[3] = _bf16x2_mul_79;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane + (unsigned int)((3 ^ tm7) << 4)), "r"(w4_5[0]), "r"(w4_5[1]), "r"(w4_5[2]), "r"(w4_5[3]));
                unsigned int plane_6 = vbase;
                unsigned int d_7_1[4];
                uint32_t _fp4_bf16x8_20[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(raw_3[4])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(raw_3[4])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_20[0], hi = (uint16_t)(_fp4_bf16x8_20[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_20[1], hi = (uint16_t)(_fp4_bf16x8_20[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_20[2], hi = (uint16_t)(_fp4_bf16x8_20[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_20[3], hi = (uint16_t)(_fp4_bf16x8_20[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_7_1[0] = _fp4_bf16x8_20[0];
                d_7_1[1] = _fp4_bf16x8_20[1];
                d_7_1[2] = _fp4_bf16x8_20[2];
                d_7_1[3] = _fp4_bf16x8_20[3];
                unsigned int w4_8[4];
                uint32_t _bf16x2_mul_80;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(d_7_1[0]), "r"(sc_1[2]));
                w4_8[0] = _bf16x2_mul_80;
                uint32_t _bf16x2_mul_81;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(d_7_1[1]), "r"(sc_1[2]));
                w4_8[1] = _bf16x2_mul_81;
                uint32_t _bf16x2_mul_82;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(d_7_1[2]), "r"(sc_1[2]));
                w4_8[2] = _bf16x2_mul_82;
                uint32_t _bf16x2_mul_83;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(d_7_1[3]), "r"(sc_1[2]));
                w4_8[3] = _bf16x2_mul_83;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_6 + (unsigned int)((4 ^ tm7) << 4)), "r"(w4_8[0]), "r"(w4_8[1]), "r"(w4_8[2]), "r"(w4_8[3]));
                unsigned int d_9_1[4];
                uint32_t _fp4_bf16x8_21[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(raw_3[5])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(raw_3[5])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_21[0], hi = (uint16_t)(_fp4_bf16x8_21[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_21[1], hi = (uint16_t)(_fp4_bf16x8_21[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_21[2], hi = (uint16_t)(_fp4_bf16x8_21[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_21[3], hi = (uint16_t)(_fp4_bf16x8_21[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_9_1[0] = _fp4_bf16x8_21[0];
                d_9_1[1] = _fp4_bf16x8_21[1];
                d_9_1[2] = _fp4_bf16x8_21[2];
                d_9_1[3] = _fp4_bf16x8_21[3];
                unsigned int w4_10[4];
                uint32_t _bf16x2_mul_84;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(d_9_1[0]), "r"(sc_1[2]));
                w4_10[0] = _bf16x2_mul_84;
                uint32_t _bf16x2_mul_85;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(d_9_1[1]), "r"(sc_1[2]));
                w4_10[1] = _bf16x2_mul_85;
                uint32_t _bf16x2_mul_86;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_86) : "r"(d_9_1[2]), "r"(sc_1[2]));
                w4_10[2] = _bf16x2_mul_86;
                uint32_t _bf16x2_mul_87;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_87) : "r"(d_9_1[3]), "r"(sc_1[2]));
                w4_10[3] = _bf16x2_mul_87;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_6 + (unsigned int)((5 ^ tm7) << 4)), "r"(w4_10[0]), "r"(w4_10[1]), "r"(w4_10[2]), "r"(w4_10[3]));
                unsigned int d_11_1[4];
                uint32_t _fp4_bf16x8_22[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(raw_3[6])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(raw_3[6])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_22[0], hi = (uint16_t)(_fp4_bf16x8_22[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_22[1], hi = (uint16_t)(_fp4_bf16x8_22[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_22[2], hi = (uint16_t)(_fp4_bf16x8_22[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_22[3], hi = (uint16_t)(_fp4_bf16x8_22[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_11_1[0] = _fp4_bf16x8_22[0];
                d_11_1[1] = _fp4_bf16x8_22[1];
                d_11_1[2] = _fp4_bf16x8_22[2];
                d_11_1[3] = _fp4_bf16x8_22[3];
                unsigned int w4_12[4];
                uint32_t _bf16x2_mul_88;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_88) : "r"(d_11_1[0]), "r"(sc_1[3]));
                w4_12[0] = _bf16x2_mul_88;
                uint32_t _bf16x2_mul_89;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_89) : "r"(d_11_1[1]), "r"(sc_1[3]));
                w4_12[1] = _bf16x2_mul_89;
                uint32_t _bf16x2_mul_90;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_90) : "r"(d_11_1[2]), "r"(sc_1[3]));
                w4_12[2] = _bf16x2_mul_90;
                uint32_t _bf16x2_mul_91;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_91) : "r"(d_11_1[3]), "r"(sc_1[3]));
                w4_12[3] = _bf16x2_mul_91;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_6 + (unsigned int)((6 ^ tm7) << 4)), "r"(w4_12[0]), "r"(w4_12[1]), "r"(w4_12[2]), "r"(w4_12[3]));
                unsigned int d_13[4];
                uint32_t _fp4_bf16x8_23[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(raw_3[7])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(raw_3[7])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_23[0], hi = (uint16_t)(_fp4_bf16x8_23[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_23[1], hi = (uint16_t)(_fp4_bf16x8_23[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_23[2], hi = (uint16_t)(_fp4_bf16x8_23[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_23[3], hi = (uint16_t)(_fp4_bf16x8_23[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_13[0] = _fp4_bf16x8_23[0];
                d_13[1] = _fp4_bf16x8_23[1];
                d_13[2] = _fp4_bf16x8_23[2];
                d_13[3] = _fp4_bf16x8_23[3];
                unsigned int w4_14[4];
                uint32_t _bf16x2_mul_92;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_92) : "r"(d_13[0]), "r"(sc_1[3]));
                w4_14[0] = _bf16x2_mul_92;
                uint32_t _bf16x2_mul_93;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_93) : "r"(d_13[1]), "r"(sc_1[3]));
                w4_14[1] = _bf16x2_mul_93;
                uint32_t _bf16x2_mul_94;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_94) : "r"(d_13[2]), "r"(sc_1[3]));
                w4_14[2] = _bf16x2_mul_94;
                uint32_t _bf16x2_mul_95;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_95) : "r"(d_13[3]), "r"(sc_1[3]));
                w4_14[3] = _bf16x2_mul_95;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_6 + (unsigned int)((7 ^ tm7) << 4)), "r"(w4_14[0]), "r"(w4_14[1]), "r"(w4_14[2]), "r"(w4_14[3]));
                unsigned int plane_15 = vbase + 8192;
                unsigned int d_16_1[4];
                uint32_t _fp4_bf16x8_24[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(raw_3[8])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(raw_3[8])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_24[0], hi = (uint16_t)(_fp4_bf16x8_24[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_24[1], hi = (uint16_t)(_fp4_bf16x8_24[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_24[2], hi = (uint16_t)(_fp4_bf16x8_24[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_24[3], hi = (uint16_t)(_fp4_bf16x8_24[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_16_1[0] = _fp4_bf16x8_24[0];
                d_16_1[1] = _fp4_bf16x8_24[1];
                d_16_1[2] = _fp4_bf16x8_24[2];
                d_16_1[3] = _fp4_bf16x8_24[3];
                unsigned int w4_17[4];
                uint32_t _bf16x2_mul_96;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_96) : "r"(d_16_1[0]), "r"(sc_1[4]));
                w4_17[0] = _bf16x2_mul_96;
                uint32_t _bf16x2_mul_97;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_97) : "r"(d_16_1[1]), "r"(sc_1[4]));
                w4_17[1] = _bf16x2_mul_97;
                uint32_t _bf16x2_mul_98;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_98) : "r"(d_16_1[2]), "r"(sc_1[4]));
                w4_17[2] = _bf16x2_mul_98;
                uint32_t _bf16x2_mul_99;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_99) : "r"(d_16_1[3]), "r"(sc_1[4]));
                w4_17[3] = _bf16x2_mul_99;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_15 + (unsigned int)((0 ^ tm7) << 4)), "r"(w4_17[0]), "r"(w4_17[1]), "r"(w4_17[2]), "r"(w4_17[3]));
                unsigned int d_18[4];
                uint32_t _fp4_bf16x8_25[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(raw_3[9])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(raw_3[9])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_25[0], hi = (uint16_t)(_fp4_bf16x8_25[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_25[1], hi = (uint16_t)(_fp4_bf16x8_25[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_25[2], hi = (uint16_t)(_fp4_bf16x8_25[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_25[3], hi = (uint16_t)(_fp4_bf16x8_25[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_18[0] = _fp4_bf16x8_25[0];
                d_18[1] = _fp4_bf16x8_25[1];
                d_18[2] = _fp4_bf16x8_25[2];
                d_18[3] = _fp4_bf16x8_25[3];
                unsigned int w4_19[4];
                uint32_t _bf16x2_mul_100;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_100) : "r"(d_18[0]), "r"(sc_1[4]));
                w4_19[0] = _bf16x2_mul_100;
                uint32_t _bf16x2_mul_101;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_101) : "r"(d_18[1]), "r"(sc_1[4]));
                w4_19[1] = _bf16x2_mul_101;
                uint32_t _bf16x2_mul_102;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_102) : "r"(d_18[2]), "r"(sc_1[4]));
                w4_19[2] = _bf16x2_mul_102;
                uint32_t _bf16x2_mul_103;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_103) : "r"(d_18[3]), "r"(sc_1[4]));
                w4_19[3] = _bf16x2_mul_103;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_15 + (unsigned int)((1 ^ tm7) << 4)), "r"(w4_19[0]), "r"(w4_19[1]), "r"(w4_19[2]), "r"(w4_19[3]));
                unsigned int d_20[4];
                uint32_t _fp4_bf16x8_26[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(raw_3[10])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(raw_3[10])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_26[0], hi = (uint16_t)(_fp4_bf16x8_26[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_26[1], hi = (uint16_t)(_fp4_bf16x8_26[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_26[2], hi = (uint16_t)(_fp4_bf16x8_26[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_26[3], hi = (uint16_t)(_fp4_bf16x8_26[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_20[0] = _fp4_bf16x8_26[0];
                d_20[1] = _fp4_bf16x8_26[1];
                d_20[2] = _fp4_bf16x8_26[2];
                d_20[3] = _fp4_bf16x8_26[3];
                unsigned int w4_21[4];
                uint32_t _bf16x2_mul_104;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_104) : "r"(d_20[0]), "r"(sc_1[5]));
                w4_21[0] = _bf16x2_mul_104;
                uint32_t _bf16x2_mul_105;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_105) : "r"(d_20[1]), "r"(sc_1[5]));
                w4_21[1] = _bf16x2_mul_105;
                uint32_t _bf16x2_mul_106;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_106) : "r"(d_20[2]), "r"(sc_1[5]));
                w4_21[2] = _bf16x2_mul_106;
                uint32_t _bf16x2_mul_107;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_107) : "r"(d_20[3]), "r"(sc_1[5]));
                w4_21[3] = _bf16x2_mul_107;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_15 + (unsigned int)((2 ^ tm7) << 4)), "r"(w4_21[0]), "r"(w4_21[1]), "r"(w4_21[2]), "r"(w4_21[3]));
                unsigned int d_22[4];
                uint32_t _fp4_bf16x8_27[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(raw_3[11])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(raw_3[11])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_27[0], hi = (uint16_t)(_fp4_bf16x8_27[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_27[1], hi = (uint16_t)(_fp4_bf16x8_27[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_27[2], hi = (uint16_t)(_fp4_bf16x8_27[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_27[3], hi = (uint16_t)(_fp4_bf16x8_27[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_22[0] = _fp4_bf16x8_27[0];
                d_22[1] = _fp4_bf16x8_27[1];
                d_22[2] = _fp4_bf16x8_27[2];
                d_22[3] = _fp4_bf16x8_27[3];
                unsigned int w4_23[4];
                uint32_t _bf16x2_mul_108;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_108) : "r"(d_22[0]), "r"(sc_1[5]));
                w4_23[0] = _bf16x2_mul_108;
                uint32_t _bf16x2_mul_109;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_109) : "r"(d_22[1]), "r"(sc_1[5]));
                w4_23[1] = _bf16x2_mul_109;
                uint32_t _bf16x2_mul_110;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_110) : "r"(d_22[2]), "r"(sc_1[5]));
                w4_23[2] = _bf16x2_mul_110;
                uint32_t _bf16x2_mul_111;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_111) : "r"(d_22[3]), "r"(sc_1[5]));
                w4_23[3] = _bf16x2_mul_111;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_15 + (unsigned int)((3 ^ tm7) << 4)), "r"(w4_23[0]), "r"(w4_23[1]), "r"(w4_23[2]), "r"(w4_23[3]));
                unsigned int plane_24 = vbase + 8192;
                unsigned int d_25[4];
                uint32_t _fp4_bf16x8_28[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(raw_3[12])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(raw_3[12])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_28[0], hi = (uint16_t)(_fp4_bf16x8_28[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_28[1], hi = (uint16_t)(_fp4_bf16x8_28[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_28[2], hi = (uint16_t)(_fp4_bf16x8_28[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_28[3], hi = (uint16_t)(_fp4_bf16x8_28[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_25[0] = _fp4_bf16x8_28[0];
                d_25[1] = _fp4_bf16x8_28[1];
                d_25[2] = _fp4_bf16x8_28[2];
                d_25[3] = _fp4_bf16x8_28[3];
                unsigned int w4_26[4];
                uint32_t _bf16x2_mul_112;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_112) : "r"(d_25[0]), "r"(sc_1[6]));
                w4_26[0] = _bf16x2_mul_112;
                uint32_t _bf16x2_mul_113;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_113) : "r"(d_25[1]), "r"(sc_1[6]));
                w4_26[1] = _bf16x2_mul_113;
                uint32_t _bf16x2_mul_114;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_114) : "r"(d_25[2]), "r"(sc_1[6]));
                w4_26[2] = _bf16x2_mul_114;
                uint32_t _bf16x2_mul_115;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_115) : "r"(d_25[3]), "r"(sc_1[6]));
                w4_26[3] = _bf16x2_mul_115;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_24 + (unsigned int)((4 ^ tm7) << 4)), "r"(w4_26[0]), "r"(w4_26[1]), "r"(w4_26[2]), "r"(w4_26[3]));
                unsigned int d_27[4];
                uint32_t _fp4_bf16x8_29[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(raw_3[13])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(raw_3[13])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_29[0], hi = (uint16_t)(_fp4_bf16x8_29[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_29[1], hi = (uint16_t)(_fp4_bf16x8_29[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_29[2], hi = (uint16_t)(_fp4_bf16x8_29[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_29[3], hi = (uint16_t)(_fp4_bf16x8_29[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_27[0] = _fp4_bf16x8_29[0];
                d_27[1] = _fp4_bf16x8_29[1];
                d_27[2] = _fp4_bf16x8_29[2];
                d_27[3] = _fp4_bf16x8_29[3];
                unsigned int w4_28[4];
                uint32_t _bf16x2_mul_116;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_116) : "r"(d_27[0]), "r"(sc_1[6]));
                w4_28[0] = _bf16x2_mul_116;
                uint32_t _bf16x2_mul_117;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_117) : "r"(d_27[1]), "r"(sc_1[6]));
                w4_28[1] = _bf16x2_mul_117;
                uint32_t _bf16x2_mul_118;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_118) : "r"(d_27[2]), "r"(sc_1[6]));
                w4_28[2] = _bf16x2_mul_118;
                uint32_t _bf16x2_mul_119;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_119) : "r"(d_27[3]), "r"(sc_1[6]));
                w4_28[3] = _bf16x2_mul_119;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_24 + (unsigned int)((5 ^ tm7) << 4)), "r"(w4_28[0]), "r"(w4_28[1]), "r"(w4_28[2]), "r"(w4_28[3]));
                unsigned int d_29[4];
                uint32_t _fp4_bf16x8_30[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(raw_3[14])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(raw_3[14])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_30[0], hi = (uint16_t)(_fp4_bf16x8_30[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_30[1], hi = (uint16_t)(_fp4_bf16x8_30[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_30[2], hi = (uint16_t)(_fp4_bf16x8_30[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_30[3], hi = (uint16_t)(_fp4_bf16x8_30[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_29[0] = _fp4_bf16x8_30[0];
                d_29[1] = _fp4_bf16x8_30[1];
                d_29[2] = _fp4_bf16x8_30[2];
                d_29[3] = _fp4_bf16x8_30[3];
                unsigned int w4_30[4];
                uint32_t _bf16x2_mul_120;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_120) : "r"(d_29[0]), "r"(sc_1[7]));
                w4_30[0] = _bf16x2_mul_120;
                uint32_t _bf16x2_mul_121;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_121) : "r"(d_29[1]), "r"(sc_1[7]));
                w4_30[1] = _bf16x2_mul_121;
                uint32_t _bf16x2_mul_122;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_122) : "r"(d_29[2]), "r"(sc_1[7]));
                w4_30[2] = _bf16x2_mul_122;
                uint32_t _bf16x2_mul_123;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_123) : "r"(d_29[3]), "r"(sc_1[7]));
                w4_30[3] = _bf16x2_mul_123;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_24 + (unsigned int)((6 ^ tm7) << 4)), "r"(w4_30[0]), "r"(w4_30[1]), "r"(w4_30[2]), "r"(w4_30[3]));
                unsigned int d_31[4];
                uint32_t _fp4_bf16x8_31[4];
                {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                    "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                    "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                    "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(raw_3[15])));
                #else
                asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                    "cvt.rn.f16x2.e2m1x2 %0, b0;"
                    "cvt.rn.f16x2.e2m1x2 %1, b1;"
                    "cvt.rn.f16x2.e2m1x2 %2, b2;"
                    "cvt.rn.f16x2.e2m1x2 %3, b3;"
                    "}"
                    : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(raw_3[15])));
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_31[0], hi = (uint16_t)(_fp4_bf16x8_31[0] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[0]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_31[1], hi = (uint16_t)(_fp4_bf16x8_31[1] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[1]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_31[2], hi = (uint16_t)(_fp4_bf16x8_31[2] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[2]) : "f"(f1), "f"(f0));
                }
                {
                uint16_t lo = (uint16_t)_fp4_bf16x8_31[3], hi = (uint16_t)(_fp4_bf16x8_31[3] >> 16);
                float f0, f1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[3]) : "f"(f1), "f"(f0));
                }
                #endif
                }
                d_31[0] = _fp4_bf16x8_31[0];
                d_31[1] = _fp4_bf16x8_31[1];
                d_31[2] = _fp4_bf16x8_31[2];
                d_31[3] = _fp4_bf16x8_31[3];
                unsigned int w4_32[4];
                uint32_t _bf16x2_mul_124;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_124) : "r"(d_31[0]), "r"(sc_1[7]));
                w4_32[0] = _bf16x2_mul_124;
                uint32_t _bf16x2_mul_125;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_125) : "r"(d_31[1]), "r"(sc_1[7]));
                w4_32[1] = _bf16x2_mul_125;
                uint32_t _bf16x2_mul_126;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_126) : "r"(d_31[2]), "r"(sc_1[7]));
                w4_32[2] = _bf16x2_mul_126;
                uint32_t _bf16x2_mul_127;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_127) : "r"(d_31[3]), "r"(sc_1[7]));
                w4_32[3] = _bf16x2_mul_127;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_24 + (unsigned int)((7 ^ tm7) << 4)), "r"(w4_32[0]), "r"(w4_32[1]), "r"(w4_32[2]), "r"(w4_32[3]));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(v_full_addr + ((int)((unsigned int)n_2 % 2)) * 8);
                    mbarrier_arrive(raw_empty_addr + ((int)((unsigned int)n_2 % 4)) * 8);
                }
            }
        }
    }
    // ---- Role: gather ----
    if (warp >= 12 && warp <= 13) {
        { // gather_main
            int gtid = tid - 384;
            int g_1 = gtid >> 5;
            int lane_4 = gtid & 31;
            int token = gtid;
            int phys[3];
            int local[3];
            int column_1 = split * 64 + token;
            bool in_range = column_1 < topk && tile_end > split;
            int logical = ((in_range) ? idx[0] : -1);
            idx[0] = logical;
            int column_0_1 = (split + num_splits) * 64 + token;
            bool in_range_1 = column_0_1 < topk && tile_end > split + num_splits;
            int logical_2 = ((in_range_1) ? idx[1] : -1);
            idx[1] = logical_2;
            int column_3_1 = (split + 2 * num_splits) * 64 + token;
            bool in_range_4 = column_3_1 < topk && tile_end > split + 2 * num_splits;
            int logical_5 = ((in_range_4) ? idx[2] : -1);
            idx[2] = logical_5;
            int column_6_1 = (split + 3 * num_splits) * 64 + token;
            bool in_range_7 = column_6_1 < topk && tile_end > split + 3 * num_splits;
            int logical_8 = ((in_range_7) ? idx[3] : -1);
            idx[3] = logical_8;
            int safe_token = ((idx[0]) > (0) ? (idx[0]) : (0));
            uint32_t _fast_div_q_0 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token);
            uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)safe_token) - _fast_div_q_0 * (uint32_t)(page_div.divisor);
            int page = (int)_fast_div_q_0;
            bool valid_5 = request_ok && idx[0] >= 0 && page < table_width;
            int table_column = ((page) < (table_width - 1) ? (page) : (table_width - 1));
            int entry = ((valid_5) ? block_table[safe_request_row + (long long)table_column] : -1);
            phys[0] = entry;
            local[0] = (int)_fast_div_r_0;
            int safe_token_9 = ((idx[1]) > (0) ? (idx[1]) : (0));
            uint32_t _fast_div_q_1 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_9), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_9);
            uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)safe_token_9) - _fast_div_q_1 * (uint32_t)(page_div.divisor);
            int page_10 = (int)_fast_div_q_1;
            bool valid_11 = request_ok && idx[1] >= 0 && page_10 < table_width;
            int table_column_12 = ((page_10) < (table_width - 1) ? (page_10) : (table_width - 1));
            int entry_13 = ((valid_11) ? block_table[safe_request_row + (long long)table_column_12] : -1);
            phys[1] = entry_13;
            local[1] = (int)_fast_div_r_1;
            int e_1 = gtid;
            int head_7 = e_1 >> 5;
            int c_1 = e_1 & 31;
            int panel = c_1 >> 3;
            int cc = c_1 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel * 2048) + (unsigned int)(head_7 * 128 + 16 * cc ^ (head_7 * 128 + 16 * cc >> 7 & 7) << 4))), "r"(qw_regs[0]), "r"(qw_regs[1]), "r"(qw_regs[2]), "r"(qw_regs[3]) : "memory");
            int e_14 = gtid + 64;
            int head_15 = e_14 >> 5;
            int c_16 = e_14 & 31;
            int panel_17 = c_16 >> 3;
            int cc_18 = c_16 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_17 * 2048) + (unsigned int)(head_15 * 128 + 16 * cc_18 ^ (head_15 * 128 + 16 * cc_18 >> 7 & 7) << 4))), "r"(qw_regs[4]), "r"(qw_regs[5]), "r"(qw_regs[6]), "r"(qw_regs[7]) : "memory");
            int e_19 = gtid + 128;
            int head_20 = e_19 >> 5;
            int c_21 = e_19 & 31;
            int panel_22 = c_21 >> 3;
            int cc_23 = c_21 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_22 * 2048) + (unsigned int)(head_20 * 128 + 16 * cc_23 ^ (head_20 * 128 + 16 * cc_23 >> 7 & 7) << 4))), "r"(qw_regs[8]), "r"(qw_regs[9]), "r"(qw_regs[10]), "r"(qw_regs[11]) : "memory");
            int e_24_1 = gtid + 192;
            int head_25_1 = e_24_1 >> 5;
            int c_26_1 = e_24_1 & 31;
            int panel_27 = c_26_1 >> 3;
            int cc_28 = c_26_1 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_27 * 2048) + (unsigned int)(head_25_1 * 128 + 16 * cc_28 ^ (head_25_1 * 128 + 16 * cc_28 >> 7 & 7) << 4))), "r"(qw_regs[12]), "r"(qw_regs[13]), "r"(qw_regs[14]), "r"(qw_regs[15]) : "memory");
            int e_29 = gtid + 256;
            int head_30 = e_29 >> 5;
            int c_31 = e_29 & 31;
            int panel_32 = c_31 >> 3;
            int cc_33 = c_31 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_32 * 2048) + (unsigned int)(head_30 * 128 + 16 * cc_33 ^ (head_30 * 128 + 16 * cc_33 >> 7 & 7) << 4))), "r"(qw_regs[16]), "r"(qw_regs[17]), "r"(qw_regs[18]), "r"(qw_regs[19]) : "memory");
            int e_34 = gtid + 320;
            int head_35 = e_34 >> 5;
            int c_36 = e_34 & 31;
            int panel_37 = c_36 >> 3;
            int cc_38 = c_36 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_37 * 2048) + (unsigned int)(head_35 * 128 + 16 * cc_38 ^ (head_35 * 128 + 16 * cc_38 >> 7 & 7) << 4))), "r"(qw_regs[20]), "r"(qw_regs[21]), "r"(qw_regs[22]), "r"(qw_regs[23]) : "memory");
            int e_39 = gtid + 384;
            int head_40 = e_39 >> 5;
            int c_41 = e_39 & 31;
            int panel_42 = c_41 >> 3;
            int cc_43 = c_41 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_42 * 2048) + (unsigned int)(head_40 * 128 + 16 * cc_43 ^ (head_40 * 128 + 16 * cc_43 >> 7 & 7) << 4))), "r"(qw_regs[24]), "r"(qw_regs[25]), "r"(qw_regs[26]), "r"(qw_regs[27]) : "memory");
            int e_44 = gtid + 448;
            int head_45 = e_44 >> 5;
            int c_46 = e_44 & 31;
            int panel_47 = c_46 >> 3;
            int cc_48 = c_46 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_47 * 2048) + (unsigned int)(head_45 * 128 + 16 * cc_48 ^ (head_45 * 128 + 16 * cc_48 >> 7 & 7) << 4))), "r"(qw_regs[28]), "r"(qw_regs[29]), "r"(qw_regs[30]), "r"(qw_regs[31]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(q_full_addr);
            }
            unsigned long long kv_head_u64 = (unsigned long long)kv_head;
            unsigned long long data_stride_block_u64 = (unsigned long long)data_stride_block;
            unsigned long long data_stride_token_u64 = (unsigned long long)data_stride_token;
            unsigned long long sf_stride_block_u64 = (unsigned long long)sf_stride_block;
            unsigned long long sf_stride_token_u64 = (unsigned long long)sf_stride_token;
            unsigned long long head_data = kv_head_u64 * (unsigned long long)data_stride_head;
            unsigned long long head_sf = kv_head_u64 * (unsigned long long)sf_stride_head;
            unsigned long long k_base = (unsigned long long)k_data + head_data;
            unsigned long long v_base = (unsigned long long)v_data + head_data;
            unsigned long long k_sf_base = (unsigned long long)k_sf + head_sf;
            unsigned long long v_sf_base = (unsigned long long)v_sf + head_sf;
            unsigned int kv_addr = raw_u8_addr;
            unsigned long long v_src_a[4];
            unsigned int v_bits_a[1];
            unsigned long long v_src_b[4];
            unsigned int v_bits_b[1];
            #pragma unroll 1
            for (int n_3 = 0; n_3 < n_tiles; n_3++) {
                int st_2 = n_3 % 4;
                int slot = n_3 & 1;
                int column_1_1 = (split + (n_3 + 4) * num_splits) * 64 + token;
                bool in_range_2 = column_1_1 < topk && tile_end > split + (n_3 + 4) * num_splits;
                int safe_column_2 = ((column_1_1) < (topk - 1) ? (column_1_1) : (topk - 1));
                int logical_3 = ((in_range_2) ? indices[indices_row + (long long)safe_column_2] : -1);
                int safe_token_4 = ((idx[2]) > (0) ? (idx[2]) : (0));
                uint32_t _fast_div_q_2 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_4), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_4);
                uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)safe_token_4) - _fast_div_q_2 * (uint32_t)(page_div.divisor);
                int page_5 = (int)_fast_div_q_2;
                bool valid_6 = request_ok && idx[2] >= 0 && page_5 < table_width;
                int table_column_7 = ((page_5) < (table_width - 1) ? (page_5) : (table_width - 1));
                int entry_8 = ((valid_6) ? block_table[safe_request_row + (long long)table_column_7] : -1);
                phys[2] = entry_8;
                local[2] = (int)_fast_div_r_2;
                bool valid_9 = phys[0] >= 0 && phys[0] < num_blocks;
                int _max_0_1 = ((phys[0]) > (0) ? (phys[0]) : (0));
                unsigned long long block = (unsigned long long)_max_0_1;
                unsigned long long local_u64 = (unsigned long long)local[0];
                unsigned long long data_off = block * data_stride_block_u64 + local_u64 * data_stride_token_u64;
                unsigned long long sf_off = block * sf_stride_block_u64 + local_u64 * sf_stride_token_u64;
                unsigned int w0 = (unsigned int)data_off;
                unsigned int w1 = (unsigned int)(data_off >> 32);
                unsigned int w2 = (unsigned int)sf_off;
                unsigned int w3 = (unsigned int)(sf_off >> 32);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(slot * 1024 + token * 16)), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
                unsigned int bits_2 = __ballot_sync(0xFFFFFFFF, valid_9);
                if (lane_4 == 0) {
                    slot_valid[slot * 2 + g_1] = bits_2;
                }
                asm volatile("barrier.sync 1, 64;" ::: "memory");
                int _stage_raw_empty = (int)((unsigned int)n_3 % 4);
                int _parity_raw_empty = (int)((unsigned int)n_3 / 4 % 2) ^ 1;
                mbarrier_wait(raw_empty_addr + (_stage_raw_empty) * 8, _parity_raw_empty);
                int _stage_valid_empty = (int)((unsigned int)n_3 % 4);
                int _parity_valid_empty = (int)((unsigned int)n_3 / 4 % 2) ^ 1;
                mbarrier_wait(valid_empty_addr + (_stage_valid_empty) * 8, _parity_valid_empty);
                if (gtid < 2) {
                    valid_s[st_2 * 2 + gtid] = slot_valid[slot * 2 + gtid];
                    mbarrier_arrive(valid_full_addr + ((int)((unsigned int)n_3 % 4)) * 8);
                }
                unsigned int bits_10 = slot_valid[slot * 2 + (g_1 >> 1)] >> (unsigned int)(16 * (g_1 & 1));
                int col = lane_4 & 7;
                int sub = lane_4 >> 3;
                int token0 = 16 * g_1 + sub;
                unsigned int entry_addr = addr_ring_addr + (unsigned int)(slot * 1024 + token0 * 16);
                unsigned int bits_sub = bits_10 >> (unsigned int)sub;
                unsigned long long col_off = (unsigned long long)(col * 16);
                unsigned int stage_base = kv_addr + (unsigned int)(st_2 * 18432);
                unsigned int dst_k0 = ((stage_base + (unsigned int)(token0 * 144 + col * 16)) < (kv_addr + 73712) ? (stage_base + (unsigned int)(token0 * 144 + col * 16)) : (kv_addr + 73712));
                int sf_lane = lane_4 & 15;
                bool sf_is_v = lane_4 >= 16;
                int sf_token = 16 * g_1 + sf_lane;
                unsigned int entries_0[4];
                unsigned int entries_1[4];
                unsigned int entries_2[4];
                unsigned int entries_3[4];
                unsigned int sf_entry[4];
                __int128_t _smem_raw_1;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_1) : "r"(entry_addr));
                entries_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[0];
                entries_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[1];
                entries_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[2];
                entries_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[3];
                __int128_t _smem_raw_2;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(entry_addr + 64));
                entries_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                entries_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                entries_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                entries_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(entry_addr + 128));
                entries_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                entries_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                entries_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                entries_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(entry_addr + 192));
                entries_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                entries_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                entries_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                entries_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                __int128_t _smem_raw_5;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(addr_ring_addr + (unsigned int)(slot * 1024 + sf_token * 16)));
                sf_entry[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                sf_entry[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                sf_entry[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                sf_entry[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                unsigned long long k_src_base = k_base + col_off;
                unsigned long long v_src_base = v_base + col_off;
                v_bits_a[0] = bits_sub;
                unsigned long long data_off_11 = (unsigned long long)entries_0[0] | (unsigned long long)entries_0[1] << 32;
                unsigned int src_size = (bits_sub & 1) << 4;
                v_src_a[0] = v_src_base + data_off_11;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0), "l"(k_src_base + data_off_11), "r"(src_size), "l"(0x12F0000000000000ULL));
                unsigned long long data_off_12 = (unsigned long long)entries_1[0] | (unsigned long long)entries_1[1] << 32;
                unsigned int src_size_13 = (bits_sub >> 4 & 1) << 4;
                v_src_a[1] = v_src_base + data_off_12;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0 + 576), "l"(k_src_base + data_off_12), "r"(src_size_13), "l"(0x12F0000000000000ULL));
                unsigned long long data_off_14 = (unsigned long long)entries_2[0] | (unsigned long long)entries_2[1] << 32;
                unsigned int src_size_15 = (bits_sub >> 8 & 1) << 4;
                v_src_a[2] = v_src_base + data_off_14;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0 + 1152), "l"(k_src_base + data_off_14), "r"(src_size_15), "l"(0x12F0000000000000ULL));
                unsigned long long data_off_16 = (unsigned long long)entries_3[0] | (unsigned long long)entries_3[1] << 32;
                unsigned int src_size_17 = (bits_sub >> 12 & 1) << 4;
                v_src_a[3] = v_src_base + data_off_16;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0 + 1728), "l"(k_src_base + data_off_16), "r"(src_size_17), "l"(0x12F0000000000000ULL));
                unsigned long long sf_off_18 = (unsigned long long)sf_entry[2] | (unsigned long long)sf_entry[3] << 32;
                unsigned long long sf_base = ((sf_is_v) ? v_sf_base : k_sf_base);
                int sf_side = ((sf_is_v) ? 9216 : 0);
                unsigned int sf_size = (bits_10 >> (unsigned int)sf_lane & 1) << 4;
                unsigned int sf_dst = ((stage_base + (unsigned int)(sf_side + sf_token * 144 + 128)) < (kv_addr + 73712) ? (stage_base + (unsigned int)(sf_side + sf_token * 144 + 128)) : (kv_addr + 73712));
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(sf_dst), "l"(sf_base + sf_off_18), "r"(sf_size), "l"(0x12F0000000000000ULL));
                unsigned int bits_19 = slot_valid[slot * 2 + (g_1 + 2 >> 1)] >> (unsigned int)(16 * (g_1 + 2 & 1));
                int col_20 = lane_4 & 7;
                int sub_21 = lane_4 >> 3;
                int token0_22 = 16 * (g_1 + 2) + sub_21;
                unsigned int entry_addr_23 = addr_ring_addr + (unsigned int)(slot * 1024 + token0_22 * 16);
                unsigned int bits_sub_24 = bits_19 >> (unsigned int)sub_21;
                unsigned long long col_off_25 = (unsigned long long)(col_20 * 16);
                unsigned int stage_base_26 = kv_addr + (unsigned int)(st_2 * 18432);
                unsigned int dst_k0_27 = ((stage_base_26 + (unsigned int)(token0_22 * 144 + col_20 * 16)) < (kv_addr + 73712) ? (stage_base_26 + (unsigned int)(token0_22 * 144 + col_20 * 16)) : (kv_addr + 73712));
                int sf_lane_28 = lane_4 & 15;
                bool sf_is_v_29 = lane_4 >= 16;
                int sf_token_30 = 16 * (g_1 + 2) + sf_lane_28;
                unsigned int entries_0_31[4];
                unsigned int entries_1_32[4];
                unsigned int entries_2_33[4];
                unsigned int entries_3_34[4];
                unsigned int sf_entry_35[4];
                __int128_t _smem_raw_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(entry_addr_23));
                entries_0_31[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                entries_0_31[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                entries_0_31[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                entries_0_31[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                __int128_t _smem_raw_7;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(entry_addr_23 + 64));
                entries_1_32[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                entries_1_32[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                entries_1_32[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                entries_1_32[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                __int128_t _smem_raw_8;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(entry_addr_23 + 128));
                entries_2_33[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                entries_2_33[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                entries_2_33[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                entries_2_33[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                __int128_t _smem_raw_9;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(entry_addr_23 + 192));
                entries_3_34[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                entries_3_34[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                entries_3_34[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                entries_3_34[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
                __int128_t _smem_raw_10;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(addr_ring_addr + (unsigned int)(slot * 1024 + sf_token_30 * 16)));
                sf_entry_35[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                sf_entry_35[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                sf_entry_35[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                sf_entry_35[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                unsigned long long k_src_base_36 = k_base + col_off_25;
                unsigned long long v_src_base_37 = v_base + col_off_25;
                v_bits_b[0] = bits_sub_24;
                unsigned long long data_off_38 = (unsigned long long)entries_0_31[0] | (unsigned long long)entries_0_31[1] << 32;
                unsigned int src_size_39 = (bits_sub_24 & 1) << 4;
                v_src_b[0] = v_src_base_37 + data_off_38;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0_27), "l"(k_src_base_36 + data_off_38), "r"(src_size_39), "l"(0x12F0000000000000ULL));
                unsigned long long data_off_40 = (unsigned long long)entries_1_32[0] | (unsigned long long)entries_1_32[1] << 32;
                unsigned int src_size_41 = (bits_sub_24 >> 4 & 1) << 4;
                v_src_b[1] = v_src_base_37 + data_off_40;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0_27 + 576), "l"(k_src_base_36 + data_off_40), "r"(src_size_41), "l"(0x12F0000000000000ULL));
                unsigned long long data_off_42 = (unsigned long long)entries_2_33[0] | (unsigned long long)entries_2_33[1] << 32;
                unsigned int src_size_43 = (bits_sub_24 >> 8 & 1) << 4;
                v_src_b[2] = v_src_base_37 + data_off_42;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0_27 + 1152), "l"(k_src_base_36 + data_off_42), "r"(src_size_43), "l"(0x12F0000000000000ULL));
                unsigned long long data_off_44 = (unsigned long long)entries_3_34[0] | (unsigned long long)entries_3_34[1] << 32;
                unsigned int src_size_45 = (bits_sub_24 >> 12 & 1) << 4;
                v_src_b[3] = v_src_base_37 + data_off_44;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_k0_27 + 1728), "l"(k_src_base_36 + data_off_44), "r"(src_size_45), "l"(0x12F0000000000000ULL));
                unsigned long long sf_off_46 = (unsigned long long)sf_entry_35[2] | (unsigned long long)sf_entry_35[3] << 32;
                unsigned long long sf_base_47 = ((sf_is_v_29) ? v_sf_base : k_sf_base);
                int sf_side_48 = ((sf_is_v_29) ? 9216 : 0);
                unsigned int sf_size_49 = (bits_19 >> (unsigned int)sf_lane_28 & 1) << 4;
                unsigned int sf_dst_50 = ((stage_base_26 + (unsigned int)(sf_side_48 + sf_token_30 * 144 + 128)) < (kv_addr + 73712) ? (stage_base_26 + (unsigned int)(sf_side_48 + sf_token_30 * 144 + 128)) : (kv_addr + 73712));
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(sf_dst_50), "l"(sf_base_47 + sf_off_46), "r"(sf_size_49), "l"(0x12F0000000000000ULL));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(raw_full_k_addr + ((int)((unsigned int)n_3 % 4)) * 8) : "memory");
                int col_51 = lane_4 & 7;
                int sub_52 = lane_4 >> 3;
                int token0_53 = 16 * g_1 + sub_52;
                unsigned int dst_v0 = ((kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_53 * 144 + col_51 * 16)) < (kv_addr + 73712) ? (kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_53 * 144 + col_51 * 16)) : (kv_addr + 73712));
                unsigned int src_size_54 = (v_bits_a[0] & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0), "l"(v_src_a[0]), "r"(src_size_54), "l"(0x12F0000000000000ULL));
                unsigned int src_size_55 = (v_bits_a[0] >> 4 & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0 + 576), "l"(v_src_a[1]), "r"(src_size_55), "l"(0x12F0000000000000ULL));
                unsigned int src_size_56 = (v_bits_a[0] >> 8 & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0 + 1152), "l"(v_src_a[2]), "r"(src_size_56), "l"(0x12F0000000000000ULL));
                unsigned int src_size_57 = (v_bits_a[0] >> 12 & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0 + 1728), "l"(v_src_a[3]), "r"(src_size_57), "l"(0x12F0000000000000ULL));
                int col_58 = lane_4 & 7;
                int sub_59 = lane_4 >> 3;
                int token0_60 = 16 * (g_1 + 2) + sub_59;
                unsigned int dst_v0_61 = ((kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_60 * 144 + col_58 * 16)) < (kv_addr + 73712) ? (kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_60 * 144 + col_58 * 16)) : (kv_addr + 73712));
                unsigned int src_size_62 = (v_bits_b[0] & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0_61), "l"(v_src_b[0]), "r"(src_size_62), "l"(0x12F0000000000000ULL));
                unsigned int src_size_63 = (v_bits_b[0] >> 4 & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0_61 + 576), "l"(v_src_b[1]), "r"(src_size_63), "l"(0x12F0000000000000ULL));
                unsigned int src_size_64 = (v_bits_b[0] >> 8 & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0_61 + 1152), "l"(v_src_b[2]), "r"(src_size_64), "l"(0x12F0000000000000ULL));
                unsigned int src_size_65 = (v_bits_b[0] >> 12 & 1) << 4;
                asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                    :: "r"(dst_v0_61 + 1728), "l"(v_src_b[3]), "r"(src_size_65), "l"(0x12F0000000000000ULL));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(raw_full_v_addr + ((int)((unsigned int)n_3 % 4)) * 8) : "memory");
                idx[0] = idx[1];
                idx[1] = idx[2];
                idx[2] = idx[3];
                idx[3] = logical_3;
                phys[0] = phys[1];
                phys[1] = phys[2];
                local[0] = local[1];
                local[1] = local[2];
            }
        }
    }
    // ---- Role: mma_s ----
    if (warp == 14) {
        { // mma_s_main
            mbarrier_wait(q_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll 1
            for (int n_4 = 0; n_4 < n_tiles; n_4++) {
                int par_1 = n_4 & 1;
                int _stage_k_full = (int)((unsigned int)n_4 % 2);
                int _parity_k_full = (int)((unsigned int)n_4 / 2 % 2);
                mbarrier_wait(k_full_addr + (_stage_k_full) * 8, _parity_k_full);
                int _stage_s_empty = (int)((unsigned int)n_4 % 2);
                int _parity_s_empty = (int)((unsigned int)n_4 / 2 % 2) ^ 1;
                mbarrier_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (par_1 == 0) {
                    uint32_t _elected_0 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                    int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_1), "r"(tmem_tmem_k), "r"(0), "r"(_elected_0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((smem_q_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                    int _smem_desc_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_2), "r"(tmem_tmem_k + 32), "r"(1), "r"(_elected_0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((smem_q_addr) >> 4) & 0x7FFF) + (2) * 128;
#else
                    int _smem_desc_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (2) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_3), "r"(tmem_tmem_k + 64), "r"(1), "r"(_elected_0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((smem_q_addr) >> 4) & 0x7FFF) + (3) * 128;
#else
                    int _smem_desc_lo_4 = (((smem_q_addr) >> 4) & 0x3FFF) + (3) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_4), "r"(tmem_tmem_k + 96), "r"(1), "r"(_elected_0) : "memory");
                    tcgen05_commit_leader(k_empty_addr + ((int)((unsigned int)n_4 % 2)) * 8, _elected_0);
                    tcgen05_commit_leader(s_full_addr + ((int)((unsigned int)n_4 % 2)) * 8, _elected_0);
                } else {
                    uint32_t _elected_5 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = (((smem_q_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                    int _smem_desc_lo_6 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_6), "r"(tmem_tmem_k + 1048576), "r"(0), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_7 = (((smem_q_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                    int _smem_desc_lo_7 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_7), "r"(tmem_tmem_k + 1048608), "r"(1), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = (((smem_q_addr) >> 4) & 0x7FFF) + (2) * 128;
#else
                    int _smem_desc_lo_8 = (((smem_q_addr) >> 4) & 0x3FFF) + (2) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_8), "r"(tmem_tmem_k + 1048640), "r"(1), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_9 = (((smem_q_addr) >> 4) & 0x7FFF) + (3) * 128;
#else
                    int _smem_desc_lo_9 = (((smem_q_addr) >> 4) & 0x3FFF) + (3) * 128;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_9), "r"(tmem_tmem_k + 1048672), "r"(1), "r"(_elected_5) : "memory");
                    tcgen05_commit_leader(k_empty_addr + ((int)((unsigned int)n_4 % 2)) * 8, _elected_5);
                    tcgen05_commit_leader(s_full_addr + ((int)((unsigned int)n_4 % 2)) * 8, _elected_5);
                }
            }
        }
    }
    // ---- Role: mma_pv ----
    if (warp == 15) {
        { // mma_pv_main
            #pragma unroll 1
            for (int n_5 = 0; n_5 < n_tiles; n_5++) {
                int vs_1 = n_5 % 2;
                int ps = n_5 % 2;
                bool first_pv = n_5 == 0;
                int _stage_v_full = (int)((unsigned int)n_5 % 2);
                int _parity_v_full = (int)((unsigned int)n_5 / 2 % 2);
                mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                int _stage_p_full = (int)((unsigned int)n_5 % 2);
                int _parity_p_full = (int)((unsigned int)n_5 / 2 % 2);
                mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_0 = elect_sync();
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_1_1 = ((((smem_vt_addr) >> 4) & 0x7FFF) | 0x2000000) + (2 * vs_1) * 1024;
                int _smem_desc_lo_2_1 = ((((smem_p_addr) >> 4) & 0x7FFF) | 0x800000) + (ps) * 128;
#else
                int _smem_desc_lo_1_1 = ((((smem_vt_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * vs_1) * 1024;
                int _smem_desc_lo_2_1 = ((((smem_p_addr) >> 4) & 0x3FFF) | 0x800000) + (ps) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_1_1), "r"(_smem_desc_lo_2_1), "r"(tmem_tmem_o), "r"(((first_pv) ? 0 : 1)), "r"(_elected_0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_3_1 = ((((smem_vt_addr) >> 4) & 0x7FFF) | 0x2000000) + (2 * vs_1 + 1) * 1024;
                int _smem_desc_lo_4_1 = ((((smem_p_addr) >> 4) & 0x7FFF) | 0x800000) + (ps) * 128;
#else
                int _smem_desc_lo_3_1 = ((((smem_vt_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * vs_1 + 1) * 1024;
                int _smem_desc_lo_4_1 = ((((smem_p_addr) >> 4) & 0x3FFF) | 0x800000) + (ps) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_3_1), "r"(_smem_desc_lo_4_1), "r"((tmem_tmem_o + (16))), "r"(((first_pv) ? 0 : 1)), "r"(_elected_0) : "memory");
                tcgen05_commit_leader(v_empty_addr + ((int)((unsigned int)n_5 % 2)) * 8, _elected_0);
                tcgen05_commit_leader(pv_done_addr + ((int)((unsigned int)n_5 % 2)) * 8, _elected_0);
            }
            mbarrier_wait(tmem_dead_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }

    // Cleanup
}

} // extern "C"
