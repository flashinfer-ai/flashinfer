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
#define SMEM_SMEM_VT_OFF 25600
#define SMEM_SMEM_VT_STAGE_BYTES 16384
#define SMEM_SMEM_VT_STRIDE 16384
#define SMEM_SMEM_P_OFF 91136
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_XCHG_OFF 95232
#define SMEM_XCHG_STAGE_BYTES 1024
#define SMEM_XCHG_STRIDE 1024
#define SMEM_VALID_S_OFF 96256
#define SMEM_VALID_S_STAGE_BYTES 32
#define SMEM_VALID_S_STRIDE 32
#define SMEM_SLOT_VALID_OFF 96288
#define SMEM_SLOT_VALID_STAGE_BYTES 16
#define SMEM_SLOT_VALID_STRIDE 16
#define SMEM_ADDR_RING_OFF 96384
#define SMEM_ADDR_RING_STAGE_BYTES 2048
#define SMEM_ADDR_RING_STRIDE 2048
#define SMEM_RAW_U8_OFF 102528
#define SMEM_RAW_U8_STAGE_BYTES 73728
#define SMEM_RAW_U8_STRIDE 73728
#define SMEM_O_XCHG_OFF 176256
#define SMEM_O_XCHG_STAGE_BYTES 16640
#define SMEM_O_XCHG_STRIDE 16640
#define SMEM_LSE_TAB_OFF 102528
#define SMEM_LSE_TAB_STAGE_BYTES 36800
#define SMEM_LSE_TAB_STRIDE 36800
#define SMEM_MAX_TAB_OFF 139328
#define SMEM_MAX_TAB_STAGE_BYTES 64
#define SMEM_MAX_TAB_STRIDE 64
#define SMEM_TOTAL 192896
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



__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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
kernel_cake_qsa_nvfp4_dfd8600d03040c9f808d(__nv_bfloat16* __restrict__ q, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int* __restrict__ indices, int* __restrict__ block_table, int* __restrict__ token_to_req, __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, float* partial_out, float* partial_lse, unsigned int* __restrict__ merge_counters, long long q_stride_row, long long q_stride_head, long long data_stride_block, long long data_stride_token, long long data_stride_head, long long sf_stride_block, long long sf_stride_token, long long sf_stride_head, long long indices_stride_row, long long table_stride_req, long long out_stride_row, long long out_stride_head, float score_scale, float output_scale, int num_rows, int num_requests, int num_blocks, CakeFastDivmod page_div, int table_width, int topk, int group_size, int num_qo_heads, int num_tiles, int num_splits)
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
    #define q_empty_addr (mbar_base + 312)
    #define q_landed_addr (mbar_base + 336)
    #define tmem_dead_addr (mbar_base + 360)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qt_addr = smem + 1024;
    __nv_bfloat16* smem_vt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_vt_addr = smem + 25600;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int smem_p_addr = smem + 91136;
    float* xchg = reinterpret_cast<float*>(smem_raw + 95232);
    const int xchg_addr = smem + 95232;
    unsigned int* valid_s = reinterpret_cast<unsigned int*>(smem_raw + 96256);
    const int valid_s_addr = smem + 96256;
    unsigned int* slot_valid = reinterpret_cast<unsigned int*>(smem_raw + 96288);
    const int slot_valid_addr = smem + 96288;
    unsigned int* addr_ring = reinterpret_cast<unsigned int*>(smem_raw + 96384);
    const int addr_ring_addr = smem + 96384;
    uint8_t* raw_u8 = reinterpret_cast<uint8_t*>(smem_raw + 102528);
    const int raw_u8_addr = smem + 102528;
    float* o_xchg = reinterpret_cast<float*>(smem_raw + 176256);
    const int o_xchg_addr = smem + 176256;
    float* lse_tab = reinterpret_cast<float*>(smem_raw + 102528);
    const int lse_tab_addr = smem + 102528;
    float* max_tab = reinterpret_cast<float*>(smem_raw + 139328);
    const int max_tab_addr = smem + 139328;
    int cta = blockIdx.x;
    int ctas = gridDim.x;
    long long cta_i64 = (long long)cta;
    long long ctas_i64 = (long long)ctas;
    int hkv = num_qo_heads / group_size;
    long long tiles_i64 = (long long)num_tiles;
    long long slots = (long long)num_rows * (long long)hkv * tiles_i64;
#if __CUDA_ARCH__ == 1070
    long long slot0 = 0;
    long long slot1 = 0;
    int pair0 = 0;
    int pair1 = 0;
    if (slots * ctas_i64 < 2147483648) {
        int slots32 = (int)slots;
        int s0 = cta * slots32 / ctas;
        int s1 = (cta + 1) * slots32 / ctas;
        slot0 = (long long)s0;
        slot1 = (long long)s1;
        pair0 = s0 / num_tiles;
        pair1 = (s1 - 1) / num_tiles;
    } else {
        slot0 = cta_i64 * slots / ctas_i64;
        slot1 = (cta_i64 + 1) * slots / ctas_i64;
        pair0 = (int)(slot0 / tiles_i64);
        pair1 = (int)((slot1 - 1) / tiles_i64);
    }
#else
    long long slot0 = cta_i64 * slots / ctas_i64;
    long long slot1 = (cta_i64 + 1) * slots / ctas_i64;
    int pair0 = (int)(slot0 / tiles_i64);
    int pair1 = (int)((slot1 - 1) / tiles_i64);
#endif
    int row = pair0 / hkv;
    int kv_head = pair0 - row * hkv;
    long long row_i64 = (long long)row;
    int first_head = kv_head * group_size;
    long long indices_row = row_i64 * indices_stride_row;
    long long pslot = (long long)pair0 * tiles_i64;
    long long _max_0 = ((slot0 - pslot) > (0) ? (slot0 - pslot) : (0));
    int tile0 = (int)_max_0;
    long long _min_1 = ((slot1 - pslot) < (tiles_i64) ? (slot1 - pslot) : (tiles_i64));
    int tile1 = (int)_min_1;
    int owner0 = (int)(((pslot + 1) * ctas_i64 - 1) / slots);
    int cta_0 = blockIdx.x;
    int seg = cta_0 - owner0;
    bool last = slot1 >= pslot + tiles_i64;
    int request = token_to_req[row];
    int valid_count = indices[indices_row + (long long)topk];
    int g_nxt = ((pair0 + 1) < (pair1) ? (pair0 + 1) : (pair1));
    int g_nrow = g_nxt / hkv;
    long long g_nidx_row = (long long)g_nrow * indices_stride_row;
    int request_0 = token_to_req[g_nrow];
    int valid_count_0 = indices[g_nidx_row + (long long)topk];
    int idx[4];
    unsigned int qw_regs[32];
    if (tid >= 384) {
        int gtid0 = tid - 384;
        int column = tile0 * 64 + gtid0;
        int safe_column = ((column) < (topk - 1) ? (column) : (topk - 1));
        int raw = indices[indices_row + (long long)safe_column];
        idx[0] = raw;
        int column_0 = (tile0 + 1) * 64 + gtid0;
        int safe_column_0 = ((column_0) < (topk - 1) ? (column_0) : (topk - 1));
        int raw_0 = indices[indices_row + (long long)safe_column_0];
        idx[1] = raw_0;
        int column_1 = (tile0 + 2) * 64 + gtid0;
        int safe_column_1 = ((column_1) < (topk - 1) ? (column_1) : (topk - 1));
        int raw_2 = indices[indices_row + (long long)safe_column_1];
        idx[2] = raw_2;
        int column_2 = (tile0 + 3) * 64 + gtid0;
        int safe_column_2 = ((column_2) < (topk - 1) ? (column_2) : (topk - 1));
        int raw_3 = indices[indices_row + (long long)safe_column_2];
        idx[3] = raw_3;
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
        int e_1_0 = gtid0;
        int head_2_0 = e_1_0 >> 5;
        int c_1_0 = e_1_0 & 31;
        if (head_2_0 < group_size) {
            long long q_src_1_0 = q_row + (long long)(first_head + head_2_0) * q_stride_head + (long long)(8 * c_1_0);
            unsigned int _vec_load_0[4];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(q + q_src_1_0 + 0);
                uint4* _vdst_2 = reinterpret_cast<uint4*>(&_vec_load_0[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_2[_blk] = _vptr_2[_blk];
                }
            }
            qw_regs[0] = _vec_load_0[0];
            qw_regs[1] = _vec_load_0[1];
            qw_regs[2] = _vec_load_0[2];
            qw_regs[3] = _vec_load_0[3];
        }
        int e_1_1 = gtid0 + 64;
        int head_2_1 = e_1_1 >> 5;
        int c_1_1 = e_1_1 & 31;
        if (head_2_1 < group_size) {
            long long q_src_1_1 = q_row + (long long)(first_head + head_2_1) * q_stride_head + (long long)(8 * c_1_1);
            unsigned int _vec_load_1[4];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(q + q_src_1_1 + 0);
                uint4* _vdst_3 = reinterpret_cast<uint4*>(&_vec_load_1[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_3[_blk] = _vptr_3[_blk];
                }
            }
            qw_regs[4] = _vec_load_1[0];
            qw_regs[5] = _vec_load_1[1];
            qw_regs[6] = _vec_load_1[2];
            qw_regs[7] = _vec_load_1[3];
        }
        int e_1_2 = gtid0 + 128;
        int head_2_2 = e_1_2 >> 5;
        int c_1_2 = e_1_2 & 31;
        if (head_2_2 < group_size) {
            long long q_src_1_2 = q_row + (long long)(first_head + head_2_2) * q_stride_head + (long long)(8 * c_1_2);
            unsigned int _vec_load_2[4];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(q + q_src_1_2 + 0);
                uint4* _vdst_4 = reinterpret_cast<uint4*>(&_vec_load_2[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_4[_blk] = _vptr_4[_blk];
                }
            }
            qw_regs[8] = _vec_load_2[0];
            qw_regs[9] = _vec_load_2[1];
            qw_regs[10] = _vec_load_2[2];
            qw_regs[11] = _vec_load_2[3];
        }
        int e_1_3 = gtid0 + 192;
        int head_2_3 = e_1_3 >> 5;
        int c_1_3 = e_1_3 & 31;
        if (head_2_3 < group_size) {
            long long q_src_1_3 = q_row + (long long)(first_head + head_2_3) * q_stride_head + (long long)(8 * c_1_3);
            unsigned int _vec_load_3[4];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(q + q_src_1_3 + 0);
                uint4* _vdst_5 = reinterpret_cast<uint4*>(&_vec_load_3[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_5[_blk] = _vptr_5[_blk];
                }
            }
            qw_regs[12] = _vec_load_3[0];
            qw_regs[13] = _vec_load_3[1];
            qw_regs[14] = _vec_load_3[2];
            qw_regs[15] = _vec_load_3[3];
        }
        int e_1_4 = gtid0 + 256;
        int head_2_4 = e_1_4 >> 5;
        int c_1_4 = e_1_4 & 31;
        if (head_2_4 < group_size) {
            long long q_src_1_4 = q_row + (long long)(first_head + head_2_4) * q_stride_head + (long long)(8 * c_1_4);
            unsigned int _vec_load_4[4];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(q + q_src_1_4 + 0);
                uint4* _vdst_6 = reinterpret_cast<uint4*>(&_vec_load_4[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_6[_blk] = _vptr_6[_blk];
                }
            }
            qw_regs[16] = _vec_load_4[0];
            qw_regs[17] = _vec_load_4[1];
            qw_regs[18] = _vec_load_4[2];
            qw_regs[19] = _vec_load_4[3];
        }
        int e_1_5 = gtid0 + 320;
        int head_2_5 = e_1_5 >> 5;
        int c_1_5 = e_1_5 & 31;
        if (head_2_5 < group_size) {
            long long q_src_1_5 = q_row + (long long)(first_head + head_2_5) * q_stride_head + (long long)(8 * c_1_5);
            unsigned int _vec_load_5[4];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(q + q_src_1_5 + 0);
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&_vec_load_5[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_7[_blk] = _vptr_7[_blk];
                }
            }
            qw_regs[20] = _vec_load_5[0];
            qw_regs[21] = _vec_load_5[1];
            qw_regs[22] = _vec_load_5[2];
            qw_regs[23] = _vec_load_5[3];
        }
        int e_1_6 = gtid0 + 384;
        int head_2_6 = e_1_6 >> 5;
        int c_1_6 = e_1_6 & 31;
        if (head_2_6 < group_size) {
            long long q_src_1_6 = q_row + (long long)(first_head + head_2_6) * q_stride_head + (long long)(8 * c_1_6);
            unsigned int _vec_load_6[4];
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(q + q_src_1_6 + 0);
                uint4* _vdst_8 = reinterpret_cast<uint4*>(&_vec_load_6[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_8[_blk] = _vptr_8[_blk];
                }
            }
            qw_regs[24] = _vec_load_6[0];
            qw_regs[25] = _vec_load_6[1];
            qw_regs[26] = _vec_load_6[2];
            qw_regs[27] = _vec_load_6[3];
        }
        int e_1_7 = gtid0 + 448;
        int head_2_7 = e_1_7 >> 5;
        int c_1_7 = e_1_7 & 31;
        if (head_2_7 < group_size) {
            long long q_src_1_7 = q_row + (long long)(first_head + head_2_7) * q_stride_head + (long long)(8 * c_1_7);
            unsigned int _vec_load_7[4];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(q + q_src_1_7 + 0);
                uint4* _vdst_9 = reinterpret_cast<uint4*>(&_vec_load_7[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_9[_blk] = _vptr_9[_blk];
                }
            }
            qw_regs[28] = _vec_load_7[0];
            qw_regs[29] = _vec_load_7[1];
            qw_regs[30] = _vec_load_7[2];
            qw_regs[31] = _vec_load_7[3];
        }
    }

    // Mbarrier init (17 pipeline groups, 0 ordered-sequence groups, 46 barriers)
    // Mbarriers at smem_raw[0..368)

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
            // q_full: 3 barriers, init_count=2
            mbarrier_init(smem + 288, 2);
            mbarrier_init(smem + 296, 2);
            mbarrier_init(smem + 304, 2);
            // q_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // q_landed: 3 barriers, init_count=64
            mbarrier_init(smem + 336, 64);
            mbarrier_init(smem + 344, 64);
            mbarrier_init(smem + 352, 64);
            // tmem_dead: 1 barriers, init_count=4
            mbarrier_init(smem + 360, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 368);
    if (warp == 0) {
        int _tmem_hold = smem + 368;
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
    int _max_0_1 = ((request) > (0) ? (request) : (0));
    int safe_request = ((_max_0_1) < (num_requests - 1) ? (_max_0_1) : (num_requests - 1));
    long long safe_request_row = (long long)safe_request * table_stride_req;
    int _min_2 = ((valid_count) < (topk) ? (valid_count) : (topk));
    int _min_1_1 = ((num_tiles) < ((_min_2 + 63) / 64) ? (num_tiles) : ((_min_2 + 63) / 64));
    int tile_end = ((_min_1_1) < (tile1) ? (_min_1_1) : (tile1));
    int n_tiles = ((tile_end - tile0) > (0) ? (tile_end - tile0) : (0));
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
            int nt0 = 0;
            int req_cur = request;
            int vc_cur = valid_count;
            #pragma unroll 1
            for (int P = pair0; P < pair1 + 1; P++) {
#if __CUDA_ARCH__ == 1070
                int row_4 = P / hkv;
                int kv_head_4 = P - row_4 * hkv;
                long long row_i64_4 = (long long)row_4;
                int first_head_4 = kv_head_4 * group_size;
                long long indices_row_4 = row_i64_4 * indices_stride_row;
                long long pslot_4 = (long long)P * tiles_i64;
                long long _max_0_2 = ((slot0 - pslot_4) > (0) ? (slot0 - pslot_4) : (0));
                int tile0_4 = (int)_max_0_2;
                long long _min_1_2 = ((slot1 - pslot_4) < (tiles_i64) ? (slot1 - pslot_4) : (tiles_i64));
                int tile1_4 = (int)_min_1_2;
                int owner0_4 = (int)(((pslot_4 + 1) * ctas_i64 - 1) / slots);
                int cta_5 = blockIdx.x;
                int seg_4 = cta_5 - owner0_4;
                bool last_4 = slot1 >= pslot_4 + tiles_i64;
                bool request_ok_4 = req_cur >= 0 && req_cur < num_requests;
#else
                int row_5 = P / hkv;
                int kv_head_5 = P - row_5 * hkv;
                long long row_i64_5 = (long long)row_5;
                int first_head_5 = kv_head_5 * group_size;
                long long indices_row_5 = row_i64_5 * indices_stride_row;
                long long pslot_5 = (long long)P * tiles_i64;
                long long _max_0_2 = ((slot0 - pslot_5) > (0) ? (slot0 - pslot_5) : (0));
                int tile0_5 = (int)_max_0_2;
                long long _min_1_2 = ((slot1 - pslot_5) < (tiles_i64) ? (slot1 - pslot_5) : (tiles_i64));
                int tile1_5 = (int)_min_1_2;
                int owner0_5 = (int)(((pslot_5 + 1) * ctas_i64 - 1) / slots);
                int cta_6 = blockIdx.x;
                int seg_5 = cta_6 - owner0_5;
                bool last_5 = slot1 >= pslot_5 + tiles_i64;
                bool request_ok_5 = req_cur >= 0 && req_cur < num_requests;
#endif
                int _max_2 = ((req_cur) > (0) ? (req_cur) : (0));
#if __CUDA_ARCH__ == 1070
                int safe_request_4 = ((_max_2) < (num_requests - 1) ? (_max_2) : (num_requests - 1));
                long long safe_request_row_4 = (long long)safe_request_4 * table_stride_req;
#else
                int safe_request_5 = ((_max_2) < (num_requests - 1) ? (_max_2) : (num_requests - 1));
                long long safe_request_row_5 = (long long)safe_request_5 * table_stride_req;
#endif
                int _min_4 = ((vc_cur) < (topk) ? (vc_cur) : (topk));
                int _min_3 = ((num_tiles) < ((_min_4 + 63) / 64) ? (num_tiles) : ((_min_4 + 63) / 64));
#if __CUDA_ARCH__ == 1070
                int tile_end_4 = ((_min_3) < (tile1_4) ? (_min_3) : (tile1_4));
                int n_tiles_4 = ((tile_end_4 - tile0_4) > (0) ? (tile_end_4 - tile0_4) : (0));
#else
                int tile_end_5 = ((_min_3) < (tile1_5) ? (_min_3) : (tile1_5));
                int n_tiles_5 = ((tile_end_5 - tile0_5) > (0) ? (tile_end_5 - tile0_5) : (0));
#endif
                int nxt = ((P + 1) < (pair1) ? (P + 1) : (pair1));
                int nrow = nxt / hkv;
                long long nrow_i64 = (long long)nrow;
                int nfirst = (nxt - nrow * hkv) * group_size;
                long long nidx_row = nrow_i64 * indices_stride_row;
#if __CUDA_ARCH__ == 1070
                int request_8 = token_to_req[nrow];
                int valid_count_8 = indices[nidx_row + (long long)topk];
#else
                int request_7 = token_to_req[nrow];
                int valid_count_7 = indices[nidx_row + (long long)topk];
#endif
                if (pair0 < P) {
                    m[0] = -1e+20f;
                    l[0] = 0.0f;
                    m[1] = -1e+20f;
                    l[1] = 0.0f;
                    m[2] = -1e+20f;
                    l[2] = 0.0f;
                    m[3] = -1e+20f;
                    l[3] = 0.0f;
                    m[4] = -1e+20f;
                    l[4] = 0.0f;
                    m[5] = -1e+20f;
                    l[5] = 0.0f;
                    m[6] = -1e+20f;
                    l[6] = 0.0f;
                    m[7] = -1e+20f;
                    l[7] = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (n_tiles_4 > 0) {
#else
                if (n_tiles_5 > 0) {
#endif
                    int kk = nt0;
                    int ks = kk % 2;
                    int kr = kk % 4;
                    int _stage_s_full = (int)((unsigned int)nt0 % 2);
                    int _parity_s_full = (int)((unsigned int)nt0 / 2 % 2);
                    mbarrier_wait(s_full_addr + (_stage_s_full) * 8, _parity_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int s_addr = s_lane + (unsigned int)(16 * ks) + (unsigned int)(16 * ks << 16);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sv_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[7]))
                        : "r"(s_addr));
                    int _stage_valid_full = (int)((unsigned int)nt0 % 4);
                    int _parity_valid_full = (int)((unsigned int)nt0 / 4 % 2);
                    mbarrier_wait(valid_full_addr + (_stage_valid_full) * 8, _parity_valid_full);
                    unsigned int bits = valid_s[kr * 2 + (tok >> 5)];
                    okn[0] = bits >> tok_bit & 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(valid_empty_addr + ((int)((unsigned int)nt0 % 4)) * 8);
                    }
                }
                #pragma unroll 1
#if __CUDA_ARCH__ == 1070
                for (int n = 0; n < n_tiles_4; n++) {
#else
                for (int n = 0; n < n_tiles_5; n++) {
#endif
                    int nt = nt0 + n;
                    int ss = nt % 2;
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + ((int)((unsigned int)(nt0 + n) % 2)) * 8);
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
                    float rv_15_0 = ((half == 0) ? mx[0] : -1e+20f);
                    float _warp_redux_5;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_5) : "f"(rv_15_0));
                    r16[0] = _warp_redux_5;
                    float rv_15_1 = ((half == 0) ? mx[1] : -1e+20f);
                    float _warp_redux_6;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_6) : "f"(rv_15_1));
                    r16[1] = _warp_redux_6;
                    float rv_15_2 = ((half == 0) ? mx[2] : -1e+20f);
                    float _warp_redux_7;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_7) : "f"(rv_15_2));
                    r16[2] = _warp_redux_7;
                    float rv_15_3 = ((half == 0) ? mx[3] : -1e+20f);
                    float _warp_redux_8;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_8) : "f"(rv_15_3));
                    r16[3] = _warp_redux_8;
                    float rv_15_4 = ((half == 0) ? mx[4] : -1e+20f);
                    float _warp_redux_9;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_9) : "f"(rv_15_4));
                    r16[4] = _warp_redux_9;
                    float rv_15_5 = ((half == 0) ? mx[5] : -1e+20f);
                    float _warp_redux_10;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_10) : "f"(rv_15_5));
                    r16[5] = _warp_redux_10;
                    float rv_15_6 = ((half == 0) ? mx[6] : -1e+20f);
                    float _warp_redux_11;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_11) : "f"(rv_15_6));
                    r16[6] = _warp_redux_11;
                    float rv_15_7 = ((half == 0) ? mx[7] : -1e+20f);
                    float _warp_redux_12;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_12) : "f"(rv_15_7));
                    r16[7] = _warp_redux_12;
                    float rv_15_8 = ((half == 1) ? mx[0] : -1e+20f);
                    float _warp_redux_13;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_13) : "f"(rv_15_8));
                    r16[8] = _warp_redux_13;
                    float rv_15_9 = ((half == 1) ? mx[1] : -1e+20f);
                    float _warp_redux_14;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_14) : "f"(rv_15_9));
                    r16[9] = _warp_redux_14;
                    float rv_15_10 = ((half == 1) ? mx[2] : -1e+20f);
                    float _warp_redux_15;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_15) : "f"(rv_15_10));
                    r16[10] = _warp_redux_15;
                    float rv_15_11 = ((half == 1) ? mx[3] : -1e+20f);
                    float _warp_redux_16;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_16) : "f"(rv_15_11));
                    r16[11] = _warp_redux_16;
                    float rv_15_12 = ((half == 1) ? mx[4] : -1e+20f);
                    float _warp_redux_17;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_17) : "f"(rv_15_12));
                    r16[12] = _warp_redux_17;
                    float rv_15_13 = ((half == 1) ? mx[5] : -1e+20f);
                    float _warp_redux_18;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_18) : "f"(rv_15_13));
                    r16[13] = _warp_redux_18;
                    float rv_15_14 = ((half == 1) ? mx[6] : -1e+20f);
                    float _warp_redux_19;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_19) : "f"(rv_15_14));
                    r16[14] = _warp_redux_19;
                    float rv_15_15 = ((half == 1) ? mx[7] : -1e+20f);
                    float _warp_redux_20;
                    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_20) : "f"(rv_15_15));
                    r16[15] = _warp_redux_20;
                    unsigned int xb = xchg_max + (unsigned int)(nt & 1) * 768;
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
                    float _max3_22;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(m[0]), "f"(mw[0]), "f"(mw[8]));
                    float _max3_21;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(_max3_22), "f"(mw[16]), "f"(mw[24]));
                    m_new[0] = _max3_21;
                    moved = moved || m_new[0] != m[0];
                    float _max3_24;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(m[1]), "f"(mw[1]), "f"(mw[9]));
                    float _max3_23;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_23) : "f"(_max3_24), "f"(mw[17]), "f"(mw[25]));
                    m_new[1] = _max3_23;
                    moved = moved || m_new[1] != m[1];
                    float _max3_26;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(m[2]), "f"(mw[2]), "f"(mw[10]));
                    float _max3_25;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(_max3_26), "f"(mw[18]), "f"(mw[26]));
                    m_new[2] = _max3_25;
                    moved = moved || m_new[2] != m[2];
                    float _max3_28;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(m[3]), "f"(mw[3]), "f"(mw[11]));
                    float _max3_27;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(_max3_28), "f"(mw[19]), "f"(mw[27]));
                    m_new[3] = _max3_27;
                    moved = moved || m_new[3] != m[3];
                    float _max3_30;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(m[4]), "f"(mw[4]), "f"(mw[12]));
                    float _max3_29;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(_max3_30), "f"(mw[20]), "f"(mw[28]));
                    m_new[4] = _max3_29;
                    moved = moved || m_new[4] != m[4];
                    float _max3_32;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(m[5]), "f"(mw[5]), "f"(mw[13]));
                    float _max3_31;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_31) : "f"(_max3_32), "f"(mw[21]), "f"(mw[29]));
                    m_new[5] = _max3_31;
                    moved = moved || m_new[5] != m[5];
                    float _max3_34;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_34) : "f"(m[6]), "f"(mw[6]), "f"(mw[14]));
                    float _max3_33;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(_max3_34), "f"(mw[22]), "f"(mw[30]));
                    m_new[6] = _max3_33;
                    moved = moved || m_new[6] != m[6];
                    float _max3_36;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_36) : "f"(m[7]), "f"(mw[7]), "f"(mw[15]));
                    float _max3_35;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_35) : "f"(_max3_36), "f"(mw[23]), "f"(mw[31]));
                    m_new[7] = _max3_35;
                    moved = moved || m_new[7] != m[7];
                    int _warp_vote_37 = __any_sync(0xFFFFFFFF, moved);
                    bool any_rescale = _warp_vote_37;
                    float _exp2_38 = approx_exp2(s[0] - m_new[0]);
                    p[0] = ((ok != 0) ? _exp2_38 : 0.0f);
                    float _exp2_39 = approx_exp2(s[1] - m_new[1]);
                    p[1] = ((ok != 0) ? _exp2_39 : 0.0f);
                    float _exp2_40 = approx_exp2(s[2] - m_new[2]);
                    p[2] = ((ok != 0) ? _exp2_40 : 0.0f);
                    float _exp2_41 = approx_exp2(s[3] - m_new[3]);
                    p[3] = ((ok != 0) ? _exp2_41 : 0.0f);
                    float _exp2_42 = approx_exp2(s[4] - m_new[4]);
                    p[4] = ((ok != 0) ? _exp2_42 : 0.0f);
                    float _exp2_43 = approx_exp2(s[5] - m_new[5]);
                    p[5] = ((ok != 0) ? _exp2_43 : 0.0f);
                    float _exp2_44 = approx_exp2(s[6] - m_new[6]);
                    p[6] = ((ok != 0) ? _exp2_44 : 0.0f);
                    float _exp2_45 = approx_exp2(s[7] - m_new[7]);
                    p[7] = ((ok != 0) ? _exp2_45 : 0.0f);
                    if (any_rescale) {
                        float _exp2_46 = approx_exp2(m[0] - m_new[0]);
                        alpha[0] = _exp2_46;
                        l[0] = __fmaf_rn(l[0], alpha[0], p[0]);
                        float _exp2_47 = approx_exp2(m[1] - m_new[1]);
                        alpha[1] = _exp2_47;
                        l[1] = __fmaf_rn(l[1], alpha[1], p[1]);
                        float _exp2_48 = approx_exp2(m[2] - m_new[2]);
                        alpha[2] = _exp2_48;
                        l[2] = __fmaf_rn(l[2], alpha[2], p[2]);
                        float _exp2_49 = approx_exp2(m[3] - m_new[3]);
                        alpha[3] = _exp2_49;
                        l[3] = __fmaf_rn(l[3], alpha[3], p[3]);
                        float _exp2_50 = approx_exp2(m[4] - m_new[4]);
                        alpha[4] = _exp2_50;
                        l[4] = __fmaf_rn(l[4], alpha[4], p[4]);
                        float _exp2_51 = approx_exp2(m[5] - m_new[5]);
                        alpha[5] = _exp2_51;
                        l[5] = __fmaf_rn(l[5], alpha[5], p[5]);
                        float _exp2_52 = approx_exp2(m[6] - m_new[6]);
                        alpha[6] = _exp2_52;
                        l[6] = __fmaf_rn(l[6], alpha[6], p[6]);
                        float _exp2_53 = approx_exp2(m[7] - m_new[7]);
                        alpha[7] = _exp2_53;
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
                    if (nt >= 2) {
                        int _stage_pv_done = (int)((unsigned int)(nt0 + n - 2) % 2);
                        int _parity_pv_done = (int)((unsigned int)(nt0 + n - 2) / 2 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done) * 8, _parity_pv_done);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)(ss * 2048) + (unsigned int)(tok * 32 + 16 * half ^ (tok * 32 + 16 * half >> 7 & 1) << 4))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[0], p[1])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[2], p[3])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[4], p[5])))), "r"(__as_u32(__float22bfloat162_rn(make_float2(p[6], p[7])))) : "memory");
                    if (n > 0 && any_rescale) {
                        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, alpha[0], 16);
                        other[0] = _shfl_xor_sync_54;
                        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, alpha[1], 16);
                        other[1] = _shfl_xor_sync_55;
                        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, alpha[2], 16);
                        other[2] = _shfl_xor_sync_56;
                        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, alpha[3], 16);
                        other[3] = _shfl_xor_sync_57;
                        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, alpha[4], 16);
                        other[4] = _shfl_xor_sync_58;
                        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, alpha[5], 16);
                        other[5] = _shfl_xor_sync_59;
                        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, alpha[6], 16);
                        other[6] = _shfl_xor_sync_60;
                        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, alpha[7], 16);
                        other[7] = _shfl_xor_sync_61;
                        int _stage_pv_done_1 = (int)((unsigned int)(nt0 + n - 1) % 2);
                        int _parity_pv_done_1 = (int)((unsigned int)(nt0 + n - 1) / 2 % 2);
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
                        mbarrier_arrive(p_full_addr + ((int)((unsigned int)(nt0 + n) % 2)) * 8);
                    }
#if __CUDA_ARCH__ == 1070
                    if (n_tiles_4 > n + 1) {
#else
                    if (n_tiles_5 > n + 1) {
#endif
                        int kk_1 = nt0 + n + 1;
                        int ks_1 = kk_1 % 2;
                        int kr_1 = kk_1 % 4;
                        int _stage_s_full_1 = (int)((unsigned int)(nt0 + n + 1) % 2);
                        int _parity_s_full_1 = (int)((unsigned int)(nt0 + n + 1) / 2 % 2);
                        mbarrier_wait(s_full_addr + (_stage_s_full_1) * 8, _parity_s_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int s_addr_1 = s_lane + (unsigned int)(16 * ks_1) + (unsigned int)(16 * ks_1 << 16);
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_next[7]))
                            : "r"(s_addr_1));
                        int _stage_valid_full_1 = (int)((unsigned int)(nt0 + n + 1) % 4);
                        int _parity_valid_full_1 = (int)((unsigned int)(nt0 + n + 1) / 4 % 2);
                        mbarrier_wait(valid_full_addr + (_stage_valid_full_1) * 8, _parity_valid_full_1);
                        unsigned int bits_1 = valid_s[kr_1 * 2 + (tok >> 5)];
                        okn[0] = bits_1 >> tok_bit & 1;
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(valid_empty_addr + ((int)((unsigned int)(nt0 + n + 1) % 4)) * 8);
                        }
                    }
                }
#if __CUDA_ARCH__ == 1070
                nt0 += n_tiles_4;
#else
                nt0 += n_tiles_5;
#endif
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                float o_acc[32];
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, l[0], 1);
                l[0] = l[0] + _shfl_xor_sync_62;
                float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, l[1], 1);
                l[1] = l[1] + _shfl_xor_sync_63;
                float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, l[2], 1);
                l[2] = l[2] + _shfl_xor_sync_64;
                float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, l[3], 1);
                l[3] = l[3] + _shfl_xor_sync_65;
                float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, l[4], 1);
                l[4] = l[4] + _shfl_xor_sync_66;
                float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, l[5], 1);
                l[5] = l[5] + _shfl_xor_sync_67;
                float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, l[6], 1);
                l[6] = l[6] + _shfl_xor_sync_68;
                float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, l[7], 1);
                l[7] = l[7] + _shfl_xor_sync_69;
                float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, l[0], 2);
                l[0] = l[0] + _shfl_xor_sync_70;
                float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, l[1], 2);
                l[1] = l[1] + _shfl_xor_sync_71;
                float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, l[2], 2);
                l[2] = l[2] + _shfl_xor_sync_72;
                float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, l[3], 2);
                l[3] = l[3] + _shfl_xor_sync_73;
                float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, l[4], 2);
                l[4] = l[4] + _shfl_xor_sync_74;
                float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, l[5], 2);
                l[5] = l[5] + _shfl_xor_sync_75;
                float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, l[6], 2);
                l[6] = l[6] + _shfl_xor_sync_76;
                float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, l[7], 2);
                l[7] = l[7] + _shfl_xor_sync_77;
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, l[0], 4);
                l[0] = l[0] + _shfl_xor_sync_78;
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, l[1], 4);
                l[1] = l[1] + _shfl_xor_sync_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, l[2], 4);
                l[2] = l[2] + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, l[3], 4);
                l[3] = l[3] + _shfl_xor_sync_81;
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, l[4], 4);
                l[4] = l[4] + _shfl_xor_sync_82;
                float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, l[5], 4);
                l[5] = l[5] + _shfl_xor_sync_83;
                float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, l[6], 4);
                l[6] = l[6] + _shfl_xor_sync_84;
                float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, l[7], 4);
                l[7] = l[7] + _shfl_xor_sync_85;
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, l[0], 8);
                l[0] = l[0] + _shfl_xor_sync_86;
                float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, l[1], 8);
                l[1] = l[1] + _shfl_xor_sync_87;
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, l[2], 8);
                l[2] = l[2] + _shfl_xor_sync_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, l[3], 8);
                l[3] = l[3] + _shfl_xor_sync_89;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, l[4], 8);
                l[4] = l[4] + _shfl_xor_sync_90;
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, l[5], 8);
                l[5] = l[5] + _shfl_xor_sync_91;
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, l[6], 8);
                l[6] = l[6] + _shfl_xor_sync_92;
                float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, l[7], 8);
                l[7] = l[7] + _shfl_xor_sync_93;
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
                float _rcp_94 = approx_rcp(fmaxf(l_sum[0], 1e-20f));
                inv_h[0] = output_scale * _rcp_94;
                float _rcp_95 = approx_rcp(fmaxf(l_sum[1], 1e-20f));
                inv_h[1] = output_scale * _rcp_95;
                float _rcp_96 = approx_rcp(fmaxf(l_sum[2], 1e-20f));
                inv_h[2] = output_scale * _rcp_96;
                float _rcp_97 = approx_rcp(fmaxf(l_sum[3], 1e-20f));
                inv_h[3] = output_scale * _rcp_97;
                float _rcp_98 = approx_rcp(fmaxf(l_sum[4], 1e-20f));
                inv_h[4] = output_scale * _rcp_98;
                float _rcp_99 = approx_rcp(fmaxf(l_sum[5], 1e-20f));
                inv_h[5] = output_scale * _rcp_99;
                float _rcp_100 = approx_rcp(fmaxf(l_sum[6], 1e-20f));
                inv_h[6] = output_scale * _rcp_100;
                float _rcp_101 = approx_rcp(fmaxf(l_sum[7], 1e-20f));
                inv_h[7] = output_scale * _rcp_101;
                float _rcp_102 = approx_rcp(fmaxf(l_sum[8], 1e-20f));
                inv_h[8] = output_scale * _rcp_102;
                float _rcp_103 = approx_rcp(fmaxf(l_sum[9], 1e-20f));
                inv_h[9] = output_scale * _rcp_103;
                float _rcp_104 = approx_rcp(fmaxf(l_sum[10], 1e-20f));
                inv_h[10] = output_scale * _rcp_104;
                float _rcp_105 = approx_rcp(fmaxf(l_sum[11], 1e-20f));
                inv_h[11] = output_scale * _rcp_105;
                float _rcp_106 = approx_rcp(fmaxf(l_sum[12], 1e-20f));
                inv_h[12] = output_scale * _rcp_106;
                float _rcp_107 = approx_rcp(fmaxf(l_sum[13], 1e-20f));
                inv_h[13] = output_scale * _rcp_107;
                float _rcp_108 = approx_rcp(fmaxf(l_sum[14], 1e-20f));
                inv_h[14] = output_scale * _rcp_108;
                float _rcp_109 = approx_rcp(fmaxf(l_sum[15], 1e-20f));
                inv_h[15] = output_scale * _rcp_109;
#if __CUDA_ARCH__ == 1070
                if (n_tiles_4 > 0) {
#else
                if (n_tiles_5 > 0) {
#endif
                    int _stage_pv_done_2 = (int)((unsigned int)(nt0 - 1) % 2);
                    int _parity_pv_done_2 = (int)((unsigned int)(nt0 - 1) / 2 % 2);
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
                    int head_1_0 = g;
                    if (head_1_0 < group_size) {
                        float l_r_1_0 = xchg[64 + head_1_0];
                        l_r_1_0 = l_r_1_0 + xchg[80 + head_1_0];
                        l_r_1_0 = l_r_1_0 + xchg[96 + head_1_0];
                        l_r_1_0 = l_r_1_0 + xchg[112 + head_1_0];
                        unsigned int o_src_1_0 = o_xchg_addr + (unsigned int)(4 * (head_1_0 * 260 + dim0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[0])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 3]))
                            : "r"(o_src_1_0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[4])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 3]))
                            : "r"(o_src_1_0 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[8])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 3]))
                            : "r"(o_src_1_0 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[12])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 3]))
                            : "r"(o_src_1_0 + 48));
                        bool has_values_1_0 = l_r_1_0 > 0.0f;
                        float _rcp_110 = approx_rcp(fmaxf(l_r_1_0, 1e-20f));
                        float inv_1_0 = output_scale * _rcp_110;
                        v[0] = ((has_values_1_0) ? v[0] * inv_1_0 : 0.0f);
                        v[1] = ((has_values_1_0) ? v[1] * inv_1_0 : 0.0f);
                        v[2] = ((has_values_1_0) ? v[2] * inv_1_0 : 0.0f);
                        v[3] = ((has_values_1_0) ? v[3] * inv_1_0 : 0.0f);
                        v[4] = ((has_values_1_0) ? v[4] * inv_1_0 : 0.0f);
                        v[5] = ((has_values_1_0) ? v[5] * inv_1_0 : 0.0f);
                        v[6] = ((has_values_1_0) ? v[6] * inv_1_0 : 0.0f);
                        v[7] = ((has_values_1_0) ? v[7] * inv_1_0 : 0.0f);
                        v[8] = ((has_values_1_0) ? v[8] * inv_1_0 : 0.0f);
                        v[9] = ((has_values_1_0) ? v[9] * inv_1_0 : 0.0f);
                        v[10] = ((has_values_1_0) ? v[10] * inv_1_0 : 0.0f);
                        v[11] = ((has_values_1_0) ? v[11] * inv_1_0 : 0.0f);
                        v[12] = ((has_values_1_0) ? v[12] * inv_1_0 : 0.0f);
                        v[13] = ((has_values_1_0) ? v[13] * inv_1_0 : 0.0f);
                        v[14] = ((has_values_1_0) ? v[14] * inv_1_0 : 0.0f);
                        v[15] = ((has_values_1_0) ? v[15] * inv_1_0 : 0.0f);
#if __CUDA_ARCH__ == 1070
                        long long qo_head_1_0 = (long long)(first_head_4 + head_1_0);
                        long long gate_off_1_0 = (row_i64_4 * (long long)num_qo_heads + qo_head_1_0) * 256 + (long long)dim0;
                        long long out_off_1_0 = row_i64_4 * out_stride_row + qo_head_1_0 * out_stride_head + (long long)dim0;
#else
                        long long qo_head_1_0 = (long long)(first_head_5 + head_1_0);
                        long long gate_off_1_0 = (row_i64_5 * (long long)num_qo_heads + qo_head_1_0) * 256 + (long long)dim0;
                        long long out_off_1_0 = row_i64_5 * out_stride_row + qo_head_1_0 * out_stride_head + (long long)dim0;
#endif
                        float _vec_load_8[8];
                        {
                            const uint4* _vptr_111 = reinterpret_cast<const uint4*>(gate + gate_off_1_0 + 0);
                            uint4 _vld_111[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_111[_blk] = _vptr_111[_blk];
                                uint32_t* _vpairs_111 = reinterpret_cast<uint32_t*>(&_vld_111[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) << 16);
                                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float attn_1_0 = (float)(__nv_bfloat16)v[0];
                        float _exp2_112 = approx_exp2((-_vec_load_8[0]) * 1.442695f);
                        float sigmoid_2_0 = approx_rcp(1.0f + _exp2_112);
                        gated[0] = attn_1_0 * sigmoid_2_0;
                        float attn_1_1 = (float)(__nv_bfloat16)v[1];
                        float _exp2_113 = approx_exp2((-_vec_load_8[1]) * 1.442695f);
                        float sigmoid_2_1 = approx_rcp(1.0f + _exp2_113);
                        gated[1] = attn_1_1 * sigmoid_2_1;
                        float attn_1_2 = (float)(__nv_bfloat16)v[2];
                        float _exp2_114 = approx_exp2((-_vec_load_8[2]) * 1.442695f);
                        float sigmoid_2_2 = approx_rcp(1.0f + _exp2_114);
                        gated[2] = attn_1_2 * sigmoid_2_2;
                        float attn_1_3 = (float)(__nv_bfloat16)v[3];
                        float _exp2_115 = approx_exp2((-_vec_load_8[3]) * 1.442695f);
                        float sigmoid_2_3 = approx_rcp(1.0f + _exp2_115);
                        gated[3] = attn_1_3 * sigmoid_2_3;
                        float attn_1_4 = (float)(__nv_bfloat16)v[4];
                        float _exp2_116 = approx_exp2((-_vec_load_8[4]) * 1.442695f);
                        float sigmoid_2_4 = approx_rcp(1.0f + _exp2_116);
                        gated[4] = attn_1_4 * sigmoid_2_4;
                        float attn_1_5 = (float)(__nv_bfloat16)v[5];
                        float _exp2_117 = approx_exp2((-_vec_load_8[5]) * 1.442695f);
                        float sigmoid_2_5 = approx_rcp(1.0f + _exp2_117);
                        gated[5] = attn_1_5 * sigmoid_2_5;
                        float attn_1_6 = (float)(__nv_bfloat16)v[6];
                        float _exp2_118 = approx_exp2((-_vec_load_8[6]) * 1.442695f);
                        float sigmoid_2_6 = approx_rcp(1.0f + _exp2_118);
                        gated[6] = attn_1_6 * sigmoid_2_6;
                        float attn_1_7 = (float)(__nv_bfloat16)v[7];
                        float _exp2_119 = approx_exp2((-_vec_load_8[7]) * 1.442695f);
                        float sigmoid_2_7 = approx_rcp(1.0f + _exp2_119);
                        gated[7] = attn_1_7 * sigmoid_2_7;
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
                        reinterpret_cast<int4*>(out + out_off_1_0)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                        float _vec_load_9[8];
                        {
                            const uint4* _vptr_120 = reinterpret_cast<const uint4*>(gate + (gate_off_1_0 + 8) + 0);
                            uint4 _vld_120[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_120[_blk] = _vptr_120[_blk];
                                uint32_t* _vpairs_120 = reinterpret_cast<uint32_t*>(&_vld_120[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_120[_pair]) << 16);
                                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_120[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float attn_1_8 = (float)(__nv_bfloat16)v[8];
                        float _exp2_121 = approx_exp2((-_vec_load_9[0]) * 1.442695f);
                        float sigmoid_2_8 = approx_rcp(1.0f + _exp2_121);
                        gated[0] = attn_1_8 * sigmoid_2_8;
                        float attn_1_9 = (float)(__nv_bfloat16)v[9];
                        float _exp2_122 = approx_exp2((-_vec_load_9[1]) * 1.442695f);
                        float sigmoid_2_9 = approx_rcp(1.0f + _exp2_122);
                        gated[1] = attn_1_9 * sigmoid_2_9;
                        float attn_1_10 = (float)(__nv_bfloat16)v[10];
                        float _exp2_123 = approx_exp2((-_vec_load_9[2]) * 1.442695f);
                        float sigmoid_2_10 = approx_rcp(1.0f + _exp2_123);
                        gated[2] = attn_1_10 * sigmoid_2_10;
                        float attn_1_11 = (float)(__nv_bfloat16)v[11];
                        float _exp2_124 = approx_exp2((-_vec_load_9[3]) * 1.442695f);
                        float sigmoid_2_11 = approx_rcp(1.0f + _exp2_124);
                        gated[3] = attn_1_11 * sigmoid_2_11;
                        float attn_1_12 = (float)(__nv_bfloat16)v[12];
                        float _exp2_125 = approx_exp2((-_vec_load_9[4]) * 1.442695f);
                        float sigmoid_2_12 = approx_rcp(1.0f + _exp2_125);
                        gated[4] = attn_1_12 * sigmoid_2_12;
                        float attn_1_13 = (float)(__nv_bfloat16)v[13];
                        float _exp2_126 = approx_exp2((-_vec_load_9[5]) * 1.442695f);
                        float sigmoid_2_13 = approx_rcp(1.0f + _exp2_126);
                        gated[5] = attn_1_13 * sigmoid_2_13;
                        float attn_1_14 = (float)(__nv_bfloat16)v[14];
                        float _exp2_127 = approx_exp2((-_vec_load_9[6]) * 1.442695f);
                        float sigmoid_2_14 = approx_rcp(1.0f + _exp2_127);
                        gated[6] = attn_1_14 * sigmoid_2_14;
                        float attn_1_15 = (float)(__nv_bfloat16)v[15];
                        float _exp2_128 = approx_exp2((-_vec_load_9[7]) * 1.442695f);
                        float sigmoid_2_15 = approx_rcp(1.0f + _exp2_128);
                        gated[7] = attn_1_15 * sigmoid_2_15;
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
                        reinterpret_cast<int4*>(out + (out_off_1_0 + 8))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                    }
                    int head_1_1 = g + 8;
                    if (head_1_1 < group_size) {
                        float l_r_1_1 = xchg[64 + head_1_1];
                        l_r_1_1 = l_r_1_1 + xchg[80 + head_1_1];
                        l_r_1_1 = l_r_1_1 + xchg[96 + head_1_1];
                        l_r_1_1 = l_r_1_1 + xchg[112 + head_1_1];
                        unsigned int o_src_1_1 = o_xchg_addr + (unsigned int)(4 * (head_1_1 * 260 + dim0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[0])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(0) + 3]))
                            : "r"(o_src_1_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[4])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(4) + 3]))
                            : "r"(o_src_1_1 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[8])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(8) + 3]))
                            : "r"(o_src_1_1 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v[12])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v[(12) + 3]))
                            : "r"(o_src_1_1 + 48));
                        bool has_values_1_1 = l_r_1_1 > 0.0f;
                        float _rcp_129 = approx_rcp(fmaxf(l_r_1_1, 1e-20f));
                        float inv_1_1 = output_scale * _rcp_129;
                        v[0] = ((has_values_1_1) ? v[0] * inv_1_1 : 0.0f);
                        v[1] = ((has_values_1_1) ? v[1] * inv_1_1 : 0.0f);
                        v[2] = ((has_values_1_1) ? v[2] * inv_1_1 : 0.0f);
                        v[3] = ((has_values_1_1) ? v[3] * inv_1_1 : 0.0f);
                        v[4] = ((has_values_1_1) ? v[4] * inv_1_1 : 0.0f);
                        v[5] = ((has_values_1_1) ? v[5] * inv_1_1 : 0.0f);
                        v[6] = ((has_values_1_1) ? v[6] * inv_1_1 : 0.0f);
                        v[7] = ((has_values_1_1) ? v[7] * inv_1_1 : 0.0f);
                        v[8] = ((has_values_1_1) ? v[8] * inv_1_1 : 0.0f);
                        v[9] = ((has_values_1_1) ? v[9] * inv_1_1 : 0.0f);
                        v[10] = ((has_values_1_1) ? v[10] * inv_1_1 : 0.0f);
                        v[11] = ((has_values_1_1) ? v[11] * inv_1_1 : 0.0f);
                        v[12] = ((has_values_1_1) ? v[12] * inv_1_1 : 0.0f);
                        v[13] = ((has_values_1_1) ? v[13] * inv_1_1 : 0.0f);
                        v[14] = ((has_values_1_1) ? v[14] * inv_1_1 : 0.0f);
                        v[15] = ((has_values_1_1) ? v[15] * inv_1_1 : 0.0f);
#if __CUDA_ARCH__ == 1070
                        long long qo_head_1_1 = (long long)(first_head_4 + head_1_1);
                        long long gate_off_1_1 = (row_i64_4 * (long long)num_qo_heads + qo_head_1_1) * 256 + (long long)dim0;
                        long long out_off_1_1 = row_i64_4 * out_stride_row + qo_head_1_1 * out_stride_head + (long long)dim0;
#else
                        long long qo_head_1_1 = (long long)(first_head_5 + head_1_1);
                        long long gate_off_1_1 = (row_i64_5 * (long long)num_qo_heads + qo_head_1_1) * 256 + (long long)dim0;
                        long long out_off_1_1 = row_i64_5 * out_stride_row + qo_head_1_1 * out_stride_head + (long long)dim0;
#endif
                        float _vec_load_10[8];
                        {
                            const uint4* _vptr_130 = reinterpret_cast<const uint4*>(gate + gate_off_1_1 + 0);
                            uint4 _vld_130[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_130[_blk] = _vptr_130[_blk];
                                uint32_t* _vpairs_130 = reinterpret_cast<uint32_t*>(&_vld_130[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_130[_pair]) << 16);
                                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_130[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float attn_1_16 = (float)(__nv_bfloat16)v[0];
                        float _exp2_131 = approx_exp2((-_vec_load_10[0]) * 1.442695f);
                        float sigmoid_2_16 = approx_rcp(1.0f + _exp2_131);
                        gated[0] = attn_1_16 * sigmoid_2_16;
                        float attn_1_17 = (float)(__nv_bfloat16)v[1];
                        float _exp2_132 = approx_exp2((-_vec_load_10[1]) * 1.442695f);
                        float sigmoid_2_17 = approx_rcp(1.0f + _exp2_132);
                        gated[1] = attn_1_17 * sigmoid_2_17;
                        float attn_1_18 = (float)(__nv_bfloat16)v[2];
                        float _exp2_133 = approx_exp2((-_vec_load_10[2]) * 1.442695f);
                        float sigmoid_2_18 = approx_rcp(1.0f + _exp2_133);
                        gated[2] = attn_1_18 * sigmoid_2_18;
                        float attn_1_19 = (float)(__nv_bfloat16)v[3];
                        float _exp2_134 = approx_exp2((-_vec_load_10[3]) * 1.442695f);
                        float sigmoid_2_19 = approx_rcp(1.0f + _exp2_134);
                        gated[3] = attn_1_19 * sigmoid_2_19;
                        float attn_1_20 = (float)(__nv_bfloat16)v[4];
                        float _exp2_135 = approx_exp2((-_vec_load_10[4]) * 1.442695f);
                        float sigmoid_2_20 = approx_rcp(1.0f + _exp2_135);
                        gated[4] = attn_1_20 * sigmoid_2_20;
                        float attn_1_21 = (float)(__nv_bfloat16)v[5];
                        float _exp2_136 = approx_exp2((-_vec_load_10[5]) * 1.442695f);
                        float sigmoid_2_21 = approx_rcp(1.0f + _exp2_136);
                        gated[5] = attn_1_21 * sigmoid_2_21;
                        float attn_1_22 = (float)(__nv_bfloat16)v[6];
                        float _exp2_137 = approx_exp2((-_vec_load_10[6]) * 1.442695f);
                        float sigmoid_2_22 = approx_rcp(1.0f + _exp2_137);
                        gated[6] = attn_1_22 * sigmoid_2_22;
                        float attn_1_23 = (float)(__nv_bfloat16)v[7];
                        float _exp2_138 = approx_exp2((-_vec_load_10[7]) * 1.442695f);
                        float sigmoid_2_23 = approx_rcp(1.0f + _exp2_138);
                        gated[7] = attn_1_23 * sigmoid_2_23;
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
                        reinterpret_cast<int4*>(out + out_off_1_1)[0] = reinterpret_cast<int4*>(_packed_2)[0];
                        float _vec_load_11[8];
                        {
                            const uint4* _vptr_139 = reinterpret_cast<const uint4*>(gate + (gate_off_1_1 + 8) + 0);
                            uint4 _vld_139[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_139[_blk] = _vptr_139[_blk];
                                uint32_t* _vpairs_139 = reinterpret_cast<uint32_t*>(&_vld_139[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_139[_pair]) << 16);
                                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_139[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float attn_1_24 = (float)(__nv_bfloat16)v[8];
                        float _exp2_140 = approx_exp2((-_vec_load_11[0]) * 1.442695f);
                        float sigmoid_2_24 = approx_rcp(1.0f + _exp2_140);
                        gated[0] = attn_1_24 * sigmoid_2_24;
                        float attn_1_25 = (float)(__nv_bfloat16)v[9];
                        float _exp2_141 = approx_exp2((-_vec_load_11[1]) * 1.442695f);
                        float sigmoid_2_25 = approx_rcp(1.0f + _exp2_141);
                        gated[1] = attn_1_25 * sigmoid_2_25;
                        float attn_1_26 = (float)(__nv_bfloat16)v[10];
                        float _exp2_142 = approx_exp2((-_vec_load_11[2]) * 1.442695f);
                        float sigmoid_2_26 = approx_rcp(1.0f + _exp2_142);
                        gated[2] = attn_1_26 * sigmoid_2_26;
                        float attn_1_27 = (float)(__nv_bfloat16)v[11];
                        float _exp2_143 = approx_exp2((-_vec_load_11[3]) * 1.442695f);
                        float sigmoid_2_27 = approx_rcp(1.0f + _exp2_143);
                        gated[3] = attn_1_27 * sigmoid_2_27;
                        float attn_1_28 = (float)(__nv_bfloat16)v[12];
                        float _exp2_144 = approx_exp2((-_vec_load_11[4]) * 1.442695f);
                        float sigmoid_2_28 = approx_rcp(1.0f + _exp2_144);
                        gated[4] = attn_1_28 * sigmoid_2_28;
                        float attn_1_29 = (float)(__nv_bfloat16)v[13];
                        float _exp2_145 = approx_exp2((-_vec_load_11[5]) * 1.442695f);
                        float sigmoid_2_29 = approx_rcp(1.0f + _exp2_145);
                        gated[5] = attn_1_29 * sigmoid_2_29;
                        float attn_1_30 = (float)(__nv_bfloat16)v[14];
                        float _exp2_146 = approx_exp2((-_vec_load_11[6]) * 1.442695f);
                        float sigmoid_2_30 = approx_rcp(1.0f + _exp2_146);
                        gated[6] = attn_1_30 * sigmoid_2_30;
                        float attn_1_31 = (float)(__nv_bfloat16)v[15];
                        float _exp2_147 = approx_exp2((-_vec_load_11[7]) * 1.442695f);
                        float sigmoid_2_31 = approx_rcp(1.0f + _exp2_147);
                        gated[7] = attn_1_31 * sigmoid_2_31;
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
                        reinterpret_cast<int4*>(out + (out_off_1_1 + 8))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                    }
                } else {
#if __CUDA_ARCH__ == 1070
                    long long p_base = ((long long)seg_4 * (long long)num_rows + row_i64_4) * (long long)num_qo_heads + (long long)first_head_4;
#else
                    long long p_base = ((long long)seg_5 * (long long)num_rows + row_i64_5) * (long long)num_qo_heads + (long long)first_head_5;
#endif
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
                            float _log2_148;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_148) : "f"(l_t));
                            *(reinterpret_cast<float*>(partial_lse + (p_base + (long long)tid_0)) + (0)) = ((l_t > 0.0f) ? m_t + _log2_148 : -CUDART_INF_F);
                        }
                    }
#if __CUDA_ARCH__ == 1070
                    if (last_4) {
#else
                    if (last_5) {
#endif
                        float zero = 0.0f;
                        float neg_inf = -CUDART_INF_F;
                        #pragma unroll 1
#if __CUDA_ARCH__ == 1070
                        for (int f = seg_4 + 1; f < num_splits; f++) {
                            long long f_base = ((long long)f * (long long)num_rows + row_i64_4) * (long long)num_qo_heads + (long long)first_head_4;
#else
                        for (int f = seg_5 + 1; f < num_splits; f++) {
                            long long f_base = ((long long)f * (long long)num_rows + row_i64_5) * (long long)num_qo_heads + (long long)first_head_5;
#endif
                            long long f_off = f_base * 256 + (long long)d0;
                            bool in_group_1 = group_size > 0;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + f_off)), "f"(zero), "r"((unsigned int)(in_group_1)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 128))), "f"(zero), "r"((unsigned int)(in_group_1)) : "memory");
                            bool in_group_2 = group_size > 1;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 256))), "f"(zero), "r"((unsigned int)(in_group_2)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 256 + 128))), "f"(zero), "r"((unsigned int)(in_group_2)) : "memory");
                            bool in_group_4 = group_size > 2;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 512))), "f"(zero), "r"((unsigned int)(in_group_4)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 512 + 128))), "f"(zero), "r"((unsigned int)(in_group_4)) : "memory");
                            bool in_group_5 = group_size > 3;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 768))), "f"(zero), "r"((unsigned int)(in_group_5)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 768 + 128))), "f"(zero), "r"((unsigned int)(in_group_5)) : "memory");
                            bool in_group_7 = group_size > 4;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1024))), "f"(zero), "r"((unsigned int)(in_group_7)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1024 + 128))), "f"(zero), "r"((unsigned int)(in_group_7)) : "memory");
                            bool in_group_8 = group_size > 5;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1280))), "f"(zero), "r"((unsigned int)(in_group_8)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1280 + 128))), "f"(zero), "r"((unsigned int)(in_group_8)) : "memory");
                            bool in_group_10 = group_size > 6;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1536))), "f"(zero), "r"((unsigned int)(in_group_10)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1536 + 128))), "f"(zero), "r"((unsigned int)(in_group_10)) : "memory");
                            bool in_group_11 = group_size > 7;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1792))), "f"(zero), "r"((unsigned int)(in_group_11)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 1792 + 128))), "f"(zero), "r"((unsigned int)(in_group_11)) : "memory");
                            bool in_group_13 = group_size > 8;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2048))), "f"(zero), "r"((unsigned int)(in_group_13)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2048 + 128))), "f"(zero), "r"((unsigned int)(in_group_13)) : "memory");
                            bool in_group_14 = group_size > 9;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2304))), "f"(zero), "r"((unsigned int)(in_group_14)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2304 + 128))), "f"(zero), "r"((unsigned int)(in_group_14)) : "memory");
                            bool in_group_16 = group_size > 10;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2560))), "f"(zero), "r"((unsigned int)(in_group_16)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2560 + 128))), "f"(zero), "r"((unsigned int)(in_group_16)) : "memory");
                            bool in_group_17 = group_size > 11;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2816))), "f"(zero), "r"((unsigned int)(in_group_17)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 2816 + 128))), "f"(zero), "r"((unsigned int)(in_group_17)) : "memory");
                            bool in_group_19 = group_size > 12;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3072))), "f"(zero), "r"((unsigned int)(in_group_19)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3072 + 128))), "f"(zero), "r"((unsigned int)(in_group_19)) : "memory");
                            bool in_group_20 = group_size > 13;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3328))), "f"(zero), "r"((unsigned int)(in_group_20)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3328 + 128))), "f"(zero), "r"((unsigned int)(in_group_20)) : "memory");
                            bool in_group_22 = group_size > 14;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3584))), "f"(zero), "r"((unsigned int)(in_group_22)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3584 + 128))), "f"(zero), "r"((unsigned int)(in_group_22)) : "memory");
                            bool in_group_23 = group_size > 15;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3840))), "f"(zero), "r"((unsigned int)(in_group_23)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(partial_out + (f_off + 3840 + 128))), "f"(zero), "r"((unsigned int)(in_group_23)) : "memory");
                            if (tid_0 < 16) {
                                if (tid_0 < group_size) {
                                    *(reinterpret_cast<float*>(partial_lse + (f_base + (long long)tid_0)) + (0)) = neg_inf;
                                }
                            }
                        }
                    }
                }
#if __CUDA_ARCH__ == 1070
                req_cur = request_8;
                vc_cur = valid_count_8;
#else
                req_cur = request_7;
                vc_cur = valid_count_7;
#endif
            }
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_dead_addr);
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
            int nt0_1 = 0;
            int req_cur_1 = request;
            int vc_cur_1 = valid_count;
            #pragma unroll 1
            for (int P_1 = pair0; P_1 < pair1 + 1; P_1++) {
#if __CUDA_ARCH__ == 1070
                int row_0 = P_1 / hkv;
                int kv_head_0 = P_1 - row_0 * hkv;
                long long row_i64_0 = (long long)row_0;
                int first_head_0 = kv_head_0 * group_size;
                long long indices_row_0 = row_i64_0 * indices_stride_row;
                long long pslot_0 = (long long)P_1 * tiles_i64;
                long long _max_0_3 = ((slot0 - pslot_0) > (0) ? (slot0 - pslot_0) : (0));
                int tile0_0 = (int)_max_0_3;
                long long _min_1_3 = ((slot1 - pslot_0) < (tiles_i64) ? (slot1 - pslot_0) : (tiles_i64));
                int tile1_0 = (int)_min_1_3;
                int owner0_0 = (int)(((pslot_0 + 1) * ctas_i64 - 1) / slots);
                int cta_1 = blockIdx.x;
                int seg_0 = cta_1 - owner0_0;
                bool last_0 = slot1 >= pslot_0 + tiles_i64;
                bool request_ok_0 = req_cur_1 >= 0 && req_cur_1 < num_requests;
#else
                int row_1 = P_1 / hkv;
                int kv_head_1 = P_1 - row_1 * hkv;
                long long row_i64_1 = (long long)row_1;
                int first_head_1 = kv_head_1 * group_size;
                long long indices_row_1 = row_i64_1 * indices_stride_row;
                long long pslot_1 = (long long)P_1 * tiles_i64;
                long long _max_0_3 = ((slot0 - pslot_1) > (0) ? (slot0 - pslot_1) : (0));
                int tile0_1 = (int)_max_0_3;
                long long _min_1_3 = ((slot1 - pslot_1) < (tiles_i64) ? (slot1 - pslot_1) : (tiles_i64));
                int tile1_1 = (int)_min_1_3;
                int owner0_1 = (int)(((pslot_1 + 1) * ctas_i64 - 1) / slots);
                int cta_2 = blockIdx.x;
                int seg_1 = cta_2 - owner0_1;
                bool last_1 = slot1 >= pslot_1 + tiles_i64;
                bool request_ok_1 = req_cur_1 >= 0 && req_cur_1 < num_requests;
#endif
                int _max_2_1 = ((req_cur_1) > (0) ? (req_cur_1) : (0));
#if __CUDA_ARCH__ == 1070
                int safe_request_0 = ((_max_2_1) < (num_requests - 1) ? (_max_2_1) : (num_requests - 1));
                long long safe_request_row_0 = (long long)safe_request_0 * table_stride_req;
#else
                int safe_request_1 = ((_max_2_1) < (num_requests - 1) ? (_max_2_1) : (num_requests - 1));
                long long safe_request_row_1 = (long long)safe_request_1 * table_stride_req;
#endif
                int _min_4_1 = ((vc_cur_1) < (topk) ? (vc_cur_1) : (topk));
                int _min_3_1 = ((num_tiles) < ((_min_4_1 + 63) / 64) ? (num_tiles) : ((_min_4_1 + 63) / 64));
#if __CUDA_ARCH__ == 1070
                int tile_end_0 = ((_min_3_1) < (tile1_0) ? (_min_3_1) : (tile1_0));
                int n_tiles_0 = ((tile_end_0 - tile0_0) > (0) ? (tile_end_0 - tile0_0) : (0));
#else
                int tile_end_1 = ((_min_3_1) < (tile1_1) ? (_min_3_1) : (tile1_1));
                int n_tiles_1 = ((tile_end_1 - tile0_1) > (0) ? (tile_end_1 - tile0_1) : (0));
#endif
                int nxt_1 = ((P_1 + 1) < (pair1) ? (P_1 + 1) : (pair1));
                int nrow_1 = nxt_1 / hkv;
                long long nrow_i64_1 = (long long)nrow_1;
                int nfirst_1 = (nxt_1 - nrow_1 * hkv) * group_size;
                long long nidx_row_1 = nrow_i64_1 * indices_stride_row;
#if __CUDA_ARCH__ == 1070
                int request_4 = token_to_req[nrow_1];
                int valid_count_4 = indices[nidx_row_1 + (long long)topk];
#else
                int request_3 = token_to_req[nrow_1];
                int valid_count_3 = indices[nidx_row_1 + (long long)topk];
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1070
                for (int n_1 = 0; n_1 < n_tiles_0; n_1++) {
#else
                for (int n_1 = 0; n_1 < n_tiles_1; n_1++) {
#endif
                    int nt_1 = nt0_1 + n_1;
                    int st = nt_1 % 4;
                    int par = nt_1 & 1;
                    int _stage_raw_full_k = (int)((unsigned int)(nt0_1 + n_1) % 4);
                    int _parity_raw_full_k = (int)((unsigned int)(nt0_1 + n_1) / 4 % 2);
                    mbarrier_wait(raw_full_k_addr + (_stage_raw_full_k) * 8, _parity_raw_full_k);
                    int _stage_k_empty = (int)((unsigned int)(nt0_1 + n_1) % 2);
                    int _parity_k_empty = (int)((unsigned int)(nt0_1 + n_1) / 2 % 2) ^ 1;
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
                    uint32_t prmt;
                    asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt) : "r"(sw[0]), "r"(sw[0]));
                    uint32_t _bf16x2_fma_0;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_0) : "r"(prmt & 2147516416u | (prmt & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfa[0] = _bf16x2_fma_0;
                    uint32_t _bf16x2_fma_1;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_1) : "r"(prmt << 8 & 2147516416u | (prmt << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfa[1] = _bf16x2_fma_1;
                    uint32_t prmt_0;
                    asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_0) : "r"(sw[1]), "r"(sw[1]));
                    uint32_t _bf16x2_fma_2;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_2) : "r"(prmt_0 & 2147516416u | (prmt_0 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfb[0] = _bf16x2_fma_2;
                    uint32_t _bf16x2_fma_3;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_3) : "r"(prmt_0 << 8 & 2147516416u | (prmt_0 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfb[1] = _bf16x2_fma_3;
                    unsigned int sc[8];
                    uint32_t _prmt_5;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_5) : "r"(sfa[0]), "r"(0));
                    sc[0] = _prmt_5;
                    uint32_t _prmt_6;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_6) : "r"(sfb[0]), "r"(0));
                    sc[4] = _prmt_6;
                    uint32_t _prmt_7;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_7) : "r"(sfa[0]), "r"(0));
                    sc[1] = _prmt_7;
                    uint32_t _prmt_8;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_8) : "r"(sfb[0]), "r"(0));
                    sc[5] = _prmt_8;
                    uint32_t _prmt_9;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_9) : "r"(sfa[1]), "r"(0));
                    sc[2] = _prmt_9;
                    uint32_t _prmt_10;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_10) : "r"(sfb[1]), "r"(0));
                    sc[6] = _prmt_10;
                    uint32_t _prmt_11;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_11) : "r"(sfa[1]), "r"(0));
                    sc[3] = _prmt_11;
                    uint32_t _prmt_12;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_12) : "r"(sfb[1]), "r"(0));
                    sc[7] = _prmt_12;
                    unsigned int kdst = k_lane0 + (unsigned int)(16 * par << 16);
                    unsigned int w16_1_0[16];
                    unsigned int d_3_0[4];
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
                    d_3_0[0] = _fp4_bf16x8_0[0];
                    d_3_0[1] = _fp4_bf16x8_0[1];
                    d_3_0[2] = _fp4_bf16x8_0[2];
                    d_3_0[3] = _fp4_bf16x8_0[3];
                    uint32_t _bf16x2_mul_0;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(d_3_0[0]), "r"(sc[0]));
                    w16_1_0[0] = _bf16x2_mul_0;
                    uint32_t _bf16x2_mul_1;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(d_3_0[1]), "r"(sc[0]));
                    w16_1_0[1] = _bf16x2_mul_1;
                    uint32_t _bf16x2_mul_2;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(d_3_0[2]), "r"(sc[0]));
                    w16_1_0[2] = _bf16x2_mul_2;
                    uint32_t _bf16x2_mul_3;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(d_3_0[3]), "r"(sc[0]));
                    w16_1_0[3] = _bf16x2_mul_3;
                    unsigned int d_3_1[4];
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
                    d_3_1[0] = _fp4_bf16x8_1[0];
                    d_3_1[1] = _fp4_bf16x8_1[1];
                    d_3_1[2] = _fp4_bf16x8_1[2];
                    d_3_1[3] = _fp4_bf16x8_1[3];
                    uint32_t _bf16x2_mul_4;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(d_3_1[0]), "r"(sc[0]));
                    w16_1_0[4] = _bf16x2_mul_4;
                    uint32_t _bf16x2_mul_5;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(d_3_1[1]), "r"(sc[0]));
                    w16_1_0[5] = _bf16x2_mul_5;
                    uint32_t _bf16x2_mul_6;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(d_3_1[2]), "r"(sc[0]));
                    w16_1_0[6] = _bf16x2_mul_6;
                    uint32_t _bf16x2_mul_7;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(d_3_1[3]), "r"(sc[0]));
                    w16_1_0[7] = _bf16x2_mul_7;
                    unsigned int d_3_2[4];
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
                    d_3_2[0] = _fp4_bf16x8_2[0];
                    d_3_2[1] = _fp4_bf16x8_2[1];
                    d_3_2[2] = _fp4_bf16x8_2[2];
                    d_3_2[3] = _fp4_bf16x8_2[3];
                    uint32_t _bf16x2_mul_8;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(d_3_2[0]), "r"(sc[1]));
                    w16_1_0[8] = _bf16x2_mul_8;
                    uint32_t _bf16x2_mul_9;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(d_3_2[1]), "r"(sc[1]));
                    w16_1_0[9] = _bf16x2_mul_9;
                    uint32_t _bf16x2_mul_10;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(d_3_2[2]), "r"(sc[1]));
                    w16_1_0[10] = _bf16x2_mul_10;
                    uint32_t _bf16x2_mul_11;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(d_3_2[3]), "r"(sc[1]));
                    w16_1_0[11] = _bf16x2_mul_11;
                    unsigned int d_3_3[4];
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
                    d_3_3[0] = _fp4_bf16x8_3[0];
                    d_3_3[1] = _fp4_bf16x8_3[1];
                    d_3_3[2] = _fp4_bf16x8_3[2];
                    d_3_3[3] = _fp4_bf16x8_3[3];
                    uint32_t _bf16x2_mul_12;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(d_3_3[0]), "r"(sc[1]));
                    w16_1_0[12] = _bf16x2_mul_12;
                    uint32_t _bf16x2_mul_13;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(d_3_3[1]), "r"(sc[1]));
                    w16_1_0[13] = _bf16x2_mul_13;
                    uint32_t _bf16x2_mul_14;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(d_3_3[2]), "r"(sc[1]));
                    w16_1_0[14] = _bf16x2_mul_14;
                    uint32_t _bf16x2_mul_15;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(d_3_3[3]), "r"(sc[1]));
                    w16_1_0[15] = _bf16x2_mul_15;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(kdst), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_0[15])));
                    unsigned int w16_1_1[16];
                    unsigned int d_3_4[4];
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
                    d_3_4[0] = _fp4_bf16x8_4[0];
                    d_3_4[1] = _fp4_bf16x8_4[1];
                    d_3_4[2] = _fp4_bf16x8_4[2];
                    d_3_4[3] = _fp4_bf16x8_4[3];
                    uint32_t _bf16x2_mul_16;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(d_3_4[0]), "r"(sc[2]));
                    w16_1_1[0] = _bf16x2_mul_16;
                    uint32_t _bf16x2_mul_17;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(d_3_4[1]), "r"(sc[2]));
                    w16_1_1[1] = _bf16x2_mul_17;
                    uint32_t _bf16x2_mul_18;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(d_3_4[2]), "r"(sc[2]));
                    w16_1_1[2] = _bf16x2_mul_18;
                    uint32_t _bf16x2_mul_19;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(d_3_4[3]), "r"(sc[2]));
                    w16_1_1[3] = _bf16x2_mul_19;
                    unsigned int d_3_5[4];
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
                    d_3_5[0] = _fp4_bf16x8_5[0];
                    d_3_5[1] = _fp4_bf16x8_5[1];
                    d_3_5[2] = _fp4_bf16x8_5[2];
                    d_3_5[3] = _fp4_bf16x8_5[3];
                    uint32_t _bf16x2_mul_20;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(d_3_5[0]), "r"(sc[2]));
                    w16_1_1[4] = _bf16x2_mul_20;
                    uint32_t _bf16x2_mul_21;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(d_3_5[1]), "r"(sc[2]));
                    w16_1_1[5] = _bf16x2_mul_21;
                    uint32_t _bf16x2_mul_22;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(d_3_5[2]), "r"(sc[2]));
                    w16_1_1[6] = _bf16x2_mul_22;
                    uint32_t _bf16x2_mul_23;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(d_3_5[3]), "r"(sc[2]));
                    w16_1_1[7] = _bf16x2_mul_23;
                    unsigned int d_3_6[4];
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
                    d_3_6[0] = _fp4_bf16x8_6[0];
                    d_3_6[1] = _fp4_bf16x8_6[1];
                    d_3_6[2] = _fp4_bf16x8_6[2];
                    d_3_6[3] = _fp4_bf16x8_6[3];
                    uint32_t _bf16x2_mul_24;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(d_3_6[0]), "r"(sc[3]));
                    w16_1_1[8] = _bf16x2_mul_24;
                    uint32_t _bf16x2_mul_25;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(d_3_6[1]), "r"(sc[3]));
                    w16_1_1[9] = _bf16x2_mul_25;
                    uint32_t _bf16x2_mul_26;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(d_3_6[2]), "r"(sc[3]));
                    w16_1_1[10] = _bf16x2_mul_26;
                    uint32_t _bf16x2_mul_27;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(d_3_6[3]), "r"(sc[3]));
                    w16_1_1[11] = _bf16x2_mul_27;
                    unsigned int d_3_7[4];
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
                    d_3_7[0] = _fp4_bf16x8_7[0];
                    d_3_7[1] = _fp4_bf16x8_7[1];
                    d_3_7[2] = _fp4_bf16x8_7[2];
                    d_3_7[3] = _fp4_bf16x8_7[3];
                    uint32_t _bf16x2_mul_28;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(d_3_7[0]), "r"(sc[3]));
                    w16_1_1[12] = _bf16x2_mul_28;
                    uint32_t _bf16x2_mul_29;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(d_3_7[1]), "r"(sc[3]));
                    w16_1_1[13] = _bf16x2_mul_29;
                    uint32_t _bf16x2_mul_30;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(d_3_7[2]), "r"(sc[3]));
                    w16_1_1[14] = _bf16x2_mul_30;
                    uint32_t _bf16x2_mul_31;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(d_3_7[3]), "r"(sc[3]));
                    w16_1_1[15] = _bf16x2_mul_31;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(kdst + 16), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_1[15])));
                    unsigned int w16_1_2[16];
                    unsigned int d_3_8[4];
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
                    d_3_8[0] = _fp4_bf16x8_8[0];
                    d_3_8[1] = _fp4_bf16x8_8[1];
                    d_3_8[2] = _fp4_bf16x8_8[2];
                    d_3_8[3] = _fp4_bf16x8_8[3];
                    uint32_t _bf16x2_mul_32;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(d_3_8[0]), "r"(sc[4]));
                    w16_1_2[0] = _bf16x2_mul_32;
                    uint32_t _bf16x2_mul_33;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(d_3_8[1]), "r"(sc[4]));
                    w16_1_2[1] = _bf16x2_mul_33;
                    uint32_t _bf16x2_mul_34;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(d_3_8[2]), "r"(sc[4]));
                    w16_1_2[2] = _bf16x2_mul_34;
                    uint32_t _bf16x2_mul_35;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(d_3_8[3]), "r"(sc[4]));
                    w16_1_2[3] = _bf16x2_mul_35;
                    unsigned int d_3_9[4];
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
                    d_3_9[0] = _fp4_bf16x8_9[0];
                    d_3_9[1] = _fp4_bf16x8_9[1];
                    d_3_9[2] = _fp4_bf16x8_9[2];
                    d_3_9[3] = _fp4_bf16x8_9[3];
                    uint32_t _bf16x2_mul_36;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(d_3_9[0]), "r"(sc[4]));
                    w16_1_2[4] = _bf16x2_mul_36;
                    uint32_t _bf16x2_mul_37;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(d_3_9[1]), "r"(sc[4]));
                    w16_1_2[5] = _bf16x2_mul_37;
                    uint32_t _bf16x2_mul_38;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(d_3_9[2]), "r"(sc[4]));
                    w16_1_2[6] = _bf16x2_mul_38;
                    uint32_t _bf16x2_mul_39;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(d_3_9[3]), "r"(sc[4]));
                    w16_1_2[7] = _bf16x2_mul_39;
                    unsigned int d_3_10[4];
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
                    d_3_10[0] = _fp4_bf16x8_10[0];
                    d_3_10[1] = _fp4_bf16x8_10[1];
                    d_3_10[2] = _fp4_bf16x8_10[2];
                    d_3_10[3] = _fp4_bf16x8_10[3];
                    uint32_t _bf16x2_mul_40;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(d_3_10[0]), "r"(sc[5]));
                    w16_1_2[8] = _bf16x2_mul_40;
                    uint32_t _bf16x2_mul_41;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(d_3_10[1]), "r"(sc[5]));
                    w16_1_2[9] = _bf16x2_mul_41;
                    uint32_t _bf16x2_mul_42;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(d_3_10[2]), "r"(sc[5]));
                    w16_1_2[10] = _bf16x2_mul_42;
                    uint32_t _bf16x2_mul_43;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(d_3_10[3]), "r"(sc[5]));
                    w16_1_2[11] = _bf16x2_mul_43;
                    unsigned int d_3_11[4];
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
                    d_3_11[0] = _fp4_bf16x8_11[0];
                    d_3_11[1] = _fp4_bf16x8_11[1];
                    d_3_11[2] = _fp4_bf16x8_11[2];
                    d_3_11[3] = _fp4_bf16x8_11[3];
                    uint32_t _bf16x2_mul_44;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(d_3_11[0]), "r"(sc[5]));
                    w16_1_2[12] = _bf16x2_mul_44;
                    uint32_t _bf16x2_mul_45;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(d_3_11[1]), "r"(sc[5]));
                    w16_1_2[13] = _bf16x2_mul_45;
                    uint32_t _bf16x2_mul_46;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(d_3_11[2]), "r"(sc[5]));
                    w16_1_2[14] = _bf16x2_mul_46;
                    uint32_t _bf16x2_mul_47;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(d_3_11[3]), "r"(sc[5]));
                    w16_1_2[15] = _bf16x2_mul_47;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(kdst + 32), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_2[15])));
                    unsigned int w16_1_3[16];
                    unsigned int d_3_12[4];
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
                    d_3_12[0] = _fp4_bf16x8_12[0];
                    d_3_12[1] = _fp4_bf16x8_12[1];
                    d_3_12[2] = _fp4_bf16x8_12[2];
                    d_3_12[3] = _fp4_bf16x8_12[3];
                    uint32_t _bf16x2_mul_48;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(d_3_12[0]), "r"(sc[6]));
                    w16_1_3[0] = _bf16x2_mul_48;
                    uint32_t _bf16x2_mul_49;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(d_3_12[1]), "r"(sc[6]));
                    w16_1_3[1] = _bf16x2_mul_49;
                    uint32_t _bf16x2_mul_50;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(d_3_12[2]), "r"(sc[6]));
                    w16_1_3[2] = _bf16x2_mul_50;
                    uint32_t _bf16x2_mul_51;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(d_3_12[3]), "r"(sc[6]));
                    w16_1_3[3] = _bf16x2_mul_51;
                    unsigned int d_3_13[4];
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
                    d_3_13[0] = _fp4_bf16x8_13[0];
                    d_3_13[1] = _fp4_bf16x8_13[1];
                    d_3_13[2] = _fp4_bf16x8_13[2];
                    d_3_13[3] = _fp4_bf16x8_13[3];
                    uint32_t _bf16x2_mul_52;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(d_3_13[0]), "r"(sc[6]));
                    w16_1_3[4] = _bf16x2_mul_52;
                    uint32_t _bf16x2_mul_53;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(d_3_13[1]), "r"(sc[6]));
                    w16_1_3[5] = _bf16x2_mul_53;
                    uint32_t _bf16x2_mul_54;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(d_3_13[2]), "r"(sc[6]));
                    w16_1_3[6] = _bf16x2_mul_54;
                    uint32_t _bf16x2_mul_55;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(d_3_13[3]), "r"(sc[6]));
                    w16_1_3[7] = _bf16x2_mul_55;
                    unsigned int d_3_14[4];
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
                    d_3_14[0] = _fp4_bf16x8_14[0];
                    d_3_14[1] = _fp4_bf16x8_14[1];
                    d_3_14[2] = _fp4_bf16x8_14[2];
                    d_3_14[3] = _fp4_bf16x8_14[3];
                    uint32_t _bf16x2_mul_56;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(d_3_14[0]), "r"(sc[7]));
                    w16_1_3[8] = _bf16x2_mul_56;
                    uint32_t _bf16x2_mul_57;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(d_3_14[1]), "r"(sc[7]));
                    w16_1_3[9] = _bf16x2_mul_57;
                    uint32_t _bf16x2_mul_58;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(d_3_14[2]), "r"(sc[7]));
                    w16_1_3[10] = _bf16x2_mul_58;
                    uint32_t _bf16x2_mul_59;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(d_3_14[3]), "r"(sc[7]));
                    w16_1_3[11] = _bf16x2_mul_59;
                    unsigned int d_3_15[4];
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
                    d_3_15[0] = _fp4_bf16x8_15[0];
                    d_3_15[1] = _fp4_bf16x8_15[1];
                    d_3_15[2] = _fp4_bf16x8_15[2];
                    d_3_15[3] = _fp4_bf16x8_15[3];
                    uint32_t _bf16x2_mul_60;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(d_3_15[0]), "r"(sc[7]));
                    w16_1_3[12] = _bf16x2_mul_60;
                    uint32_t _bf16x2_mul_61;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(d_3_15[1]), "r"(sc[7]));
                    w16_1_3[13] = _bf16x2_mul_61;
                    uint32_t _bf16x2_mul_62;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(d_3_15[2]), "r"(sc[7]));
                    w16_1_3[14] = _bf16x2_mul_62;
                    uint32_t _bf16x2_mul_63;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(d_3_15[3]), "r"(sc[7]));
                    w16_1_3[15] = _bf16x2_mul_63;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(kdst + 48), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&w16_1_3[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(k_full_addr + ((int)((unsigned int)(nt0_1 + n_1) % 2)) * 8);
                        mbarrier_arrive(raw_empty_addr + ((int)((unsigned int)(nt0_1 + n_1) % 4)) * 8);
                    }
                }
#if __CUDA_ARCH__ == 1070
                nt0_1 += n_tiles_0;
                req_cur_1 = request_4;
                vc_cur_1 = valid_count_4;
#else
                nt0_1 += n_tiles_1;
                req_cur_1 = request_3;
                vc_cur_1 = valid_count_3;
#endif
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
            int nt0_2 = 0;
            int req_cur_2 = request;
            int vc_cur_2 = valid_count;
            #pragma unroll 1
            for (int P_2 = pair0; P_2 < pair1 + 1; P_2++) {
#if __CUDA_ARCH__ == 1070
                int row_1 = P_2 / hkv;
                int kv_head_1 = P_2 - row_1 * hkv;
                long long row_i64_1 = (long long)row_1;
                int first_head_1 = kv_head_1 * group_size;
                long long indices_row_1 = row_i64_1 * indices_stride_row;
                long long pslot_1 = (long long)P_2 * tiles_i64;
                long long _max_0_4 = ((slot0 - pslot_1) > (0) ? (slot0 - pslot_1) : (0));
                int tile0_1 = (int)_max_0_4;
                long long _min_1_4 = ((slot1 - pslot_1) < (tiles_i64) ? (slot1 - pslot_1) : (tiles_i64));
                int tile1_1 = (int)_min_1_4;
                int owner0_1 = (int)(((pslot_1 + 1) * ctas_i64 - 1) / slots);
                int cta_2 = blockIdx.x;
                int seg_1 = cta_2 - owner0_1;
                bool last_1 = slot1 >= pslot_1 + tiles_i64;
                bool request_ok_1 = req_cur_2 >= 0 && req_cur_2 < num_requests;
#else
                int row_2 = P_2 / hkv;
                int kv_head_2 = P_2 - row_2 * hkv;
                long long row_i64_2 = (long long)row_2;
                int first_head_2 = kv_head_2 * group_size;
                long long indices_row_2 = row_i64_2 * indices_stride_row;
                long long pslot_2 = (long long)P_2 * tiles_i64;
                long long _max_0_4 = ((slot0 - pslot_2) > (0) ? (slot0 - pslot_2) : (0));
                int tile0_2 = (int)_max_0_4;
                long long _min_1_4 = ((slot1 - pslot_2) < (tiles_i64) ? (slot1 - pslot_2) : (tiles_i64));
                int tile1_2 = (int)_min_1_4;
                int owner0_2 = (int)(((pslot_2 + 1) * ctas_i64 - 1) / slots);
                int cta_3 = blockIdx.x;
                int seg_2 = cta_3 - owner0_2;
                bool last_2 = slot1 >= pslot_2 + tiles_i64;
                bool request_ok_2 = req_cur_2 >= 0 && req_cur_2 < num_requests;
#endif
                int _max_2_2 = ((req_cur_2) > (0) ? (req_cur_2) : (0));
#if __CUDA_ARCH__ == 1070
                int safe_request_1 = ((_max_2_2) < (num_requests - 1) ? (_max_2_2) : (num_requests - 1));
                long long safe_request_row_1 = (long long)safe_request_1 * table_stride_req;
#else
                int safe_request_2 = ((_max_2_2) < (num_requests - 1) ? (_max_2_2) : (num_requests - 1));
                long long safe_request_row_2 = (long long)safe_request_2 * table_stride_req;
#endif
                int _min_4_2 = ((vc_cur_2) < (topk) ? (vc_cur_2) : (topk));
                int _min_3_2 = ((num_tiles) < ((_min_4_2 + 63) / 64) ? (num_tiles) : ((_min_4_2 + 63) / 64));
#if __CUDA_ARCH__ == 1070
                int tile_end_1 = ((_min_3_2) < (tile1_1) ? (_min_3_2) : (tile1_1));
                int n_tiles_1 = ((tile_end_1 - tile0_1) > (0) ? (tile_end_1 - tile0_1) : (0));
#else
                int tile_end_2 = ((_min_3_2) < (tile1_2) ? (_min_3_2) : (tile1_2));
                int n_tiles_2 = ((tile_end_2 - tile0_2) > (0) ? (tile_end_2 - tile0_2) : (0));
#endif
                int nxt_2 = ((P_2 + 1) < (pair1) ? (P_2 + 1) : (pair1));
                int nrow_2 = nxt_2 / hkv;
                long long nrow_i64_2 = (long long)nrow_2;
                int nfirst_2 = (nxt_2 - nrow_2 * hkv) * group_size;
                long long nidx_row_2 = nrow_i64_2 * indices_stride_row;
#if __CUDA_ARCH__ == 1070
                int request_5 = token_to_req[nrow_2];
                int valid_count_5 = indices[nidx_row_2 + (long long)topk];
#else
                int request_4 = token_to_req[nrow_2];
                int valid_count_4 = indices[nidx_row_2 + (long long)topk];
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1070
                for (int n_2 = 0; n_2 < n_tiles_1; n_2++) {
#else
                for (int n_2 = 0; n_2 < n_tiles_2; n_2++) {
#endif
                    int nt_2 = nt0_2 + n_2;
                    int st_1 = nt_2 % 4;
                    int vs = nt_2 % 2;
                    int _stage_raw_full_v = (int)((unsigned int)(nt0_2 + n_2) % 4);
                    int _parity_raw_full_v = (int)((unsigned int)(nt0_2 + n_2) / 4 % 2);
                    mbarrier_wait(raw_full_v_addr + (_stage_raw_full_v) * 8, _parity_raw_full_v);
                    int _stage_v_empty = (int)((unsigned int)(nt0_2 + n_2) % 2);
                    int _parity_v_empty = (int)((unsigned int)(nt0_2 + n_2) / 2 % 2) ^ 1;
                    mbarrier_wait(v_empty_addr + (_stage_v_empty) * 8, _parity_v_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    unsigned int row_addr_1 = raw_addr_1 + (unsigned int)(st_1 * 18432 + 9216 + tv * 144);
                    unsigned int raw_4[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(0) + 3]))
                        : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(4) + 3]))
                        : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1 + 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(8) + 3]))
                        : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1 + 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(12) + 3]))
                        : "r"(row_addr_1 + (unsigned int)(16 * (4 * hh_1 + 3))));
                    unsigned int sw_1[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 1]))
                        : "r"(row_addr_1 + (unsigned int)(128 + 8 * hh_1)));
                    unsigned int sfa_1[2];
                    unsigned int sfb_1[2];
                    uint32_t prmt_1;
                    asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_1) : "r"(sw_1[0]), "r"(sw_1[0]));
                    uint32_t _bf16x2_fma_4;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_4) : "r"(prmt_1 & 2147516416u | (prmt_1 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfa_1[0] = _bf16x2_fma_4;
                    uint32_t _bf16x2_fma_5;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_5) : "r"(prmt_1 << 8 & 2147516416u | (prmt_1 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfa_1[1] = _bf16x2_fma_5;
                    uint32_t prmt_1_1;
                    asm("prmt.b32 %0, %1, %2, 0x1302;" : "=r"(prmt_1_1) : "r"(sw_1[1]), "r"(sw_1[1]));
                    uint32_t _bf16x2_fma_6;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_6) : "r"(prmt_1_1 & 2147516416u | (prmt_1_1 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfb_1[0] = _bf16x2_fma_6;
                    uint32_t _bf16x2_fma_7;
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_7) : "r"(prmt_1_1 << 8 & 2147516416u | (prmt_1_1 << 8 & 2130738944) >> 4), "r"(2072017792), "r"(2147516416u));
                    sfb_1[1] = _bf16x2_fma_7;
                    unsigned int sc_1[8];
                    uint32_t _prmt_5_1;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_5_1) : "r"(sfa_1[0]), "r"(0));
                    sc_1[0] = _prmt_5_1;
                    uint32_t _prmt_6_1;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_6_1) : "r"(sfb_1[0]), "r"(0));
                    sc_1[4] = _prmt_6_1;
                    uint32_t _prmt_7_1;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_7_1) : "r"(sfa_1[0]), "r"(0));
                    sc_1[1] = _prmt_7_1;
                    uint32_t _prmt_8_1;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_8_1) : "r"(sfb_1[0]), "r"(0));
                    sc_1[5] = _prmt_8_1;
                    uint32_t _prmt_9_1;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_9_1) : "r"(sfa_1[1]), "r"(0));
                    sc_1[2] = _prmt_9_1;
                    uint32_t _prmt_10_1;
                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_10_1) : "r"(sfb_1[1]), "r"(0));
                    sc_1[6] = _prmt_10_1;
                    uint32_t _prmt_11_1;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_11_1) : "r"(sfa_1[1]), "r"(0));
                    sc_1[3] = _prmt_11_1;
                    uint32_t _prmt_12_1;
                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_12_1) : "r"(sfb_1[1]), "r"(0));
                    sc_1[7] = _prmt_12_1;
                    unsigned int vbase = smem_vt_addr + (unsigned int)((2 * vs + hh_1) * 16384) + (unsigned int)(128 * tv);
                    unsigned int plane_1_0 = vbase;
                    unsigned int d_1_0[4];
                    uint32_t _fp4_bf16x8_16[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw_4[0])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw_4[0])));
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
                    d_1_0[0] = _fp4_bf16x8_16[0];
                    d_1_0[1] = _fp4_bf16x8_16[1];
                    d_1_0[2] = _fp4_bf16x8_16[2];
                    d_1_0[3] = _fp4_bf16x8_16[3];
                    unsigned int w4_2_0[4];
                    uint32_t _bf16x2_mul_64;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(d_1_0[0]), "r"(sc_1[0]));
                    w4_2_0[0] = _bf16x2_mul_64;
                    uint32_t _bf16x2_mul_65;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(d_1_0[1]), "r"(sc_1[0]));
                    w4_2_0[1] = _bf16x2_mul_65;
                    uint32_t _bf16x2_mul_66;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(d_1_0[2]), "r"(sc_1[0]));
                    w4_2_0[2] = _bf16x2_mul_66;
                    uint32_t _bf16x2_mul_67;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(d_1_0[3]), "r"(sc_1[0]));
                    w4_2_0[3] = _bf16x2_mul_67;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_0 + (unsigned int)((0 ^ tm7) << 4)), "r"(w4_2_0[0]), "r"(w4_2_0[1]), "r"(w4_2_0[2]), "r"(w4_2_0[3]));
                    unsigned int d_1_1[4];
                    uint32_t _fp4_bf16x8_17[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(raw_4[1])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(raw_4[1])));
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
                    d_1_1[0] = _fp4_bf16x8_17[0];
                    d_1_1[1] = _fp4_bf16x8_17[1];
                    d_1_1[2] = _fp4_bf16x8_17[2];
                    d_1_1[3] = _fp4_bf16x8_17[3];
                    unsigned int w4_2_1[4];
                    uint32_t _bf16x2_mul_68;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(d_1_1[0]), "r"(sc_1[0]));
                    w4_2_1[0] = _bf16x2_mul_68;
                    uint32_t _bf16x2_mul_69;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(d_1_1[1]), "r"(sc_1[0]));
                    w4_2_1[1] = _bf16x2_mul_69;
                    uint32_t _bf16x2_mul_70;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(d_1_1[2]), "r"(sc_1[0]));
                    w4_2_1[2] = _bf16x2_mul_70;
                    uint32_t _bf16x2_mul_71;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(d_1_1[3]), "r"(sc_1[0]));
                    w4_2_1[3] = _bf16x2_mul_71;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_0 + (unsigned int)((1 ^ tm7) << 4)), "r"(w4_2_1[0]), "r"(w4_2_1[1]), "r"(w4_2_1[2]), "r"(w4_2_1[3]));
                    unsigned int d_1_2[4];
                    uint32_t _fp4_bf16x8_18[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(raw_4[2])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(raw_4[2])));
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
                    d_1_2[0] = _fp4_bf16x8_18[0];
                    d_1_2[1] = _fp4_bf16x8_18[1];
                    d_1_2[2] = _fp4_bf16x8_18[2];
                    d_1_2[3] = _fp4_bf16x8_18[3];
                    unsigned int w4_2_2[4];
                    uint32_t _bf16x2_mul_72;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(d_1_2[0]), "r"(sc_1[1]));
                    w4_2_2[0] = _bf16x2_mul_72;
                    uint32_t _bf16x2_mul_73;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(d_1_2[1]), "r"(sc_1[1]));
                    w4_2_2[1] = _bf16x2_mul_73;
                    uint32_t _bf16x2_mul_74;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(d_1_2[2]), "r"(sc_1[1]));
                    w4_2_2[2] = _bf16x2_mul_74;
                    uint32_t _bf16x2_mul_75;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(d_1_2[3]), "r"(sc_1[1]));
                    w4_2_2[3] = _bf16x2_mul_75;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_0 + (unsigned int)((2 ^ tm7) << 4)), "r"(w4_2_2[0]), "r"(w4_2_2[1]), "r"(w4_2_2[2]), "r"(w4_2_2[3]));
                    unsigned int d_1_3[4];
                    uint32_t _fp4_bf16x8_19[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(raw_4[3])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(raw_4[3])));
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
                    d_1_3[0] = _fp4_bf16x8_19[0];
                    d_1_3[1] = _fp4_bf16x8_19[1];
                    d_1_3[2] = _fp4_bf16x8_19[2];
                    d_1_3[3] = _fp4_bf16x8_19[3];
                    unsigned int w4_2_3[4];
                    uint32_t _bf16x2_mul_76;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(d_1_3[0]), "r"(sc_1[1]));
                    w4_2_3[0] = _bf16x2_mul_76;
                    uint32_t _bf16x2_mul_77;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(d_1_3[1]), "r"(sc_1[1]));
                    w4_2_3[1] = _bf16x2_mul_77;
                    uint32_t _bf16x2_mul_78;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(d_1_3[2]), "r"(sc_1[1]));
                    w4_2_3[2] = _bf16x2_mul_78;
                    uint32_t _bf16x2_mul_79;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(d_1_3[3]), "r"(sc_1[1]));
                    w4_2_3[3] = _bf16x2_mul_79;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_0 + (unsigned int)((3 ^ tm7) << 4)), "r"(w4_2_3[0]), "r"(w4_2_3[1]), "r"(w4_2_3[2]), "r"(w4_2_3[3]));
                    unsigned int plane_1_1 = vbase;
                    unsigned int d_1_4[4];
                    uint32_t _fp4_bf16x8_20[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(raw_4[4])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(raw_4[4])));
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
                    d_1_4[0] = _fp4_bf16x8_20[0];
                    d_1_4[1] = _fp4_bf16x8_20[1];
                    d_1_4[2] = _fp4_bf16x8_20[2];
                    d_1_4[3] = _fp4_bf16x8_20[3];
                    unsigned int w4_2_4[4];
                    uint32_t _bf16x2_mul_80;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(d_1_4[0]), "r"(sc_1[2]));
                    w4_2_4[0] = _bf16x2_mul_80;
                    uint32_t _bf16x2_mul_81;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(d_1_4[1]), "r"(sc_1[2]));
                    w4_2_4[1] = _bf16x2_mul_81;
                    uint32_t _bf16x2_mul_82;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(d_1_4[2]), "r"(sc_1[2]));
                    w4_2_4[2] = _bf16x2_mul_82;
                    uint32_t _bf16x2_mul_83;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(d_1_4[3]), "r"(sc_1[2]));
                    w4_2_4[3] = _bf16x2_mul_83;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_1 + (unsigned int)((4 ^ tm7) << 4)), "r"(w4_2_4[0]), "r"(w4_2_4[1]), "r"(w4_2_4[2]), "r"(w4_2_4[3]));
                    unsigned int d_1_5[4];
                    uint32_t _fp4_bf16x8_21[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(raw_4[5])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(raw_4[5])));
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
                    d_1_5[0] = _fp4_bf16x8_21[0];
                    d_1_5[1] = _fp4_bf16x8_21[1];
                    d_1_5[2] = _fp4_bf16x8_21[2];
                    d_1_5[3] = _fp4_bf16x8_21[3];
                    unsigned int w4_2_5[4];
                    uint32_t _bf16x2_mul_84;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(d_1_5[0]), "r"(sc_1[2]));
                    w4_2_5[0] = _bf16x2_mul_84;
                    uint32_t _bf16x2_mul_85;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(d_1_5[1]), "r"(sc_1[2]));
                    w4_2_5[1] = _bf16x2_mul_85;
                    uint32_t _bf16x2_mul_86;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_86) : "r"(d_1_5[2]), "r"(sc_1[2]));
                    w4_2_5[2] = _bf16x2_mul_86;
                    uint32_t _bf16x2_mul_87;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_87) : "r"(d_1_5[3]), "r"(sc_1[2]));
                    w4_2_5[3] = _bf16x2_mul_87;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_1 + (unsigned int)((5 ^ tm7) << 4)), "r"(w4_2_5[0]), "r"(w4_2_5[1]), "r"(w4_2_5[2]), "r"(w4_2_5[3]));
                    unsigned int d_1_6[4];
                    uint32_t _fp4_bf16x8_22[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(raw_4[6])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(raw_4[6])));
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
                    d_1_6[0] = _fp4_bf16x8_22[0];
                    d_1_6[1] = _fp4_bf16x8_22[1];
                    d_1_6[2] = _fp4_bf16x8_22[2];
                    d_1_6[3] = _fp4_bf16x8_22[3];
                    unsigned int w4_2_6[4];
                    uint32_t _bf16x2_mul_88;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_88) : "r"(d_1_6[0]), "r"(sc_1[3]));
                    w4_2_6[0] = _bf16x2_mul_88;
                    uint32_t _bf16x2_mul_89;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_89) : "r"(d_1_6[1]), "r"(sc_1[3]));
                    w4_2_6[1] = _bf16x2_mul_89;
                    uint32_t _bf16x2_mul_90;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_90) : "r"(d_1_6[2]), "r"(sc_1[3]));
                    w4_2_6[2] = _bf16x2_mul_90;
                    uint32_t _bf16x2_mul_91;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_91) : "r"(d_1_6[3]), "r"(sc_1[3]));
                    w4_2_6[3] = _bf16x2_mul_91;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_1 + (unsigned int)((6 ^ tm7) << 4)), "r"(w4_2_6[0]), "r"(w4_2_6[1]), "r"(w4_2_6[2]), "r"(w4_2_6[3]));
                    unsigned int d_1_7[4];
                    uint32_t _fp4_bf16x8_23[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(raw_4[7])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(raw_4[7])));
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
                    d_1_7[0] = _fp4_bf16x8_23[0];
                    d_1_7[1] = _fp4_bf16x8_23[1];
                    d_1_7[2] = _fp4_bf16x8_23[2];
                    d_1_7[3] = _fp4_bf16x8_23[3];
                    unsigned int w4_2_7[4];
                    uint32_t _bf16x2_mul_92;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_92) : "r"(d_1_7[0]), "r"(sc_1[3]));
                    w4_2_7[0] = _bf16x2_mul_92;
                    uint32_t _bf16x2_mul_93;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_93) : "r"(d_1_7[1]), "r"(sc_1[3]));
                    w4_2_7[1] = _bf16x2_mul_93;
                    uint32_t _bf16x2_mul_94;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_94) : "r"(d_1_7[2]), "r"(sc_1[3]));
                    w4_2_7[2] = _bf16x2_mul_94;
                    uint32_t _bf16x2_mul_95;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_95) : "r"(d_1_7[3]), "r"(sc_1[3]));
                    w4_2_7[3] = _bf16x2_mul_95;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_1 + (unsigned int)((7 ^ tm7) << 4)), "r"(w4_2_7[0]), "r"(w4_2_7[1]), "r"(w4_2_7[2]), "r"(w4_2_7[3]));
                    unsigned int plane_1_2 = vbase + 8192;
                    unsigned int d_1_8[4];
                    uint32_t _fp4_bf16x8_24[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(raw_4[8])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(raw_4[8])));
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
                    d_1_8[0] = _fp4_bf16x8_24[0];
                    d_1_8[1] = _fp4_bf16x8_24[1];
                    d_1_8[2] = _fp4_bf16x8_24[2];
                    d_1_8[3] = _fp4_bf16x8_24[3];
                    unsigned int w4_2_8[4];
                    uint32_t _bf16x2_mul_96;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_96) : "r"(d_1_8[0]), "r"(sc_1[4]));
                    w4_2_8[0] = _bf16x2_mul_96;
                    uint32_t _bf16x2_mul_97;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_97) : "r"(d_1_8[1]), "r"(sc_1[4]));
                    w4_2_8[1] = _bf16x2_mul_97;
                    uint32_t _bf16x2_mul_98;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_98) : "r"(d_1_8[2]), "r"(sc_1[4]));
                    w4_2_8[2] = _bf16x2_mul_98;
                    uint32_t _bf16x2_mul_99;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_99) : "r"(d_1_8[3]), "r"(sc_1[4]));
                    w4_2_8[3] = _bf16x2_mul_99;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_2 + (unsigned int)((0 ^ tm7) << 4)), "r"(w4_2_8[0]), "r"(w4_2_8[1]), "r"(w4_2_8[2]), "r"(w4_2_8[3]));
                    unsigned int d_1_9[4];
                    uint32_t _fp4_bf16x8_25[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(raw_4[9])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(raw_4[9])));
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
                    d_1_9[0] = _fp4_bf16x8_25[0];
                    d_1_9[1] = _fp4_bf16x8_25[1];
                    d_1_9[2] = _fp4_bf16x8_25[2];
                    d_1_9[3] = _fp4_bf16x8_25[3];
                    unsigned int w4_2_9[4];
                    uint32_t _bf16x2_mul_100;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_100) : "r"(d_1_9[0]), "r"(sc_1[4]));
                    w4_2_9[0] = _bf16x2_mul_100;
                    uint32_t _bf16x2_mul_101;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_101) : "r"(d_1_9[1]), "r"(sc_1[4]));
                    w4_2_9[1] = _bf16x2_mul_101;
                    uint32_t _bf16x2_mul_102;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_102) : "r"(d_1_9[2]), "r"(sc_1[4]));
                    w4_2_9[2] = _bf16x2_mul_102;
                    uint32_t _bf16x2_mul_103;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_103) : "r"(d_1_9[3]), "r"(sc_1[4]));
                    w4_2_9[3] = _bf16x2_mul_103;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_2 + (unsigned int)((1 ^ tm7) << 4)), "r"(w4_2_9[0]), "r"(w4_2_9[1]), "r"(w4_2_9[2]), "r"(w4_2_9[3]));
                    unsigned int d_1_10[4];
                    uint32_t _fp4_bf16x8_26[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(raw_4[10])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(raw_4[10])));
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
                    d_1_10[0] = _fp4_bf16x8_26[0];
                    d_1_10[1] = _fp4_bf16x8_26[1];
                    d_1_10[2] = _fp4_bf16x8_26[2];
                    d_1_10[3] = _fp4_bf16x8_26[3];
                    unsigned int w4_2_10[4];
                    uint32_t _bf16x2_mul_104;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_104) : "r"(d_1_10[0]), "r"(sc_1[5]));
                    w4_2_10[0] = _bf16x2_mul_104;
                    uint32_t _bf16x2_mul_105;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_105) : "r"(d_1_10[1]), "r"(sc_1[5]));
                    w4_2_10[1] = _bf16x2_mul_105;
                    uint32_t _bf16x2_mul_106;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_106) : "r"(d_1_10[2]), "r"(sc_1[5]));
                    w4_2_10[2] = _bf16x2_mul_106;
                    uint32_t _bf16x2_mul_107;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_107) : "r"(d_1_10[3]), "r"(sc_1[5]));
                    w4_2_10[3] = _bf16x2_mul_107;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_2 + (unsigned int)((2 ^ tm7) << 4)), "r"(w4_2_10[0]), "r"(w4_2_10[1]), "r"(w4_2_10[2]), "r"(w4_2_10[3]));
                    unsigned int d_1_11[4];
                    uint32_t _fp4_bf16x8_27[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(raw_4[11])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(raw_4[11])));
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
                    d_1_11[0] = _fp4_bf16x8_27[0];
                    d_1_11[1] = _fp4_bf16x8_27[1];
                    d_1_11[2] = _fp4_bf16x8_27[2];
                    d_1_11[3] = _fp4_bf16x8_27[3];
                    unsigned int w4_2_11[4];
                    uint32_t _bf16x2_mul_108;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_108) : "r"(d_1_11[0]), "r"(sc_1[5]));
                    w4_2_11[0] = _bf16x2_mul_108;
                    uint32_t _bf16x2_mul_109;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_109) : "r"(d_1_11[1]), "r"(sc_1[5]));
                    w4_2_11[1] = _bf16x2_mul_109;
                    uint32_t _bf16x2_mul_110;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_110) : "r"(d_1_11[2]), "r"(sc_1[5]));
                    w4_2_11[2] = _bf16x2_mul_110;
                    uint32_t _bf16x2_mul_111;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_111) : "r"(d_1_11[3]), "r"(sc_1[5]));
                    w4_2_11[3] = _bf16x2_mul_111;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_2 + (unsigned int)((3 ^ tm7) << 4)), "r"(w4_2_11[0]), "r"(w4_2_11[1]), "r"(w4_2_11[2]), "r"(w4_2_11[3]));
                    unsigned int plane_1_3 = vbase + 8192;
                    unsigned int d_1_12[4];
                    uint32_t _fp4_bf16x8_28[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(raw_4[12])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(raw_4[12])));
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
                    d_1_12[0] = _fp4_bf16x8_28[0];
                    d_1_12[1] = _fp4_bf16x8_28[1];
                    d_1_12[2] = _fp4_bf16x8_28[2];
                    d_1_12[3] = _fp4_bf16x8_28[3];
                    unsigned int w4_2_12[4];
                    uint32_t _bf16x2_mul_112;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_112) : "r"(d_1_12[0]), "r"(sc_1[6]));
                    w4_2_12[0] = _bf16x2_mul_112;
                    uint32_t _bf16x2_mul_113;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_113) : "r"(d_1_12[1]), "r"(sc_1[6]));
                    w4_2_12[1] = _bf16x2_mul_113;
                    uint32_t _bf16x2_mul_114;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_114) : "r"(d_1_12[2]), "r"(sc_1[6]));
                    w4_2_12[2] = _bf16x2_mul_114;
                    uint32_t _bf16x2_mul_115;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_115) : "r"(d_1_12[3]), "r"(sc_1[6]));
                    w4_2_12[3] = _bf16x2_mul_115;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_3 + (unsigned int)((4 ^ tm7) << 4)), "r"(w4_2_12[0]), "r"(w4_2_12[1]), "r"(w4_2_12[2]), "r"(w4_2_12[3]));
                    unsigned int d_1_13[4];
                    uint32_t _fp4_bf16x8_29[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(raw_4[13])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(raw_4[13])));
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
                    d_1_13[0] = _fp4_bf16x8_29[0];
                    d_1_13[1] = _fp4_bf16x8_29[1];
                    d_1_13[2] = _fp4_bf16x8_29[2];
                    d_1_13[3] = _fp4_bf16x8_29[3];
                    unsigned int w4_2_13[4];
                    uint32_t _bf16x2_mul_116;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_116) : "r"(d_1_13[0]), "r"(sc_1[6]));
                    w4_2_13[0] = _bf16x2_mul_116;
                    uint32_t _bf16x2_mul_117;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_117) : "r"(d_1_13[1]), "r"(sc_1[6]));
                    w4_2_13[1] = _bf16x2_mul_117;
                    uint32_t _bf16x2_mul_118;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_118) : "r"(d_1_13[2]), "r"(sc_1[6]));
                    w4_2_13[2] = _bf16x2_mul_118;
                    uint32_t _bf16x2_mul_119;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_119) : "r"(d_1_13[3]), "r"(sc_1[6]));
                    w4_2_13[3] = _bf16x2_mul_119;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_3 + (unsigned int)((5 ^ tm7) << 4)), "r"(w4_2_13[0]), "r"(w4_2_13[1]), "r"(w4_2_13[2]), "r"(w4_2_13[3]));
                    unsigned int d_1_14[4];
                    uint32_t _fp4_bf16x8_30[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(raw_4[14])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(raw_4[14])));
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
                    d_1_14[0] = _fp4_bf16x8_30[0];
                    d_1_14[1] = _fp4_bf16x8_30[1];
                    d_1_14[2] = _fp4_bf16x8_30[2];
                    d_1_14[3] = _fp4_bf16x8_30[3];
                    unsigned int w4_2_14[4];
                    uint32_t _bf16x2_mul_120;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_120) : "r"(d_1_14[0]), "r"(sc_1[7]));
                    w4_2_14[0] = _bf16x2_mul_120;
                    uint32_t _bf16x2_mul_121;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_121) : "r"(d_1_14[1]), "r"(sc_1[7]));
                    w4_2_14[1] = _bf16x2_mul_121;
                    uint32_t _bf16x2_mul_122;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_122) : "r"(d_1_14[2]), "r"(sc_1[7]));
                    w4_2_14[2] = _bf16x2_mul_122;
                    uint32_t _bf16x2_mul_123;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_123) : "r"(d_1_14[3]), "r"(sc_1[7]));
                    w4_2_14[3] = _bf16x2_mul_123;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_3 + (unsigned int)((6 ^ tm7) << 4)), "r"(w4_2_14[0]), "r"(w4_2_14[1]), "r"(w4_2_14[2]), "r"(w4_2_14[3]));
                    unsigned int d_1_15[4];
                    uint32_t _fp4_bf16x8_31[4];
                    {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(raw_4[15])));
                    #else
                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                        "}"
                        : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(raw_4[15])));
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
                    d_1_15[0] = _fp4_bf16x8_31[0];
                    d_1_15[1] = _fp4_bf16x8_31[1];
                    d_1_15[2] = _fp4_bf16x8_31[2];
                    d_1_15[3] = _fp4_bf16x8_31[3];
                    unsigned int w4_2_15[4];
                    uint32_t _bf16x2_mul_124;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_124) : "r"(d_1_15[0]), "r"(sc_1[7]));
                    w4_2_15[0] = _bf16x2_mul_124;
                    uint32_t _bf16x2_mul_125;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_125) : "r"(d_1_15[1]), "r"(sc_1[7]));
                    w4_2_15[1] = _bf16x2_mul_125;
                    uint32_t _bf16x2_mul_126;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_126) : "r"(d_1_15[2]), "r"(sc_1[7]));
                    w4_2_15[2] = _bf16x2_mul_126;
                    uint32_t _bf16x2_mul_127;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_127) : "r"(d_1_15[3]), "r"(sc_1[7]));
                    w4_2_15[3] = _bf16x2_mul_127;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plane_1_3 + (unsigned int)((7 ^ tm7) << 4)), "r"(w4_2_15[0]), "r"(w4_2_15[1]), "r"(w4_2_15[2]), "r"(w4_2_15[3]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(v_full_addr + ((int)((unsigned int)(nt0_2 + n_2) % 2)) * 8);
                        mbarrier_arrive(raw_empty_addr + ((int)((unsigned int)(nt0_2 + n_2) % 4)) * 8);
                    }
                }
#if __CUDA_ARCH__ == 1070
                nt0_2 += n_tiles_1;
                req_cur_2 = request_5;
                vc_cur_2 = valid_count_5;
#else
                nt0_2 += n_tiles_2;
                req_cur_2 = request_4;
                vc_cur_2 = valid_count_4;
#endif
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
#if __CUDA_ARCH__ == 1070
            int phys[3];
            int local[3];
            int column_3 = tile0 * 64 + token;
            bool in_range = column_3 < topk && tile_end > tile0;
            int logical = ((in_range) ? idx[0] : -1);
            idx[0] = logical;
            int column_3_1 = (tile0 + 1) * 64 + token;
            bool in_range_0 = column_3_1 < topk && tile_end > tile0 + 1;
            int logical_0 = ((in_range_0) ? idx[1] : -1);
            idx[1] = logical_0;
            int column_4 = (tile0 + 2) * 64 + token;
            bool in_range_1 = column_4 < topk && tile_end > tile0 + 2;
            int logical_1 = ((in_range_1) ? idx[2] : -1);
            idx[2] = logical_1;
            int column_5 = (tile0 + 3) * 64 + token;
            bool in_range_2 = column_5 < topk && tile_end > tile0 + 3;
            int logical_2 = ((in_range_2) ? idx[3] : -1);
            idx[3] = logical_2;
            int safe_token = ((idx[0]) > (0) ? (idx[0]) : (0));
            uint32_t fast_div_q = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token);
            uint32_t fast_div_r = (uint32_t)((unsigned int)safe_token) - fast_div_q * (uint32_t)(page_div.divisor);
            int page = (int)fast_div_q;
            bool valid = request_ok && idx[0] >= 0 && page < table_width;
            int table_column = ((page) < (table_width - 1) ? (page) : (table_width - 1));
            int entry = ((valid) ? block_table[safe_request_row + (long long)table_column] : -1);
            phys[0] = entry;
            local[0] = (int)fast_div_r;
            int safe_token_0 = ((idx[1]) > (0) ? (idx[1]) : (0));
            uint32_t fast_div_q_0 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_0), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_0);
            uint32_t fast_div_r_0 = (uint32_t)((unsigned int)safe_token_0) - fast_div_q_0 * (uint32_t)(page_div.divisor);
            int page_0 = (int)fast_div_q_0;
            bool valid_0 = request_ok && idx[1] >= 0 && page_0 < table_width;
            int table_column_0 = ((page_0) < (table_width - 1) ? (page_0) : (table_width - 1));
            int entry_0 = ((valid_0) ? block_table[safe_request_row + (long long)table_column_0] : -1);
            phys[1] = entry_0;
            local[1] = (int)fast_div_r_0;
            int nkv1 = g_nxt - g_nrow * hkv;
            long long nrow1_i64 = (long long)g_nrow;
            int nfirst1 = nkv1 * group_size;
            long long _min_0 = ((slot1 - (long long)g_nxt * tiles_i64) < (tiles_i64) ? (slot1 - (long long)g_nxt * tiles_i64) : (tiles_i64));
            int ntile1_0 = (int)_min_0;
            bool request_ok_45 = request_0 >= 0 && request_0 < num_requests;
            int _max_1 = ((request_0) > (0) ? (request_0) : (0));
            int safe_request_46 = ((_max_1) < (num_requests - 1) ? (_max_1) : (num_requests - 1));
            long long safe_request_row_47 = (long long)safe_request_46 * table_stride_req;
            int _min_3_3 = ((valid_count_0) < (topk) ? (valid_count_0) : (topk));
            int _min_2_1 = ((num_tiles) < ((_min_3_3 + 63) / 64) ? (num_tiles) : ((_min_3_3 + 63) / 64));
            int tile_end_48 = ((_min_2_1) < (ntile1_0) ? (_min_2_1) : (ntile1_0));
            int n_tiles_49 = ((tile_end_48) > (0) ? (tile_end_48) : (0));
            int nn0 = ((pair0 + 2) < (pair1) ? (pair0 + 2) : (pair1));
            int nnrow0 = nn0 / hkv;
            long long nnidx_row0 = (long long)nnrow0 * indices_stride_row;
            int request_2 = token_to_req[nnrow0];
            int valid_count_1 = indices[nnidx_row0 + (long long)topk];
#endif
            int e_1_8 = gtid;
            int head_2_8 = e_1_8 >> 5;
            int c_1_8 = e_1_8 & 31;
            int panel_1_0 = c_1_8 >> 3;
            int cc_1_0 = c_1_8 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_0 * 2048) + (unsigned int)(head_2_8 * 128 + 16 * cc_1_0 ^ (head_2_8 * 128 + 16 * cc_1_0 >> 7 & 7) << 4))), "r"(qw_regs[0]), "r"(qw_regs[1]), "r"(qw_regs[2]), "r"(qw_regs[3]) : "memory");
            int e_1_9 = gtid + 64;
            int head_2_9 = e_1_9 >> 5;
            int c_1_9 = e_1_9 & 31;
            int panel_1_1 = c_1_9 >> 3;
            int cc_1_1 = c_1_9 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_1 * 2048) + (unsigned int)(head_2_9 * 128 + 16 * cc_1_1 ^ (head_2_9 * 128 + 16 * cc_1_1 >> 7 & 7) << 4))), "r"(qw_regs[4]), "r"(qw_regs[5]), "r"(qw_regs[6]), "r"(qw_regs[7]) : "memory");
            int e_1_10 = gtid + 128;
            int head_2_10 = e_1_10 >> 5;
            int c_1_10 = e_1_10 & 31;
            int panel_1_2 = c_1_10 >> 3;
            int cc_1_2 = c_1_10 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_2 * 2048) + (unsigned int)(head_2_10 * 128 + 16 * cc_1_2 ^ (head_2_10 * 128 + 16 * cc_1_2 >> 7 & 7) << 4))), "r"(qw_regs[8]), "r"(qw_regs[9]), "r"(qw_regs[10]), "r"(qw_regs[11]) : "memory");
            int e_1_11 = gtid + 192;
            int head_2_11 = e_1_11 >> 5;
            int c_1_11 = e_1_11 & 31;
            int panel_1_3 = c_1_11 >> 3;
            int cc_1_3 = c_1_11 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_3 * 2048) + (unsigned int)(head_2_11 * 128 + 16 * cc_1_3 ^ (head_2_11 * 128 + 16 * cc_1_3 >> 7 & 7) << 4))), "r"(qw_regs[12]), "r"(qw_regs[13]), "r"(qw_regs[14]), "r"(qw_regs[15]) : "memory");
            int e_1_12 = gtid + 256;
            int head_2_12 = e_1_12 >> 5;
            int c_1_12 = e_1_12 & 31;
            int panel_1_4 = c_1_12 >> 3;
            int cc_1_4 = c_1_12 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_4 * 2048) + (unsigned int)(head_2_12 * 128 + 16 * cc_1_4 ^ (head_2_12 * 128 + 16 * cc_1_4 >> 7 & 7) << 4))), "r"(qw_regs[16]), "r"(qw_regs[17]), "r"(qw_regs[18]), "r"(qw_regs[19]) : "memory");
            int e_1_13 = gtid + 320;
            int head_2_13 = e_1_13 >> 5;
            int c_1_13 = e_1_13 & 31;
            int panel_1_5 = c_1_13 >> 3;
            int cc_1_5 = c_1_13 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_5 * 2048) + (unsigned int)(head_2_13 * 128 + 16 * cc_1_5 ^ (head_2_13 * 128 + 16 * cc_1_5 >> 7 & 7) << 4))), "r"(qw_regs[20]), "r"(qw_regs[21]), "r"(qw_regs[22]), "r"(qw_regs[23]) : "memory");
            int e_1_14 = gtid + 384;
            int head_2_14 = e_1_14 >> 5;
            int c_1_14 = e_1_14 & 31;
            int panel_1_6 = c_1_14 >> 3;
            int cc_1_6 = c_1_14 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_6 * 2048) + (unsigned int)(head_2_14 * 128 + 16 * cc_1_6 ^ (head_2_14 * 128 + 16 * cc_1_6 >> 7 & 7) << 4))), "r"(qw_regs[24]), "r"(qw_regs[25]), "r"(qw_regs[26]), "r"(qw_regs[27]) : "memory");
            int e_1_15 = gtid + 448;
            int head_2_15 = e_1_15 >> 5;
            int c_1_15 = e_1_15 & 31;
            int panel_1_7 = c_1_15 >> 3;
            int cc_1_7 = c_1_15 & 7;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_addr + (unsigned int)(panel_1_7 * 2048) + (unsigned int)(head_2_15 * 128 + 16 * cc_1_7 ^ (head_2_15 * 128 + 16 * cc_1_7 >> 7 & 7) << 4))), "r"(qw_regs[28]), "r"(qw_regs[29]), "r"(qw_regs[30]), "r"(qw_regs[31]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(q_full_addr);
            }
            int nt0_3 = 0;
            int req_cur_3 = request;
            int vc_cur_3 = valid_count;
            int walk_nt0 = 0;
            int walk_pairs = 0;
#if __CUDA_ARCH__ == 1070
            bool req_ok = request_ok;
            long long safe_req_row = safe_request_row;
            int tend = tile_end;
            int ntiles = n_tiles;
            long long indices_row_4_1 = indices_row;
            int tile0_6 = tile0;
            int kv_head_1_1 = kv_head;
            bool n_rok = request_ok_45;
            long long n_sreq = safe_request_row_47;
            int n_tend = tile_end_48;
            int n_ntiles = n_tiles_49;
            int n_kv = nkv1;
            long long nrow_i64_3 = nrow1_i64;
            int nfirst_3 = nfirst1;
            long long nidx_row_3 = g_nidx_row;
            int l_req = request_2;
            int l_vc = valid_count_1;
#else
            bool request_ok_0 = request >= 0 && request < num_requests;
            int _max_0_5 = ((request) > (0) ? (request) : (0));
            int safe_request_0 = ((_max_0_5) < (num_requests - 1) ? (_max_0_5) : (num_requests - 1));
            long long safe_request_row_0 = (long long)safe_request_0 * table_stride_req;
            int _min_2_1 = ((valid_count) < (topk) ? (valid_count) : (topk));
            int _min_1_5 = ((num_tiles) < ((_min_2_1 + 63) / 64) ? (num_tiles) : ((_min_2_1 + 63) / 64));
            int tile_end_0 = ((_min_1_5) < (tile1) ? (_min_1_5) : (tile1));
            int n_tiles_0 = ((tile_end_0 - tile0) > (0) ? (tile_end_0 - tile0) : (0));
            bool n_rok = request_ok_0;
            long long n_sreq = safe_request_row_0;
            int n_tend = tile_end_0;
            int n_ntiles = n_tiles_0;
            int l_req = request_0;
            int l_vc = valid_count_0;
#endif
            int pre_idx = 4;
#if __CUDA_ARCH__ == 1070
            int pre_blk = 2;
            unsigned long long data_stride_block_u64 = (unsigned long long)data_stride_block;
            unsigned long long data_stride_token_u64 = (unsigned long long)data_stride_token;
            unsigned long long sf_stride_block_u64 = (unsigned long long)sf_stride_block;
            unsigned long long sf_stride_token_u64 = (unsigned long long)sf_stride_token;
            unsigned long long kv_head_u64 = (unsigned long long)kv_head;
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
#else
            int pre_blk = 0;
            int phys[3];
            int local[3];
#endif
            #pragma unroll 1
            for (int P_3 = pair0; P_3 < pair1 + 1; P_3++) {
                int pi = P_3 - pair0;
                bool has_next = pair1 > P_3;
#if __CUDA_ARCH__ == 1070
                if (pi > 0) {
                    req_ok = n_rok;
                    safe_req_row = n_sreq;
                    tend = n_tend;
                    ntiles = n_ntiles;
                    indices_row_4_1 = nidx_row_3;
                    tile0_6 = 0;
                    kv_head_1_1 = n_kv;
                    int nxt_3 = ((P_3 + 1) < (pair1) ? (P_3 + 1) : (pair1));
                    int nrow_3 = nxt_3 / hkv;
                    n_kv = nxt_3 - nrow_3 * hkv;
                    nrow_i64_3 = (long long)nrow_3;
                    nfirst_3 = n_kv * group_size;
                    nidx_row_3 = nrow_i64_3 * indices_stride_row;
                    long long _min_4_3 = ((slot1 - (long long)nxt_3 * tiles_i64) < (tiles_i64) ? (slot1 - (long long)nxt_3 * tiles_i64) : (tiles_i64));
                    int ntile1 = (int)_min_4_3;
                    bool request_ok_12 = l_req >= 0 && l_req < num_requests;
                    int _max_5 = ((l_req) > (0) ? (l_req) : (0));
                    int safe_request_13 = ((_max_5) < (num_requests - 1) ? (_max_5) : (num_requests - 1));
                    long long safe_request_row_14 = (long long)safe_request_13 * table_stride_req;
                    int _min_7 = ((l_vc) < (topk) ? (l_vc) : (topk));
                    int _min_6 = ((num_tiles) < ((_min_7 + 63) / 64) ? (num_tiles) : ((_min_7 + 63) / 64));
                    int tile_end_15 = ((_min_6) < (ntile1) ? (_min_6) : (ntile1));
                    int n_tiles_16 = ((tile_end_15) > (0) ? (tile_end_15) : (0));
                    n_rok = request_ok_12;
                    n_sreq = safe_request_row_14;
                    n_tend = tile_end_15;
                    n_ntiles = n_tiles_16;
                    int nn = ((P_3 + 2) < (pair1) ? (P_3 + 2) : (pair1));
                    int nnrow = nn / hkv;
                    long long nnidx_row = (long long)nnrow * indices_stride_row;
                    int request_3 = token_to_req[nnrow];
                    int valid_count_3 = indices[nnidx_row + (long long)topk];
                    l_req = request_3;
                    l_vc = valid_count_3;
                    if (4 - pre_idx > 0) {
                        int column_6 = tile0_6 * 64 + token;
                        int safe_column_3 = ((column_6) < (topk - 1) ? (column_6) : (topk - 1));
                        int raw_5 = indices[indices_row_4_1 + (long long)safe_column_3];
                        idx[0] = raw_5;
                    }
                    if (4 - pre_idx > 1) {
                        int column_7 = (tile0_6 + 1) * 64 + token;
                        int safe_column_4 = ((column_7) < (topk - 1) ? (column_7) : (topk - 1));
                        int raw_6 = indices[indices_row_4_1 + (long long)safe_column_4];
                        idx[1] = raw_6;
                    }
                    if (4 - pre_idx > 2) {
                        int column_8 = (tile0_6 + 2) * 64 + token;
                        int safe_column_5 = ((column_8) < (topk - 1) ? (column_8) : (topk - 1));
                        int raw_7 = indices[indices_row_4_1 + (long long)safe_column_5];
                        idx[2] = raw_7;
                    }
                    if (4 - pre_idx > 3) {
                        int column_9 = (tile0_6 + 3) * 64 + token;
                        int safe_column_6 = ((column_9) < (topk - 1) ? (column_9) : (topk - 1));
                        int raw_8 = indices[indices_row_4_1 + (long long)safe_column_6];
                        idx[3] = raw_8;
                    }
                    int column_10 = tile0_6 * 64 + token;
                    bool in_range_3 = column_10 < topk && tend > tile0_6;
                    int logical_3 = ((in_range_3) ? idx[0] : -1);
                    idx[0] = logical_3;
                    int column_11 = (tile0_6 + 1) * 64 + token;
                    bool in_range_4 = column_11 < topk && tend > tile0_6 + 1;
                    int logical_4 = ((in_range_4) ? idx[1] : -1);
                    idx[1] = logical_4;
                    int column_12 = (tile0_6 + 2) * 64 + token;
                    bool in_range_5 = column_12 < topk && tend > tile0_6 + 2;
                    int logical_5 = ((in_range_5) ? idx[2] : -1);
                    idx[2] = logical_5;
                    int column_13 = (tile0_6 + 3) * 64 + token;
                    bool in_range_6 = column_13 < topk && tend > tile0_6 + 3;
                    int logical_6 = ((in_range_6) ? idx[3] : -1);
                    idx[3] = logical_6;
                    if (2 - pre_blk > 0) {
                        int safe_token_1 = ((idx[0]) > (0) ? (idx[0]) : (0));
                        uint32_t fast_div_q_1 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_1), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_1);
                        uint32_t fast_div_r_1 = (uint32_t)((unsigned int)safe_token_1) - fast_div_q_1 * (uint32_t)(page_div.divisor);
                        int page_1 = (int)fast_div_q_1;
                        bool valid_1 = req_ok && idx[0] >= 0 && page_1 < table_width;
                        int table_column_1 = ((page_1) < (table_width - 1) ? (page_1) : (table_width - 1));
                        int entry_1 = ((valid_1) ? block_table[safe_req_row + (long long)table_column_1] : -1);
                        phys[0] = entry_1;
                        local[0] = (int)fast_div_r_1;
                    }
                    if (2 - pre_blk > 1) {
                        int safe_token_2 = ((idx[1]) > (0) ? (idx[1]) : (0));
                        uint32_t fast_div_q_2 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_2), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_2);
                        uint32_t fast_div_r_2 = (uint32_t)((unsigned int)safe_token_2) - fast_div_q_2 * (uint32_t)(page_div.divisor);
                        int page_2 = (int)fast_div_q_2;
                        bool valid_2 = req_ok && idx[1] >= 0 && page_2 < table_width;
                        int table_column_2 = ((page_2) < (table_width - 1) ? (page_2) : (table_width - 1));
                        int entry_2 = ((valid_2) ? block_table[safe_req_row + (long long)table_column_2] : -1);
                        phys[1] = entry_2;
                        local[1] = (int)fast_div_r_2;
                    }
                    kv_head_u64 = (unsigned long long)kv_head_1_1;
                    head_data = kv_head_u64 * (unsigned long long)data_stride_head;
                    head_sf = kv_head_u64 * (unsigned long long)sf_stride_head;
                    k_base = (unsigned long long)k_data + head_data;
                    v_base = (unsigned long long)v_data + head_data;
                    k_sf_base = (unsigned long long)k_sf + head_sf;
                    v_sf_base = (unsigned long long)v_sf + head_sf;
#else
                int row_0 = P_3 / hkv;
                int kv_head_0 = P_3 - row_0 * hkv;
                long long row_i64_0 = (long long)row_0;
                int first_head_0 = kv_head_0 * group_size;
                long long indices_row_0 = row_i64_0 * indices_stride_row;
                long long pslot_0 = (long long)P_3 * tiles_i64;
                long long _max_3 = ((slot0 - pslot_0) > (0) ? (slot0 - pslot_0) : (0));
                int tile0_0 = (int)_max_3;
                long long _min_4_3 = ((slot1 - pslot_0) < (tiles_i64) ? (slot1 - pslot_0) : (tiles_i64));
                int tile1_0 = (int)_min_4_3;
                int owner0_0 = (int)(((pslot_0 + 1) * ctas_i64 - 1) / slots);
                int cta_1 = blockIdx.x;
                int seg_0 = cta_1 - owner0_0;
                bool last_0 = slot1 >= pslot_0 + tiles_i64;
                bool req_ok = n_rok;
                long long safe_req_row = n_sreq;
                int tend = n_tend;
                int ntiles = n_ntiles;
                int nxt_3 = ((P_3 + 1) < (pair1) ? (P_3 + 1) : (pair1));
                int nrow_3 = nxt_3 / hkv;
                long long nrow_i64_3 = (long long)nrow_3;
                int nfirst_3 = (nxt_3 - nrow_3 * hkv) * group_size;
                long long nidx_row_3 = nrow_i64_3 * indices_stride_row;
                long long _min_5 = ((slot1 - (long long)nxt_3 * tiles_i64) < (tiles_i64) ? (slot1 - (long long)nxt_3 * tiles_i64) : (tiles_i64));
                int ntile1 = (int)_min_5;
                bool request_ok_12 = l_req >= 0 && l_req < num_requests;
                int _max_6 = ((l_req) > (0) ? (l_req) : (0));
                int safe_request_13 = ((_max_6) < (num_requests - 1) ? (_max_6) : (num_requests - 1));
                long long safe_request_row_14 = (long long)safe_request_13 * table_stride_req;
                int _min_8 = ((l_vc) < (topk) ? (l_vc) : (topk));
                int _min_7 = ((num_tiles) < ((_min_8 + 63) / 64) ? (num_tiles) : ((_min_8 + 63) / 64));
                int tile_end_15 = ((_min_7) < (ntile1) ? (_min_7) : (ntile1));
                int n_tiles_16 = ((tile_end_15) > (0) ? (tile_end_15) : (0));
                n_rok = request_ok_12;
                n_sreq = safe_request_row_14;
                n_tend = tile_end_15;
                n_ntiles = n_tiles_16;
                int nn = ((P_3 + 2) < (pair1) ? (P_3 + 2) : (pair1));
                int nnrow = nn / hkv;
                long long nnidx_row = (long long)nnrow * indices_stride_row;
                int request_2 = token_to_req[nnrow];
                int valid_count_1 = indices[nnidx_row + (long long)topk];
                l_req = request_2;
                l_vc = valid_count_1;
                if (4 - pre_idx > 0) {
                    int column_3 = tile0_0 * 64 + token;
                    int safe_column_3 = ((column_3) < (topk - 1) ? (column_3) : (topk - 1));
                    int raw_5 = indices[indices_row_0 + (long long)safe_column_3];
                    idx[0] = raw_5;
#endif
                }
#if __CUDA_ARCH__ == 1070
#else
                if (4 - pre_idx > 1) {
                    int column_4 = (tile0_0 + 1) * 64 + token;
                    int safe_column_4 = ((column_4) < (topk - 1) ? (column_4) : (topk - 1));
                    int raw_6 = indices[indices_row_0 + (long long)safe_column_4];
                    idx[1] = raw_6;
                }
                if (4 - pre_idx > 2) {
                    int column_5 = (tile0_0 + 2) * 64 + token;
                    int safe_column_5 = ((column_5) < (topk - 1) ? (column_5) : (topk - 1));
                    int raw_7 = indices[indices_row_0 + (long long)safe_column_5];
                    idx[2] = raw_7;
                }
                if (4 - pre_idx > 3) {
                    int column_6 = (tile0_0 + 3) * 64 + token;
                    int safe_column_6 = ((column_6) < (topk - 1) ? (column_6) : (topk - 1));
                    int raw_8 = indices[indices_row_0 + (long long)safe_column_6];
                    idx[3] = raw_8;
                }
                int column_7 = tile0_0 * 64 + token;
                bool in_range = column_7 < topk && tend > tile0_0;
                int logical = ((in_range) ? idx[0] : -1);
                idx[0] = logical;
                int column_3_1 = (tile0_0 + 1) * 64 + token;
                bool in_range_0 = column_3_1 < topk && tend > tile0_0 + 1;
                int logical_0 = ((in_range_0) ? idx[1] : -1);
                idx[1] = logical_0;
                int column_4_1 = (tile0_0 + 2) * 64 + token;
                bool in_range_1 = column_4_1 < topk && tend > tile0_0 + 2;
                int logical_1 = ((in_range_1) ? idx[2] : -1);
                idx[2] = logical_1;
                int column_5_1 = (tile0_0 + 3) * 64 + token;
                bool in_range_2 = column_5_1 < topk && tend > tile0_0 + 3;
                int logical_2 = ((in_range_2) ? idx[3] : -1);
                idx[3] = logical_2;
                if (2 - pre_blk > 0) {
                    int safe_token = ((idx[0]) > (0) ? (idx[0]) : (0));
                    uint32_t fast_div_q = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token);
                    uint32_t fast_div_r = (uint32_t)((unsigned int)safe_token) - fast_div_q * (uint32_t)(page_div.divisor);
                    int page = (int)fast_div_q;
                    bool valid = req_ok && idx[0] >= 0 && page < table_width;
                    int table_column = ((page) < (table_width - 1) ? (page) : (table_width - 1));
                    int entry = ((valid) ? block_table[safe_req_row + (long long)table_column] : -1);
                    phys[0] = entry;
                    local[0] = (int)fast_div_r;
                }
                if (2 - pre_blk > 1) {
                    int safe_token_1 = ((idx[1]) > (0) ? (idx[1]) : (0));
                    uint32_t fast_div_q_1 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_1), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_1);
                    uint32_t fast_div_r_1 = (uint32_t)((unsigned int)safe_token_1) - fast_div_q_1 * (uint32_t)(page_div.divisor);
                    int page_1 = (int)fast_div_q_1;
                    bool valid_1 = req_ok && idx[1] >= 0 && page_1 < table_width;
                    int table_column_1 = ((page_1) < (table_width - 1) ? (page_1) : (table_width - 1));
                    int entry_1 = ((valid_1) ? block_table[safe_req_row + (long long)table_column_1] : -1);
                    phys[1] = entry_1;
                    local[1] = (int)fast_div_r_1;
                }
                unsigned long long kv_head_u64 = (unsigned long long)kv_head_0;
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
#endif
                #pragma unroll 1
                for (int n_3 = 0; n_3 < ntiles; n_3++) {
                    int nt_3 = walk_nt0 + n_3;
                    int st_2 = nt_3 % 4;
                    int slot = nt_3 & 1;
                    int a4_1 = n_3 + 4;
                    bool cross4 = a4_1 >= ntiles;
#if __CUDA_ARCH__ == 1070
                    int t4 = ((cross4) ? a4_1 - ntiles : tile0_6 + a4_1);
                    long long row4 = ((cross4) ? nidx_row_3 : indices_row_4_1);
#else
                    int t4 = ((cross4) ? a4_1 - ntiles : tile0_0 + a4_1);
                    long long row4 = ((cross4) ? nidx_row_3 : indices_row_0);
#endif
                    int tend_next = ((has_next) ? n_tend : 0);
                    int tend4 = ((cross4) ? tend_next : tend);
                    int column_0_1 = t4 * 64 + token;
                    bool in_range_1_0 = column_0_1 < topk && t4 < tend4;
                    int safe_column_7 = ((column_0_1) < (topk - 1) ? (column_0_1) : (topk - 1));
                    int logical_2_0 = ((in_range_1_0) ? indices[row4 + (long long)safe_column_7] : -1);
                    bool cross2 = ntiles <= n_3 + 2;
                    long long sreq2 = ((cross2) ? n_sreq : safe_req_row);
                    bool rok_next = n_rok && has_next;
                    bool rok2 = ((cross2) ? rok_next : req_ok);
#if __CUDA_ARCH__ == 1070
                    int safe_token_3 = ((idx[2]) > (0) ? (idx[2]) : (0));
                    uint32_t fast_div_q_3 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_3), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_3);
                    uint32_t fast_div_r_3 = (uint32_t)((unsigned int)safe_token_3) - fast_div_q_3 * (uint32_t)(page_div.divisor);
                    int page_3 = (int)fast_div_q_3;
                    bool valid_3 = rok2 && idx[2] >= 0 && page_3 < table_width;
                    int table_column_3 = ((page_3) < (table_width - 1) ? (page_3) : (table_width - 1));
                    int entry_3 = ((valid_3) ? block_table[sreq2 + (long long)table_column_3] : -1);
                    phys[2] = entry_3;
                    local[2] = (int)fast_div_r_3;
                    bool valid_3_0 = phys[0] >= 0 && phys[0] < num_blocks;
                    int _max_8 = ((phys[0]) > (0) ? (phys[0]) : (0));
                    unsigned long long block = (unsigned long long)_max_8;
#else
                    int safe_token_2 = ((idx[2]) > (0) ? (idx[2]) : (0));
                    uint32_t fast_div_q_2 = (page_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)safe_token_2), page_div.multiplier) >> page_div.shift_right) : (uint32_t)((unsigned int)safe_token_2);
                    uint32_t fast_div_r_2 = (uint32_t)((unsigned int)safe_token_2) - fast_div_q_2 * (uint32_t)(page_div.divisor);
                    int page_2 = (int)fast_div_q_2;
                    bool valid_2 = rok2 && idx[2] >= 0 && page_2 < table_width;
                    int table_column_2 = ((page_2) < (table_width - 1) ? (page_2) : (table_width - 1));
                    int entry_2 = ((valid_2) ? block_table[sreq2 + (long long)table_column_2] : -1);
                    phys[2] = entry_2;
                    local[2] = (int)fast_div_r_2;
                    bool valid_3 = phys[0] >= 0 && phys[0] < num_blocks;
                    int _max_9 = ((phys[0]) > (0) ? (phys[0]) : (0));
                    unsigned long long block = (unsigned long long)_max_9;
#endif
                    unsigned long long local_u64 = (unsigned long long)local[0];
                    unsigned long long data_off = block * data_stride_block_u64 + local_u64 * data_stride_token_u64;
                    unsigned long long sf_off = block * sf_stride_block_u64 + local_u64 * sf_stride_token_u64;
                    unsigned int w0 = (unsigned int)data_off;
                    unsigned int w1 = (unsigned int)(data_off >> 32);
                    unsigned int w2 = (unsigned int)sf_off;
                    unsigned int w3 = (unsigned int)(sf_off >> 32);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_ring_addr + (unsigned int)(slot * 1024 + token * 16)), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
#if __CUDA_ARCH__ == 1070
                    unsigned int bits_2 = __ballot_sync(0xFFFFFFFF, valid_3_0);
#else
                    unsigned int bits_2 = __ballot_sync(0xFFFFFFFF, valid_3);
#endif
                    if (lane_4 == 0) {
                        slot_valid[slot * 2 + g_1] = bits_2;
                    }
                    asm volatile("barrier.sync 1, 64;" ::: "memory");
                    int _stage_raw_empty = (int)((unsigned int)(walk_nt0 + n_3) % 4);
                    int _parity_raw_empty = (int)((unsigned int)(walk_nt0 + n_3) / 4 % 2) ^ 1;
                    mbarrier_wait(raw_empty_addr + (_stage_raw_empty) * 8, _parity_raw_empty);
                    int _stage_valid_empty = (int)((unsigned int)(walk_nt0 + n_3) % 4);
                    int _parity_valid_empty = (int)((unsigned int)(walk_nt0 + n_3) / 4 % 2) ^ 1;
                    mbarrier_wait(valid_empty_addr + (_stage_valid_empty) * 8, _parity_valid_empty);
                    if (gtid < 2) {
                        valid_s[st_2 * 2 + gtid] = slot_valid[slot * 2 + gtid];
                        mbarrier_arrive(valid_full_addr + ((int)((unsigned int)(walk_nt0 + n_3) % 4)) * 8);
                    }
                    unsigned int bits_4 = slot_valid[slot * 2 + (g_1 >> 1)] >> (unsigned int)(16 * (g_1 & 1));
                    int col = lane_4 & 7;
                    int sub = lane_4 >> 3;
                    int token0 = 16 * g_1 + sub;
                    unsigned int entry_addr = addr_ring_addr + (unsigned int)(slot * 1024 + token0 * 16);
                    unsigned int bits_sub = bits_4 >> (unsigned int)sub;
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
#if __CUDA_ARCH__ == 1070
                    __int128_t _smem_raw_9;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(entry_addr));
                    entries_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                    entries_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                    entries_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                    entries_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
#endif
                    __int128_t _smem_raw_10;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(entry_addr + 64));
                    entries_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                    entries_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                    entries_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                    entries_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(entry_addr));
                    entries_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                    entries_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                    entries_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                    entries_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
#endif
                    __int128_t _smem_raw_11;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(entry_addr + 128));
                    entries_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                    entries_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                    entries_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                    entries_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(entry_addr + 64));
                    entries_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                    entries_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                    entries_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                    entries_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
#endif
                    __int128_t _smem_raw_12;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(entry_addr + 192));
                    entries_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
                    entries_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
                    entries_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
                    entries_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(entry_addr + 128));
                    entries_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
                    entries_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
                    entries_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
                    entries_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
#endif
                    __int128_t _smem_raw_13;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(addr_ring_addr + (unsigned int)(slot * 1024 + sf_token * 16)));
                    sf_entry[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
                    sf_entry[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
                    sf_entry[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
                    sf_entry[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(entry_addr + 192));
                    entries_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
                    entries_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
                    entries_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
                    entries_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
                    __int128_t _smem_raw_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(addr_ring_addr + (unsigned int)(slot * 1024 + sf_token * 16)));
                    sf_entry[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                    sf_entry[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                    sf_entry[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                    sf_entry[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
#endif
                    unsigned long long k_src_base = k_base + col_off;
                    unsigned long long v_src_base = v_base + col_off;
                    v_bits_a[0] = bits_sub;
                    unsigned long long data_off_5 = (unsigned long long)entries_0[0] | (unsigned long long)entries_0[1] << 32;
                    unsigned int src_size = (bits_sub & 1) << 4;
                    v_src_a[0] = v_src_base + data_off_5;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0), "l"(k_src_base + data_off_5), "r"(src_size), "l"(0x12F0000000000000ULL));
                    unsigned long long data_off_6 = (unsigned long long)entries_1[0] | (unsigned long long)entries_1[1] << 32;
                    unsigned int src_size_7 = (bits_sub >> 4 & 1) << 4;
                    v_src_a[1] = v_src_base + data_off_6;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0 + 576), "l"(k_src_base + data_off_6), "r"(src_size_7), "l"(0x12F0000000000000ULL));
                    unsigned long long data_off_8 = (unsigned long long)entries_2[0] | (unsigned long long)entries_2[1] << 32;
                    unsigned int src_size_9 = (bits_sub >> 8 & 1) << 4;
                    v_src_a[2] = v_src_base + data_off_8;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0 + 1152), "l"(k_src_base + data_off_8), "r"(src_size_9), "l"(0x12F0000000000000ULL));
                    unsigned long long data_off_10 = (unsigned long long)entries_3[0] | (unsigned long long)entries_3[1] << 32;
                    unsigned int src_size_11 = (bits_sub >> 12 & 1) << 4;
                    v_src_a[3] = v_src_base + data_off_10;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0 + 1728), "l"(k_src_base + data_off_10), "r"(src_size_11), "l"(0x12F0000000000000ULL));
                    unsigned long long sf_off_12 = (unsigned long long)sf_entry[2] | (unsigned long long)sf_entry[3] << 32;
                    unsigned long long sf_base = ((sf_is_v) ? v_sf_base : k_sf_base);
                    int sf_side = ((sf_is_v) ? 9216 : 0);
                    unsigned int sf_size = (bits_4 >> (unsigned int)sf_lane & 1) << 4;
                    unsigned int sf_dst = ((stage_base + (unsigned int)(sf_side + sf_token * 144 + 128)) < (kv_addr + 73712) ? (stage_base + (unsigned int)(sf_side + sf_token * 144 + 128)) : (kv_addr + 73712));
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(sf_dst), "l"(sf_base + sf_off_12), "r"(sf_size), "l"(0x12F0000000000000ULL));
                    unsigned int bits_13 = slot_valid[slot * 2 + (g_1 + 2 >> 1)] >> (unsigned int)(16 * (g_1 + 2 & 1));
                    int col_14 = lane_4 & 7;
                    int sub_15 = lane_4 >> 3;
                    int token0_16 = 16 * (g_1 + 2) + sub_15;
                    unsigned int entry_addr_17 = addr_ring_addr + (unsigned int)(slot * 1024 + token0_16 * 16);
                    unsigned int bits_sub_18 = bits_13 >> (unsigned int)sub_15;
                    unsigned long long col_off_19 = (unsigned long long)(col_14 * 16);
                    unsigned int stage_base_20 = kv_addr + (unsigned int)(st_2 * 18432);
                    unsigned int dst_k0_21 = ((stage_base_20 + (unsigned int)(token0_16 * 144 + col_14 * 16)) < (kv_addr + 73712) ? (stage_base_20 + (unsigned int)(token0_16 * 144 + col_14 * 16)) : (kv_addr + 73712));
                    int sf_lane_22 = lane_4 & 15;
                    bool sf_is_v_23 = lane_4 >= 16;
                    int sf_token_24 = 16 * (g_1 + 2) + sf_lane_22;
                    unsigned int entries_0_25[4];
                    unsigned int entries_1_26[4];
                    unsigned int entries_2_27[4];
                    unsigned int entries_3_28[4];
                    unsigned int sf_entry_29[4];
#if __CUDA_ARCH__ == 1070
                    __int128_t _smem_raw_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(entry_addr_17));
                    entries_0_25[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                    entries_0_25[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                    entries_0_25[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                    entries_0_25[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
#endif
                    __int128_t _smem_raw_15;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_15) : "r"(entry_addr_17 + 64));
                    entries_1_26[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[0];
                    entries_1_26[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[1];
                    entries_1_26[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[2];
                    entries_1_26[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_15) : "r"(entry_addr_17));
                    entries_0_25[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[0];
                    entries_0_25[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[1];
                    entries_0_25[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[2];
                    entries_0_25[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[3];
#endif
                    __int128_t _smem_raw_16;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_16) : "r"(entry_addr_17 + 128));
                    entries_2_27[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[0];
                    entries_2_27[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[1];
                    entries_2_27[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[2];
                    entries_2_27[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_16) : "r"(entry_addr_17 + 64));
                    entries_1_26[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[0];
                    entries_1_26[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[1];
                    entries_1_26[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[2];
                    entries_1_26[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[3];
#endif
                    __int128_t _smem_raw_17;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(entry_addr_17 + 192));
                    entries_3_28[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
                    entries_3_28[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
                    entries_3_28[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
                    entries_3_28[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(entry_addr_17 + 128));
                    entries_2_27[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
                    entries_2_27[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
                    entries_2_27[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
                    entries_2_27[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
#endif
                    __int128_t _smem_raw_18;
#if __CUDA_ARCH__ == 1070
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(addr_ring_addr + (unsigned int)(slot * 1024 + sf_token_24 * 16)));
                    sf_entry_29[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
                    sf_entry_29[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
                    sf_entry_29[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
                    sf_entry_29[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
#else
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(entry_addr_17 + 192));
                    entries_3_28[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
                    entries_3_28[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
                    entries_3_28[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
                    entries_3_28[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
                    __int128_t _smem_raw_19;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(addr_ring_addr + (unsigned int)(slot * 1024 + sf_token_24 * 16)));
                    sf_entry_29[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
                    sf_entry_29[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
                    sf_entry_29[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
                    sf_entry_29[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
#endif
                    unsigned long long k_src_base_30 = k_base + col_off_19;
                    unsigned long long v_src_base_31 = v_base + col_off_19;
                    v_bits_b[0] = bits_sub_18;
                    unsigned long long data_off_0 = (unsigned long long)entries_0_25[0] | (unsigned long long)entries_0_25[1] << 32;
                    unsigned int src_size_0 = (bits_sub_18 & 1) << 4;
                    v_src_b[0] = v_src_base_31 + data_off_0;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0_21), "l"(k_src_base_30 + data_off_0), "r"(src_size_0), "l"(0x12F0000000000000ULL));
                    unsigned long long data_off_1 = (unsigned long long)entries_1_26[0] | (unsigned long long)entries_1_26[1] << 32;
                    unsigned int src_size_1 = (bits_sub_18 >> 4 & 1) << 4;
                    v_src_b[1] = v_src_base_31 + data_off_1;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0_21 + 576), "l"(k_src_base_30 + data_off_1), "r"(src_size_1), "l"(0x12F0000000000000ULL));
                    unsigned long long data_off_2 = (unsigned long long)entries_2_27[0] | (unsigned long long)entries_2_27[1] << 32;
                    unsigned int src_size_2 = (bits_sub_18 >> 8 & 1) << 4;
                    v_src_b[2] = v_src_base_31 + data_off_2;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0_21 + 1152), "l"(k_src_base_30 + data_off_2), "r"(src_size_2), "l"(0x12F0000000000000ULL));
                    unsigned long long data_off_3 = (unsigned long long)entries_3_28[0] | (unsigned long long)entries_3_28[1] << 32;
                    unsigned int src_size_3 = (bits_sub_18 >> 12 & 1) << 4;
                    v_src_b[3] = v_src_base_31 + data_off_3;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_k0_21 + 1728), "l"(k_src_base_30 + data_off_3), "r"(src_size_3), "l"(0x12F0000000000000ULL));
                    unsigned long long sf_off_40 = (unsigned long long)sf_entry_29[2] | (unsigned long long)sf_entry_29[3] << 32;
                    unsigned long long sf_base_41 = ((sf_is_v_23) ? v_sf_base : k_sf_base);
                    int sf_side_42 = ((sf_is_v_23) ? 9216 : 0);
                    unsigned int sf_size_43 = (bits_13 >> (unsigned int)sf_lane_22 & 1) << 4;
                    unsigned int sf_dst_44 = ((stage_base_20 + (unsigned int)(sf_side_42 + sf_token_24 * 144 + 128)) < (kv_addr + 73712) ? (stage_base_20 + (unsigned int)(sf_side_42 + sf_token_24 * 144 + 128)) : (kv_addr + 73712));
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(sf_dst_44), "l"(sf_base_41 + sf_off_40), "r"(sf_size_43), "l"(0x12F0000000000000ULL));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(raw_full_k_addr + ((int)((unsigned int)(walk_nt0 + n_3) % 4)) * 8) : "memory");
                    int col_45 = lane_4 & 7;
                    int sub_46 = lane_4 >> 3;
                    int token0_47 = 16 * g_1 + sub_46;
                    unsigned int dst_v0 = ((kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_47 * 144 + col_45 * 16)) < (kv_addr + 73712) ? (kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_47 * 144 + col_45 * 16)) : (kv_addr + 73712));
                    unsigned int src_size_4 = (v_bits_a[0] & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0), "l"(v_src_a[0]), "r"(src_size_4), "l"(0x12F0000000000000ULL));
                    unsigned int src_size_5 = (v_bits_a[0] >> 4 & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0 + 576), "l"(v_src_a[1]), "r"(src_size_5), "l"(0x12F0000000000000ULL));
                    unsigned int src_size_6 = (v_bits_a[0] >> 8 & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0 + 1152), "l"(v_src_a[2]), "r"(src_size_6), "l"(0x12F0000000000000ULL));
                    unsigned int src_size_8 = (v_bits_a[0] >> 12 & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0 + 1728), "l"(v_src_a[3]), "r"(src_size_8), "l"(0x12F0000000000000ULL));
                    int col_52 = lane_4 & 7;
                    int sub_53 = lane_4 >> 3;
                    int token0_54 = 16 * (g_1 + 2) + sub_53;
                    unsigned int dst_v0_55 = ((kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_54 * 144 + col_52 * 16)) < (kv_addr + 73712) ? (kv_addr + (unsigned int)(st_2 * 18432 + 9216 + token0_54 * 144 + col_52 * 16)) : (kv_addr + 73712));
                    unsigned int src_size_10 = (v_bits_b[0] & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0_55), "l"(v_src_b[0]), "r"(src_size_10), "l"(0x12F0000000000000ULL));
                    unsigned int src_size_12 = (v_bits_b[0] >> 4 & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0_55 + 576), "l"(v_src_b[1]), "r"(src_size_12), "l"(0x12F0000000000000ULL));
                    unsigned int src_size_13 = (v_bits_b[0] >> 8 & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0_55 + 1152), "l"(v_src_b[2]), "r"(src_size_13), "l"(0x12F0000000000000ULL));
                    unsigned int src_size_14 = (v_bits_b[0] >> 12 & 1) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                        :: "r"(dst_v0_55 + 1728), "l"(v_src_b[3]), "r"(src_size_14), "l"(0x12F0000000000000ULL));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(raw_full_v_addr + ((int)((unsigned int)(walk_nt0 + n_3) % 4)) * 8) : "memory");
                    if (n_3 == 0 && has_next) {
                        if (pi >= 2) {
                            int _stage_q_empty = (int)((unsigned int)(walk_pairs - 2) % 3);
                            int _parity_q_empty = (int)((unsigned int)(walk_pairs - 2) / 3 % 2);
                            mbarrier_wait(q_empty_addr + (_stage_q_empty) * 8, _parity_q_empty);
                        }
                        int nbuf = (pi + 1) % 3;
                        int landed = pi % 3;
                        long long q_row_1 = nrow_i64_3 * q_stride_row;
                        unsigned long long q_u64 = (unsigned long long)q;
                        int e_0 = gtid;
                        int head_0 = e_0 >> 5;
                        int c_0 = e_0 & 31;
                        int panel_0 = 4 * nbuf + (c_0 >> 3);
                        int cc_0 = c_0 & 7;
                        bool in_group_1_0 = head_0 < group_size;
                        int safe_head_1_0 = ((head_0) < (group_size - 1) ? (head_0) : (group_size - 1));
                        long long q_src_1_8 = q_row_1 + (long long)(nfirst_3 + safe_head_1_0) * q_stride_head + (long long)(8 * c_0);
                        unsigned int src_size_15 = ((in_group_1_0) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_0 * 2048) + (unsigned int)(head_0 * 128 + 16 * cc_0 ^ (head_0 * 128 + 16 * cc_0 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_8 * 2), "r"(src_size_15));
                        int e_2 = gtid + 64;
                        int head_1 = e_2 >> 5;
                        int c_2 = e_2 & 31;
                        int panel_2 = 4 * nbuf + (c_2 >> 3);
                        int cc_2 = c_2 & 7;
                        bool in_group_1_1 = head_1 < group_size;
                        int safe_head_1_1 = ((head_1) < (group_size - 1) ? (head_1) : (group_size - 1));
                        long long q_src_1_9 = q_row_1 + (long long)(nfirst_3 + safe_head_1_1) * q_stride_head + (long long)(8 * c_2);
                        unsigned int src_size_16 = ((in_group_1_1) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_2 * 2048) + (unsigned int)(head_1 * 128 + 16 * cc_2 ^ (head_1 * 128 + 16 * cc_2 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_9 * 2), "r"(src_size_16));
                        int e_3 = gtid + 128;
                        int head_3 = e_3 >> 5;
                        int c_3 = e_3 & 31;
                        int panel_3 = 4 * nbuf + (c_3 >> 3);
                        int cc_3 = c_3 & 7;
                        bool in_group_1_2 = head_3 < group_size;
                        int safe_head_1_2 = ((head_3) < (group_size - 1) ? (head_3) : (group_size - 1));
                        long long q_src_1_10 = q_row_1 + (long long)(nfirst_3 + safe_head_1_2) * q_stride_head + (long long)(8 * c_3);
                        unsigned int src_size_17 = ((in_group_1_2) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_3 * 2048) + (unsigned int)(head_3 * 128 + 16 * cc_3 ^ (head_3 * 128 + 16 * cc_3 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_10 * 2), "r"(src_size_17));
                        int e_4 = gtid + 192;
                        int head_4 = e_4 >> 5;
                        int c_4 = e_4 & 31;
                        int panel_4 = 4 * nbuf + (c_4 >> 3);
                        int cc_4 = c_4 & 7;
                        bool in_group_1_3 = head_4 < group_size;
                        int safe_head_1_3 = ((head_4) < (group_size - 1) ? (head_4) : (group_size - 1));
                        long long q_src_1_11 = q_row_1 + (long long)(nfirst_3 + safe_head_1_3) * q_stride_head + (long long)(8 * c_4);
                        unsigned int src_size_18 = ((in_group_1_3) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_4 * 2048) + (unsigned int)(head_4 * 128 + 16 * cc_4 ^ (head_4 * 128 + 16 * cc_4 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_11 * 2), "r"(src_size_18));
                        int e_5 = gtid + 256;
                        int head_5 = e_5 >> 5;
                        int c_5 = e_5 & 31;
                        int panel_5 = 4 * nbuf + (c_5 >> 3);
                        int cc_5 = c_5 & 7;
                        bool in_group_1_4 = head_5 < group_size;
                        int safe_head_1_4 = ((head_5) < (group_size - 1) ? (head_5) : (group_size - 1));
                        long long q_src_1_12 = q_row_1 + (long long)(nfirst_3 + safe_head_1_4) * q_stride_head + (long long)(8 * c_5);
                        unsigned int src_size_19 = ((in_group_1_4) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_5 * 2048) + (unsigned int)(head_5 * 128 + 16 * cc_5 ^ (head_5 * 128 + 16 * cc_5 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_12 * 2), "r"(src_size_19));
                        int e_6 = gtid + 320;
                        int head_6 = e_6 >> 5;
                        int c_6 = e_6 & 31;
                        int panel_6 = 4 * nbuf + (c_6 >> 3);
                        int cc_6 = c_6 & 7;
                        bool in_group_1_5 = head_6 < group_size;
                        int safe_head_1_5 = ((head_6) < (group_size - 1) ? (head_6) : (group_size - 1));
                        long long q_src_1_13 = q_row_1 + (long long)(nfirst_3 + safe_head_1_5) * q_stride_head + (long long)(8 * c_6);
                        unsigned int src_size_20 = ((in_group_1_5) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_6 * 2048) + (unsigned int)(head_6 * 128 + 16 * cc_6 ^ (head_6 * 128 + 16 * cc_6 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_13 * 2), "r"(src_size_20));
                        int e_7 = gtid + 384;
                        int head_7 = e_7 >> 5;
                        int c_7 = e_7 & 31;
                        int panel_7 = 4 * nbuf + (c_7 >> 3);
                        int cc_7 = c_7 & 7;
                        bool in_group_1_6 = head_7 < group_size;
                        int safe_head_1_6 = ((head_7) < (group_size - 1) ? (head_7) : (group_size - 1));
                        long long q_src_1_14 = q_row_1 + (long long)(nfirst_3 + safe_head_1_6) * q_stride_head + (long long)(8 * c_7);
                        unsigned int src_size_21 = ((in_group_1_6) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_7 * 2048) + (unsigned int)(head_7 * 128 + 16 * cc_7 ^ (head_7 * 128 + 16 * cc_7 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_14 * 2), "r"(src_size_21));
                        int e_8 = gtid + 448;
                        int head_8 = e_8 >> 5;
                        int c_8 = e_8 & 31;
                        int panel_8 = 4 * nbuf + (c_8 >> 3);
                        int cc_8 = c_8 & 7;
                        bool in_group_1_7 = head_8 < group_size;
                        int safe_head_1_7 = ((head_8) < (group_size - 1) ? (head_8) : (group_size - 1));
                        long long q_src_1_15 = q_row_1 + (long long)(nfirst_3 + safe_head_1_7) * q_stride_head + (long long)(8 * c_8);
                        unsigned int src_size_22 = ((in_group_1_7) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_8 * 2048) + (unsigned int)(head_8 * 128 + 16 * cc_8 ^ (head_8 * 128 + 16 * cc_8 >> 7 & 7) << 4))), "l"(q_u64 + (unsigned long long)q_src_1_15 * 2), "r"(src_size_22));
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(q_landed_addr + ((int)((unsigned int)walk_pairs % 3)) * 8) : "memory");
                    }
                    idx[0] = idx[1];
                    idx[1] = idx[2];
                    idx[2] = idx[3];
                    idx[3] = logical_2_0;
                    phys[0] = phys[1];
                    phys[1] = phys[2];
                    local[0] = local[1];
                    local[1] = local[2];
                }
                if (has_next) {
                    if (ntiles == 0) {
                        if (pi >= 2) {
                            int _stage_q_empty_1 = (int)((unsigned int)(walk_pairs - 2) % 3);
                            int _parity_q_empty_1 = (int)((unsigned int)(walk_pairs - 2) / 3 % 2);
                            mbarrier_wait(q_empty_addr + (_stage_q_empty_1) * 8, _parity_q_empty_1);
                        }
                        int nbuf_1 = (pi + 1) % 3;
                        int landed_1 = pi % 3;
                        long long q_row_2 = nrow_i64_3 * q_stride_row;
                        unsigned long long q_u64_1 = (unsigned long long)q;
                        int e_9 = gtid;
                        int head_9 = e_9 >> 5;
                        int c_9 = e_9 & 31;
                        int panel_9 = 4 * nbuf_1 + (c_9 >> 3);
                        int cc_9 = c_9 & 7;
                        bool in_group_1_8 = head_9 < group_size;
                        int safe_head_1_8 = ((head_9) < (group_size - 1) ? (head_9) : (group_size - 1));
                        long long q_src_1_16 = q_row_2 + (long long)(nfirst_3 + safe_head_1_8) * q_stride_head + (long long)(8 * c_9);
                        unsigned int src_size_23 = ((in_group_1_8) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_9 * 2048) + (unsigned int)(head_9 * 128 + 16 * cc_9 ^ (head_9 * 128 + 16 * cc_9 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_16 * 2), "r"(src_size_23));
                        int e_10 = gtid + 64;
                        int head_10 = e_10 >> 5;
                        int c_10 = e_10 & 31;
                        int panel_10 = 4 * nbuf_1 + (c_10 >> 3);
                        int cc_10 = c_10 & 7;
                        bool in_group_1_9 = head_10 < group_size;
                        int safe_head_1_9 = ((head_10) < (group_size - 1) ? (head_10) : (group_size - 1));
                        long long q_src_1_17 = q_row_2 + (long long)(nfirst_3 + safe_head_1_9) * q_stride_head + (long long)(8 * c_10);
                        unsigned int src_size_24 = ((in_group_1_9) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_10 * 2048) + (unsigned int)(head_10 * 128 + 16 * cc_10 ^ (head_10 * 128 + 16 * cc_10 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_17 * 2), "r"(src_size_24));
                        int e_11 = gtid + 128;
                        int head_11 = e_11 >> 5;
                        int c_11 = e_11 & 31;
                        int panel_11 = 4 * nbuf_1 + (c_11 >> 3);
                        int cc_11 = c_11 & 7;
                        bool in_group_1_10 = head_11 < group_size;
                        int safe_head_1_10 = ((head_11) < (group_size - 1) ? (head_11) : (group_size - 1));
                        long long q_src_1_18 = q_row_2 + (long long)(nfirst_3 + safe_head_1_10) * q_stride_head + (long long)(8 * c_11);
                        unsigned int src_size_25 = ((in_group_1_10) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_11 * 2048) + (unsigned int)(head_11 * 128 + 16 * cc_11 ^ (head_11 * 128 + 16 * cc_11 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_18 * 2), "r"(src_size_25));
                        int e_12 = gtid + 192;
                        int head_12 = e_12 >> 5;
                        int c_12 = e_12 & 31;
                        int panel_12 = 4 * nbuf_1 + (c_12 >> 3);
                        int cc_12 = c_12 & 7;
                        bool in_group_1_11 = head_12 < group_size;
                        int safe_head_1_11 = ((head_12) < (group_size - 1) ? (head_12) : (group_size - 1));
                        long long q_src_1_19 = q_row_2 + (long long)(nfirst_3 + safe_head_1_11) * q_stride_head + (long long)(8 * c_12);
                        unsigned int src_size_26 = ((in_group_1_11) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_12 * 2048) + (unsigned int)(head_12 * 128 + 16 * cc_12 ^ (head_12 * 128 + 16 * cc_12 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_19 * 2), "r"(src_size_26));
                        int e_13 = gtid + 256;
                        int head_13 = e_13 >> 5;
                        int c_13 = e_13 & 31;
                        int panel_13 = 4 * nbuf_1 + (c_13 >> 3);
                        int cc_13 = c_13 & 7;
                        bool in_group_1_12 = head_13 < group_size;
                        int safe_head_1_12 = ((head_13) < (group_size - 1) ? (head_13) : (group_size - 1));
                        long long q_src_1_20 = q_row_2 + (long long)(nfirst_3 + safe_head_1_12) * q_stride_head + (long long)(8 * c_13);
                        unsigned int src_size_27 = ((in_group_1_12) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_13 * 2048) + (unsigned int)(head_13 * 128 + 16 * cc_13 ^ (head_13 * 128 + 16 * cc_13 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_20 * 2), "r"(src_size_27));
                        int e_14 = gtid + 320;
                        int head_14 = e_14 >> 5;
                        int c_14 = e_14 & 31;
                        int panel_14 = 4 * nbuf_1 + (c_14 >> 3);
                        int cc_14 = c_14 & 7;
                        bool in_group_1_13 = head_14 < group_size;
                        int safe_head_1_13 = ((head_14) < (group_size - 1) ? (head_14) : (group_size - 1));
                        long long q_src_1_21 = q_row_2 + (long long)(nfirst_3 + safe_head_1_13) * q_stride_head + (long long)(8 * c_14);
                        unsigned int src_size_28 = ((in_group_1_13) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_14 * 2048) + (unsigned int)(head_14 * 128 + 16 * cc_14 ^ (head_14 * 128 + 16 * cc_14 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_21 * 2), "r"(src_size_28));
                        int e_15 = gtid + 384;
                        int head_15 = e_15 >> 5;
                        int c_15 = e_15 & 31;
                        int panel_15 = 4 * nbuf_1 + (c_15 >> 3);
                        int cc_15 = c_15 & 7;
                        bool in_group_1_14 = head_15 < group_size;
                        int safe_head_1_14 = ((head_15) < (group_size - 1) ? (head_15) : (group_size - 1));
                        long long q_src_1_22 = q_row_2 + (long long)(nfirst_3 + safe_head_1_14) * q_stride_head + (long long)(8 * c_15);
                        unsigned int src_size_29 = ((in_group_1_14) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_15 * 2048) + (unsigned int)(head_15 * 128 + 16 * cc_15 ^ (head_15 * 128 + 16 * cc_15 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_22 * 2), "r"(src_size_29));
                        int e_16 = gtid + 448;
                        int head_16 = e_16 >> 5;
                        int c_16 = e_16 & 31;
                        int panel_16 = 4 * nbuf_1 + (c_16 >> 3);
                        int cc_16 = c_16 & 7;
                        bool in_group_1_15 = head_16 < group_size;
                        int safe_head_1_15 = ((head_16) < (group_size - 1) ? (head_16) : (group_size - 1));
                        long long q_src_1_23 = q_row_2 + (long long)(nfirst_3 + safe_head_1_15) * q_stride_head + (long long)(8 * c_16);
                        unsigned int src_size_30 = ((in_group_1_15) ? 16 : 0);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_q_addr + (unsigned int)(panel_16 * 2048) + (unsigned int)(head_16 * 128 + 16 * cc_16 ^ (head_16 * 128 + 16 * cc_16 >> 7 & 7) << 4))), "l"(q_u64_1 + (unsigned long long)q_src_1_23 * 2), "r"(src_size_30));
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(q_landed_addr + ((int)((unsigned int)walk_pairs % 3)) * 8) : "memory");
                    }
                    int _stage_q_landed = (int)((unsigned int)walk_pairs % 3);
                    int _parity_q_landed = (int)((unsigned int)walk_pairs / 3 % 2);
                    mbarrier_wait(q_landed_addr + (_stage_q_landed) * 8, _parity_q_landed);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    int nbuf_2 = (pi + 1) % 3;
                    if (elect_sync()) {
                        mbarrier_arrive(q_full_addr + ((int)((unsigned int)(walk_pairs + 1) % 3)) * 8);
                    }
                }
                walk_nt0 += ntiles;
                walk_pairs += 1;
#if __CUDA_ARCH__ == 1070
                int _min_19 = ((ntiles) < (4) ? (ntiles) : (4));
                pre_idx = _min_19;
                int _max_21 = ((ntiles - 2) > (0) ? (ntiles - 2) : (0));
                int _min_20 = ((_max_21) < (2) ? (_max_21) : (2));
                pre_blk = _min_20;
#else
                int _min_20 = ((ntiles) < (4) ? (ntiles) : (4));
                pre_idx = _min_20;
                int _max_22 = ((ntiles - 2) > (0) ? (ntiles - 2) : (0));
                int _min_21 = ((_max_22) < (2) ? (_max_22) : (2));
                pre_blk = _min_21;
#endif
            }
        }
    }
    // ---- Role: mma_s ----
    if (warp == 14) {
        { // mma_s_main
            int nt0_4 = 0;
            int mma_pairs = 0;
            int req_cur_4 = request;
            int vc_cur_4 = valid_count;
            #pragma unroll 1
            for (int P_4 = pair0; P_4 < pair1 + 1; P_4++) {
#if __CUDA_ARCH__ == 1070
                int row_2 = P_4 / hkv;
                int kv_head_2 = P_4 - row_2 * hkv;
                long long row_i64_2 = (long long)row_2;
                int first_head_2 = kv_head_2 * group_size;
                long long indices_row_2 = row_i64_2 * indices_stride_row;
                long long pslot_2 = (long long)P_4 * tiles_i64;
                long long _max_0_5 = ((slot0 - pslot_2) > (0) ? (slot0 - pslot_2) : (0));
                int tile0_2 = (int)_max_0_5;
                long long _min_1_5 = ((slot1 - pslot_2) < (tiles_i64) ? (slot1 - pslot_2) : (tiles_i64));
                int tile1_2 = (int)_min_1_5;
                int owner0_2 = (int)(((pslot_2 + 1) * ctas_i64 - 1) / slots);
                int cta_3 = blockIdx.x;
                int seg_2 = cta_3 - owner0_2;
                bool last_2 = slot1 >= pslot_2 + tiles_i64;
                bool request_ok_2 = req_cur_4 >= 0 && req_cur_4 < num_requests;
#else
                int row_3 = P_4 / hkv;
                int kv_head_3 = P_4 - row_3 * hkv;
                long long row_i64_3 = (long long)row_3;
                int first_head_3 = kv_head_3 * group_size;
                long long indices_row_3 = row_i64_3 * indices_stride_row;
                long long pslot_3 = (long long)P_4 * tiles_i64;
                long long _max_0_6 = ((slot0 - pslot_3) > (0) ? (slot0 - pslot_3) : (0));
                int tile0_3 = (int)_max_0_6;
                long long _min_1_6 = ((slot1 - pslot_3) < (tiles_i64) ? (slot1 - pslot_3) : (tiles_i64));
                int tile1_3 = (int)_min_1_6;
                int owner0_3 = (int)(((pslot_3 + 1) * ctas_i64 - 1) / slots);
                int cta_4 = blockIdx.x;
                int seg_3 = cta_4 - owner0_3;
                bool last_3 = slot1 >= pslot_3 + tiles_i64;
                bool request_ok_3 = req_cur_4 >= 0 && req_cur_4 < num_requests;
#endif
                int _max_2_3 = ((req_cur_4) > (0) ? (req_cur_4) : (0));
#if __CUDA_ARCH__ == 1070
                int safe_request_2 = ((_max_2_3) < (num_requests - 1) ? (_max_2_3) : (num_requests - 1));
                long long safe_request_row_2 = (long long)safe_request_2 * table_stride_req;
#else
                int safe_request_3 = ((_max_2_3) < (num_requests - 1) ? (_max_2_3) : (num_requests - 1));
                long long safe_request_row_3 = (long long)safe_request_3 * table_stride_req;
#endif
                int _min_4_4 = ((vc_cur_4) < (topk) ? (vc_cur_4) : (topk));
#if __CUDA_ARCH__ == 1070
                int _min_3_4 = ((num_tiles) < ((_min_4_4 + 63) / 64) ? (num_tiles) : ((_min_4_4 + 63) / 64));
                int tile_end_2 = ((_min_3_4) < (tile1_2) ? (_min_3_4) : (tile1_2));
                int n_tiles_2 = ((tile_end_2 - tile0_2) > (0) ? (tile_end_2 - tile0_2) : (0));
#else
                int _min_3_3 = ((num_tiles) < ((_min_4_4 + 63) / 64) ? (num_tiles) : ((_min_4_4 + 63) / 64));
                int tile_end_3 = ((_min_3_3) < (tile1_3) ? (_min_3_3) : (tile1_3));
                int n_tiles_3 = ((tile_end_3 - tile0_3) > (0) ? (tile_end_3 - tile0_3) : (0));
#endif
                int nxt_4 = ((P_4 + 1) < (pair1) ? (P_4 + 1) : (pair1));
                int nrow_4 = nxt_4 / hkv;
                long long nrow_i64_4 = (long long)nrow_4;
                int nfirst_4 = (nxt_4 - nrow_4 * hkv) * group_size;
                long long nidx_row_4 = nrow_i64_4 * indices_stride_row;
#if __CUDA_ARCH__ == 1070
                int request_6 = token_to_req[nrow_4];
                int valid_count_6 = indices[nidx_row_4 + (long long)topk];
#else
                int request_5 = token_to_req[nrow_4];
                int valid_count_5 = indices[nidx_row_4 + (long long)topk];
#endif
                int pi_1 = P_4 - pair0;
                int qpanel0 = 4 * (pi_1 % 3);
                int _stage_q_full = (int)((unsigned int)mma_pairs % 3);
                int _parity_q_full = (int)((unsigned int)mma_pairs / 3 % 2);
                mbarrier_wait(q_full_addr + (_stage_q_full) * 8, _parity_q_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
#if __CUDA_ARCH__ == 1070
                for (int n_4 = 0; n_4 < n_tiles_2; n_4++) {
#else
                for (int n_4 = 0; n_4 < n_tiles_3; n_4++) {
#endif
                    int nt_4 = nt0_4 + n_4;
                    int par_1 = nt_4 & 1;
                    int _stage_k_full = (int)((unsigned int)(nt0_4 + n_4) % 2);
                    int _parity_k_full = (int)((unsigned int)(nt0_4 + n_4) / 2 % 2);
                    mbarrier_wait(k_full_addr + (_stage_k_full) * 8, _parity_k_full);
                    int _stage_s_empty = (int)((unsigned int)(nt0_4 + n_4) % 2);
                    int _parity_s_empty = (int)((unsigned int)(nt0_4 + n_4) / 2 % 2) ^ 1;
                    mbarrier_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (par_1 == 0) {
                        uint32_t _elected_5 = elect_sync();
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0) * 128;
#else
                        int _smem_desc_lo_6 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0) * 128;
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
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_6), "r"(tmem_tmem_k), "r"(0), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_7 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0 + 1) * 128;
#else
                        int _smem_desc_lo_7 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0 + 1) * 128;
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
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_7), "r"(tmem_tmem_k + 32), "r"(1), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_8 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0 + 2) * 128;
#else
                        int _smem_desc_lo_8 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0 + 2) * 128;
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
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_8), "r"(tmem_tmem_k + 64), "r"(1), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_9 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0 + 3) * 128;
#else
                        int _smem_desc_lo_9 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0 + 3) * 128;
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
                    :: "r"(tmem_tmem_s), "r"(_smem_desc_lo_9), "r"(tmem_tmem_k + 96), "r"(1), "r"(_elected_5) : "memory");
                        tcgen05_commit_leader(k_empty_addr + ((int)((unsigned int)(nt0_4 + n_4) % 2)) * 8, _elected_5);
                        tcgen05_commit_leader(s_full_addr + ((int)((unsigned int)(nt0_4 + n_4) % 2)) * 8, _elected_5);
                    } else {
                        uint32_t _elected_10 = elect_sync();
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_11 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0) * 128;
#else
                        int _smem_desc_lo_11 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0) * 128;
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
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_11), "r"(tmem_tmem_k + 1048576), "r"(0), "r"(_elected_10) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_12 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0 + 1) * 128;
#else
                        int _smem_desc_lo_12 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0 + 1) * 128;
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
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_12), "r"(tmem_tmem_k + 1048608), "r"(1), "r"(_elected_10) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_13 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0 + 2) * 128;
#else
                        int _smem_desc_lo_13 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0 + 2) * 128;
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
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_13), "r"(tmem_tmem_k + 1048640), "r"(1), "r"(_elected_10) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_14 = (((smem_q_addr) >> 4) & 0x7FFF) + (qpanel0 + 3) * 128;
#else
                        int _smem_desc_lo_14 = (((smem_q_addr) >> 4) & 0x3FFF) + (qpanel0 + 3) * 128;
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
                    :: "r"((tmem_tmem_s + (1048592))), "r"(_smem_desc_lo_14), "r"(tmem_tmem_k + 1048672), "r"(1), "r"(_elected_10) : "memory");
                        tcgen05_commit_leader(k_empty_addr + ((int)((unsigned int)(nt0_4 + n_4) % 2)) * 8, _elected_10);
                        tcgen05_commit_leader(s_full_addr + ((int)((unsigned int)(nt0_4 + n_4) % 2)) * 8, _elected_10);
                    }
                }
                int qb = pi_1 % 3;
                if (elect_sync()) {
                    if (qb == 0) {
                        tcgen05_commit(q_empty_addr + ((int)((unsigned int)mma_pairs % 3)) * 8);
                    } else if (qb == 1) {
                        tcgen05_commit(q_empty_addr + ((int)((unsigned int)mma_pairs % 3)) * 8);
                    } else {
                        tcgen05_commit(q_empty_addr + ((int)((unsigned int)mma_pairs % 3)) * 8);
                    }
                }
#if __CUDA_ARCH__ == 1070
                nt0_4 += n_tiles_2;
#else
                nt0_4 += n_tiles_3;
#endif
                mma_pairs += 1;
#if __CUDA_ARCH__ == 1070
                req_cur_4 = request_6;
                vc_cur_4 = valid_count_6;
#else
                req_cur_4 = request_5;
                vc_cur_4 = valid_count_5;
#endif
            }
        }
    }
    // ---- Role: mma_pv ----
    if (warp == 15) {
        { // mma_pv_main
            int nt0_5 = 0;
            int req_cur_5 = request;
            int vc_cur_5 = valid_count;
            #pragma unroll 1
            for (int P_5 = pair0; P_5 < pair1 + 1; P_5++) {
#if __CUDA_ARCH__ == 1070
                int row_3 = P_5 / hkv;
                int kv_head_3 = P_5 - row_3 * hkv;
                long long row_i64_3 = (long long)row_3;
                int first_head_3 = kv_head_3 * group_size;
                long long indices_row_3 = row_i64_3 * indices_stride_row;
                long long pslot_3 = (long long)P_5 * tiles_i64;
                long long _max_0_6 = ((slot0 - pslot_3) > (0) ? (slot0 - pslot_3) : (0));
                int tile0_3 = (int)_max_0_6;
                long long _min_1_6 = ((slot1 - pslot_3) < (tiles_i64) ? (slot1 - pslot_3) : (tiles_i64));
                int tile1_3 = (int)_min_1_6;
                int owner0_3 = (int)(((pslot_3 + 1) * ctas_i64 - 1) / slots);
                int cta_4 = blockIdx.x;
                int seg_3 = cta_4 - owner0_3;
                bool last_3 = slot1 >= pslot_3 + tiles_i64;
                bool request_ok_3 = req_cur_5 >= 0 && req_cur_5 < num_requests;
#else
                int row_4 = P_5 / hkv;
                int kv_head_4 = P_5 - row_4 * hkv;
                long long row_i64_4 = (long long)row_4;
                int first_head_4 = kv_head_4 * group_size;
                long long indices_row_4 = row_i64_4 * indices_stride_row;
                long long pslot_4 = (long long)P_5 * tiles_i64;
                long long _max_0_7 = ((slot0 - pslot_4) > (0) ? (slot0 - pslot_4) : (0));
                int tile0_4 = (int)_max_0_7;
                long long _min_1_7 = ((slot1 - pslot_4) < (tiles_i64) ? (slot1 - pslot_4) : (tiles_i64));
                int tile1_4 = (int)_min_1_7;
                int owner0_4 = (int)(((pslot_4 + 1) * ctas_i64 - 1) / slots);
                int cta_5 = blockIdx.x;
                int seg_4 = cta_5 - owner0_4;
                bool last_4 = slot1 >= pslot_4 + tiles_i64;
                bool request_ok_4 = req_cur_5 >= 0 && req_cur_5 < num_requests;
#endif
                int _max_2_4 = ((req_cur_5) > (0) ? (req_cur_5) : (0));
#if __CUDA_ARCH__ == 1070
                int safe_request_3 = ((_max_2_4) < (num_requests - 1) ? (_max_2_4) : (num_requests - 1));
                long long safe_request_row_3 = (long long)safe_request_3 * table_stride_req;
#else
                int safe_request_4 = ((_max_2_4) < (num_requests - 1) ? (_max_2_4) : (num_requests - 1));
                long long safe_request_row_4 = (long long)safe_request_4 * table_stride_req;
#endif
                int _min_4_5 = ((vc_cur_5) < (topk) ? (vc_cur_5) : (topk));
#if __CUDA_ARCH__ == 1070
                int _min_3_5 = ((num_tiles) < ((_min_4_5 + 63) / 64) ? (num_tiles) : ((_min_4_5 + 63) / 64));
                int tile_end_3 = ((_min_3_5) < (tile1_3) ? (_min_3_5) : (tile1_3));
                int n_tiles_3 = ((tile_end_3 - tile0_3) > (0) ? (tile_end_3 - tile0_3) : (0));
#else
                int _min_3_4 = ((num_tiles) < ((_min_4_5 + 63) / 64) ? (num_tiles) : ((_min_4_5 + 63) / 64));
                int tile_end_4 = ((_min_3_4) < (tile1_4) ? (_min_3_4) : (tile1_4));
                int n_tiles_4 = ((tile_end_4 - tile0_4) > (0) ? (tile_end_4 - tile0_4) : (0));
#endif
                int nxt_5 = ((P_5 + 1) < (pair1) ? (P_5 + 1) : (pair1));
                int nrow_5 = nxt_5 / hkv;
                long long nrow_i64_5 = (long long)nrow_5;
                int nfirst_5 = (nxt_5 - nrow_5 * hkv) * group_size;
                long long nidx_row_5 = nrow_i64_5 * indices_stride_row;
#if __CUDA_ARCH__ == 1070
                int request_7 = token_to_req[nrow_5];
                int valid_count_7 = indices[nidx_row_5 + (long long)topk];
#else
                int request_6 = token_to_req[nrow_5];
                int valid_count_6 = indices[nidx_row_5 + (long long)topk];
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1070
                for (int n_5 = 0; n_5 < n_tiles_3; n_5++) {
#else
                for (int n_5 = 0; n_5 < n_tiles_4; n_5++) {
#endif
                    int nt_5 = nt0_5 + n_5;
                    int vs_1 = nt_5 % 2;
                    int ps = nt_5 % 2;
                    bool first_pv = n_5 == 0;
                    int _stage_v_full = (int)((unsigned int)(nt0_5 + n_5) % 2);
                    int _parity_v_full = (int)((unsigned int)(nt0_5 + n_5) / 2 % 2);
                    mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                    int _stage_p_full = (int)((unsigned int)(nt0_5 + n_5) % 2);
                    int _parity_p_full = (int)((unsigned int)(nt0_5 + n_5) / 2 % 2);
                    mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_5 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6_1 = ((((smem_vt_addr) >> 4) & 0x7FFF) | 0x2000000) + (2 * vs_1) * 1024;
                    int _smem_desc_lo_7_1 = ((((smem_p_addr) >> 4) & 0x7FFF) | 0x800000) + (ps) * 128;
#else
                    int _smem_desc_lo_6_1 = ((((smem_vt_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * vs_1) * 1024;
                    int _smem_desc_lo_7_1 = ((((smem_p_addr) >> 4) & 0x3FFF) | 0x800000) + (ps) * 128;
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
                    :: "r"(_smem_desc_lo_6_1), "r"(_smem_desc_lo_7_1), "r"(tmem_tmem_o), "r"(((first_pv) ? 0 : 1)), "r"(_elected_5) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8_1 = ((((smem_vt_addr) >> 4) & 0x7FFF) | 0x2000000) + (2 * vs_1 + 1) * 1024;
                    int _smem_desc_lo_9_1 = ((((smem_p_addr) >> 4) & 0x7FFF) | 0x800000) + (ps) * 128;
#else
                    int _smem_desc_lo_8_1 = ((((smem_vt_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * vs_1 + 1) * 1024;
                    int _smem_desc_lo_9_1 = ((((smem_p_addr) >> 4) & 0x3FFF) | 0x800000) + (ps) * 128;
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
                    :: "r"(_smem_desc_lo_8_1), "r"(_smem_desc_lo_9_1), "r"((tmem_tmem_o + (16))), "r"(((first_pv) ? 0 : 1)), "r"(_elected_5) : "memory");
                    tcgen05_commit_leader(v_empty_addr + ((int)((unsigned int)(nt0_5 + n_5) % 2)) * 8, _elected_5);
                    tcgen05_commit_leader(pv_done_addr + ((int)((unsigned int)(nt0_5 + n_5) % 2)) * 8, _elected_5);
                }
#if __CUDA_ARCH__ == 1070
                nt0_5 += n_tiles_3;
                req_cur_5 = request_7;
                vc_cur_5 = valid_count_7;
#else
                nt0_5 += n_tiles_4;
                req_cur_5 = request_6;
                vc_cur_5 = valid_count_6;
#endif
            }
            mbarrier_wait(tmem_dead_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }

    // Cleanup
}

} // extern "C"
