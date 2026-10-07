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
#define SMEM_RAKE_W_OFF 0
#define SMEM_RAKE_W_STAGE_BYTES 4224
#define SMEM_RAKE_W_STRIDE 4224
#define SMEM_RAKE_N_OFF 4224
#define SMEM_RAKE_N_STAGE_BYTES 4224
#define SMEM_RAKE_N_STRIDE 4224
#define SMEM_RAKE_A_OFF 8448
#define SMEM_RAKE_A_STAGE_BYTES 4224
#define SMEM_RAKE_A_STRIDE 4224
#define SMEM_AGG_OFF 12672
#define SMEM_AGG_STAGE_BYTES 16
#define SMEM_AGG_STRIDE 16
#define SMEM_SUM_W_OFF 12688
#define SMEM_SUM_W_STAGE_BYTES 128
#define SMEM_SUM_W_STRIDE 128
#define SMEM_SUM_A_OFF 12816
#define SMEM_SUM_A_STAGE_BYTES 128
#define SMEM_SUM_A_STRIDE 128
#define SMEM_USE_WIDE_BUF_OFF 12944
#define SMEM_USE_WIDE_BUF_STAGE_BYTES 16
#define SMEM_USE_WIDE_BUF_STRIDE 16
#define SMEM_TOTAL 13056
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_mxfp4_situ_moe_d274b75887a3b8524df0(int* __restrict__ mn_limit, int* __restrict__ num_groups_ptr, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ narrow_list, int* __restrict__ narrow_count, int* __restrict__ all_list, int* __restrict__ all_count, int group_rows, int narrow_tile, int wide_min_rows, int wide_min_permille)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
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
    int* rake_w = reinterpret_cast<int*>(smem_raw + 0);
    const int rake_w_addr = smem + 0;
    int* rake_n = reinterpret_cast<int*>(smem_raw + 4224);
    const int rake_n_addr = smem + 4224;
    int* rake_a = reinterpret_cast<int*>(smem_raw + 8448);
    const int rake_a_addr = smem + 8448;
    int* agg = reinterpret_cast<int*>(smem_raw + 12672);
    const int agg_addr = smem + 12672;
    int* sum_w = reinterpret_cast<int*>(smem_raw + 12688);
    const int sum_w_addr = smem + 12688;
    int* sum_a = reinterpret_cast<int*>(smem_raw + 12816);
    const int sum_a_addr = smem + 12816;
    int* use_wide_buf = reinterpret_cast<int*>(smem_raw + 12944);
    const int use_wide_buf_addr = smem + 12944;

    // === Task calls (dependency order) ===
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int num_groups = num_groups_ptr[0];
    int sub = group_rows / narrow_tile;
    int wide_min = wide_min_rows;
    if (wide_min_permille > 0) {
        int wide_rows = 0;
        int all_rows = 0;
        #pragma unroll 1
        for (int g0 = tid_0; g0 < num_groups; g0 += 1024) {
            int limit = mn_limit[g0];
            int _min_0 = ((group_rows) < (limit - g0 * group_rows) ? (group_rows) : (limit - g0 * group_rows));
            int rows = _min_0;
            int _max_0 = ((rows) > (0) ? (rows) : (0));
            rows = _max_0;
            int rows0 = rows;
            all_rows = all_rows + rows0;
            if (rows0 > wide_min_rows) {
                wide_rows = wide_rows + rows0;
            }
        }
        int acc = wide_rows;
        int _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, acc, 1, 32);
        int other = _shfl_down_0;
        acc = acc + other;
        int _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, acc, 2, 32);
        int other_0 = _shfl_down_1;
        acc = acc + other_0;
        int _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, acc, 4, 32);
        int other_1 = _shfl_down_2;
        acc = acc + other_1;
        int _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, acc, 8, 32);
        int other_2 = _shfl_down_3;
        acc = acc + other_2;
        int _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, acc, 16, 32);
        int other_3 = _shfl_down_4;
        acc = acc + other_3;
        int warp_wide = acc;
        int acc_4 = all_rows;
        int _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, acc_4, 1, 32);
        int other_5 = _shfl_down_5;
        acc_4 = acc_4 + other_5;
        int _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, acc_4, 2, 32);
        int other_6 = _shfl_down_6;
        acc_4 = acc_4 + other_6;
        int _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, acc_4, 4, 32);
        int other_7 = _shfl_down_7;
        acc_4 = acc_4 + other_7;
        int _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, acc_4, 8, 32);
        int other_8 = _shfl_down_8;
        acc_4 = acc_4 + other_8;
        int _shfl_down_9 = __shfl_down_sync(0xFFFFFFFF, acc_4, 16, 32);
        int other_9 = _shfl_down_9;
        acc_4 = acc_4 + other_9;
        int warp_all = acc_4;
        if (lane_1 == 0) {
            sum_w[warp_2] = warp_wide;
            sum_a[warp_2] = warp_all;
        }
        __syncthreads();
        if (tid_0 == 0) {
            int total_wide = warp_wide;
            int total_all = warp_all;
            #pragma unroll 1
            for (int wi = 1; wi < 32; wi++) {
                int part_w = sum_w[wi];
                int part_a = sum_a[wi];
                total_wide = total_wide + part_w;
                total_all = total_all + part_a;
            }
            int use_wide = 0;
            if ((long long)total_wide * 1000 >= (long long)total_all * (long long)wide_min_permille) {
                use_wide = 1;
            }
            use_wide_buf[0] = use_wide;
        }
        __syncthreads();
        int use_wide_read = use_wide_buf[0];
        if (use_wide_read == 0) {
            wide_min = group_rows;
        }
    }
    int carry_w = 0;
    int carry_n = 0;
    int carry_a = 0;
    #pragma unroll 1
    for (int base = 0; base < num_groups; base += 1024) {
        int g = base + tid_0;
        int rows_1 = 0;
        if (g < num_groups) {
            int limit_1 = mn_limit[g];
            int _min_1 = ((group_rows) < (limit_1 - g * group_rows) ? (group_rows) : (limit_1 - g * group_rows));
            int rows_0 = _min_1;
            int _max_1 = ((rows_0) > (0) ? (rows_0) : (0));
            rows_0 = _max_1;
            rows_1 = rows_0;
        }
        int mine_all = 0;
        if (rows_1 > 0) {
            mine_all = (rows_1 + narrow_tile - 1) / narrow_tile;
        }
        int mine_wide = 0;
        int mine_narrow = 0;
        if (rows_1 > wide_min) {
            mine_wide = 1;
        } else {
            mine_narrow = mine_all;
        }
        int tid_1 = tid;
        int slot = tid_1 + tid_1 / 32;
        rake_w[slot] = mine_wide;
        rake_n[slot] = mine_narrow;
        rake_a[slot] = mine_all;
        __syncthreads();
        if (tid_1 < 32) {
            int partial = rake_w[tid_1 * 33];
            int item = rake_w[tid_1 * 33 + 1];
            partial = partial + item;
            int item_0 = rake_w[tid_1 * 33 + 2];
            partial = partial + item_0;
            int item_1 = rake_w[tid_1 * 33 + 3];
            partial = partial + item_1;
            int item_2 = rake_w[tid_1 * 33 + 4];
            partial = partial + item_2;
            int item_3 = rake_w[tid_1 * 33 + 5];
            partial = partial + item_3;
            int item_4 = rake_w[tid_1 * 33 + 6];
            partial = partial + item_4;
            int item_5 = rake_w[tid_1 * 33 + 7];
            partial = partial + item_5;
            int item_6 = rake_w[tid_1 * 33 + 8];
            partial = partial + item_6;
            int item_7 = rake_w[tid_1 * 33 + 9];
            partial = partial + item_7;
            int item_8 = rake_w[tid_1 * 33 + 10];
            partial = partial + item_8;
            int item_9 = rake_w[tid_1 * 33 + 11];
            partial = partial + item_9;
            int item_10 = rake_w[tid_1 * 33 + 12];
            partial = partial + item_10;
            int item_11 = rake_w[tid_1 * 33 + 13];
            partial = partial + item_11;
            int item_12 = rake_w[tid_1 * 33 + 14];
            partial = partial + item_12;
            int item_13 = rake_w[tid_1 * 33 + 15];
            partial = partial + item_13;
            int item_14 = rake_w[tid_1 * 33 + 16];
            partial = partial + item_14;
            int item_15 = rake_w[tid_1 * 33 + 17];
            partial = partial + item_15;
            int item_16 = rake_w[tid_1 * 33 + 18];
            partial = partial + item_16;
            int item_17 = rake_w[tid_1 * 33 + 19];
            partial = partial + item_17;
            int item_18 = rake_w[tid_1 * 33 + 20];
            partial = partial + item_18;
            int item_19 = rake_w[tid_1 * 33 + 21];
            partial = partial + item_19;
            int item_20 = rake_w[tid_1 * 33 + 22];
            partial = partial + item_20;
            int item_21 = rake_w[tid_1 * 33 + 23];
            partial = partial + item_21;
            int item_22 = rake_w[tid_1 * 33 + 24];
            partial = partial + item_22;
            int item_23 = rake_w[tid_1 * 33 + 25];
            partial = partial + item_23;
            int item_24 = rake_w[tid_1 * 33 + 26];
            partial = partial + item_24;
            int item_25 = rake_w[tid_1 * 33 + 27];
            partial = partial + item_25;
            int item_26 = rake_w[tid_1 * 33 + 28];
            partial = partial + item_26;
            int item_27 = rake_w[tid_1 * 33 + 29];
            partial = partial + item_27;
            int item_28 = rake_w[tid_1 * 33 + 30];
            partial = partial + item_28;
            int item_29 = rake_w[tid_1 * 33 + 31];
            partial = partial + item_29;
            int acc_1 = partial;
            int lane_30 = lane;
            int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, acc_1, 1, 32);
            int other_4 = _shfl_up_0;
            if (lane_30 >= 1) {
                acc_1 = acc_1 + other_4;
            }
            int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, acc_1, 2, 32);
            int other_31 = _shfl_up_1;
            if (lane_30 >= 2) {
                acc_1 = acc_1 + other_31;
            }
            int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, acc_1, 4, 32);
            int other_32 = _shfl_up_2;
            if (lane_30 >= 4) {
                acc_1 = acc_1 + other_32;
            }
            int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, acc_1, 8, 32);
            int other_33 = _shfl_up_3;
            if (lane_30 >= 8) {
                acc_1 = acc_1 + other_33;
            }
            int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, acc_1, 16, 32);
            int other_34 = _shfl_up_4;
            if (lane_30 >= 16) {
                acc_1 = acc_1 + other_34;
            }
            int inclusive = acc_1;
            int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
            int exclusive = _shfl_up_5;
            int lane_35 = lane;
            if (lane_35 == 0) {
                exclusive = 0;
            }
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
            int aggregate = _shfl_0;
            int running = exclusive;
            int item_i = rake_w[tid_1 * 33];
            rake_w[tid_1 * 33] = running;
            running = running + item_i;
            int item_i_36 = rake_w[tid_1 * 33 + 1];
            rake_w[tid_1 * 33 + 1] = running;
            running = running + item_i_36;
            int item_i_37 = rake_w[tid_1 * 33 + 2];
            rake_w[tid_1 * 33 + 2] = running;
            running = running + item_i_37;
            int item_i_38 = rake_w[tid_1 * 33 + 3];
            rake_w[tid_1 * 33 + 3] = running;
            running = running + item_i_38;
            int item_i_39 = rake_w[tid_1 * 33 + 4];
            rake_w[tid_1 * 33 + 4] = running;
            running = running + item_i_39;
            int item_i_40 = rake_w[tid_1 * 33 + 5];
            rake_w[tid_1 * 33 + 5] = running;
            running = running + item_i_40;
            int item_i_41 = rake_w[tid_1 * 33 + 6];
            rake_w[tid_1 * 33 + 6] = running;
            running = running + item_i_41;
            int item_i_42 = rake_w[tid_1 * 33 + 7];
            rake_w[tid_1 * 33 + 7] = running;
            running = running + item_i_42;
            int item_i_43 = rake_w[tid_1 * 33 + 8];
            rake_w[tid_1 * 33 + 8] = running;
            running = running + item_i_43;
            int item_i_44 = rake_w[tid_1 * 33 + 9];
            rake_w[tid_1 * 33 + 9] = running;
            running = running + item_i_44;
            int item_i_45 = rake_w[tid_1 * 33 + 10];
            rake_w[tid_1 * 33 + 10] = running;
            running = running + item_i_45;
            int item_i_46 = rake_w[tid_1 * 33 + 11];
            rake_w[tid_1 * 33 + 11] = running;
            running = running + item_i_46;
            int item_i_47 = rake_w[tid_1 * 33 + 12];
            rake_w[tid_1 * 33 + 12] = running;
            running = running + item_i_47;
            int item_i_48 = rake_w[tid_1 * 33 + 13];
            rake_w[tid_1 * 33 + 13] = running;
            running = running + item_i_48;
            int item_i_49 = rake_w[tid_1 * 33 + 14];
            rake_w[tid_1 * 33 + 14] = running;
            running = running + item_i_49;
            int item_i_50 = rake_w[tid_1 * 33 + 15];
            rake_w[tid_1 * 33 + 15] = running;
            running = running + item_i_50;
            int item_i_51 = rake_w[tid_1 * 33 + 16];
            rake_w[tid_1 * 33 + 16] = running;
            running = running + item_i_51;
            int item_i_52 = rake_w[tid_1 * 33 + 17];
            rake_w[tid_1 * 33 + 17] = running;
            running = running + item_i_52;
            int item_i_53 = rake_w[tid_1 * 33 + 18];
            rake_w[tid_1 * 33 + 18] = running;
            running = running + item_i_53;
            int item_i_54 = rake_w[tid_1 * 33 + 19];
            rake_w[tid_1 * 33 + 19] = running;
            running = running + item_i_54;
            int item_i_55 = rake_w[tid_1 * 33 + 20];
            rake_w[tid_1 * 33 + 20] = running;
            running = running + item_i_55;
            int item_i_56 = rake_w[tid_1 * 33 + 21];
            rake_w[tid_1 * 33 + 21] = running;
            running = running + item_i_56;
            int item_i_57 = rake_w[tid_1 * 33 + 22];
            rake_w[tid_1 * 33 + 22] = running;
            running = running + item_i_57;
            int item_i_58 = rake_w[tid_1 * 33 + 23];
            rake_w[tid_1 * 33 + 23] = running;
            running = running + item_i_58;
            int item_i_59 = rake_w[tid_1 * 33 + 24];
            rake_w[tid_1 * 33 + 24] = running;
            running = running + item_i_59;
            int item_i_60 = rake_w[tid_1 * 33 + 25];
            rake_w[tid_1 * 33 + 25] = running;
            running = running + item_i_60;
            int item_i_61 = rake_w[tid_1 * 33 + 26];
            rake_w[tid_1 * 33 + 26] = running;
            running = running + item_i_61;
            int item_i_62 = rake_w[tid_1 * 33 + 27];
            rake_w[tid_1 * 33 + 27] = running;
            running = running + item_i_62;
            int item_i_63 = rake_w[tid_1 * 33 + 28];
            rake_w[tid_1 * 33 + 28] = running;
            running = running + item_i_63;
            int item_i_64 = rake_w[tid_1 * 33 + 29];
            rake_w[tid_1 * 33 + 29] = running;
            running = running + item_i_64;
            int item_i_65 = rake_w[tid_1 * 33 + 30];
            rake_w[tid_1 * 33 + 30] = running;
            running = running + item_i_65;
            int item_i_66 = rake_w[tid_1 * 33 + 31];
            rake_w[tid_1 * 33 + 31] = running;
            running = running + item_i_66;
            int agg_w = aggregate;
            int partial_67 = rake_n[tid_1 * 33];
            int item_68 = rake_n[tid_1 * 33 + 1];
            partial_67 = partial_67 + item_68;
            int item_69 = rake_n[tid_1 * 33 + 2];
            partial_67 = partial_67 + item_69;
            int item_70 = rake_n[tid_1 * 33 + 3];
            partial_67 = partial_67 + item_70;
            int item_71 = rake_n[tid_1 * 33 + 4];
            partial_67 = partial_67 + item_71;
            int item_72 = rake_n[tid_1 * 33 + 5];
            partial_67 = partial_67 + item_72;
            int item_73 = rake_n[tid_1 * 33 + 6];
            partial_67 = partial_67 + item_73;
            int item_74 = rake_n[tid_1 * 33 + 7];
            partial_67 = partial_67 + item_74;
            int item_75 = rake_n[tid_1 * 33 + 8];
            partial_67 = partial_67 + item_75;
            int item_76 = rake_n[tid_1 * 33 + 9];
            partial_67 = partial_67 + item_76;
            int item_77 = rake_n[tid_1 * 33 + 10];
            partial_67 = partial_67 + item_77;
            int item_78 = rake_n[tid_1 * 33 + 11];
            partial_67 = partial_67 + item_78;
            int item_79 = rake_n[tid_1 * 33 + 12];
            partial_67 = partial_67 + item_79;
            int item_80 = rake_n[tid_1 * 33 + 13];
            partial_67 = partial_67 + item_80;
            int item_81 = rake_n[tid_1 * 33 + 14];
            partial_67 = partial_67 + item_81;
            int item_82 = rake_n[tid_1 * 33 + 15];
            partial_67 = partial_67 + item_82;
            int item_83 = rake_n[tid_1 * 33 + 16];
            partial_67 = partial_67 + item_83;
            int item_84 = rake_n[tid_1 * 33 + 17];
            partial_67 = partial_67 + item_84;
            int item_85 = rake_n[tid_1 * 33 + 18];
            partial_67 = partial_67 + item_85;
            int item_86 = rake_n[tid_1 * 33 + 19];
            partial_67 = partial_67 + item_86;
            int item_87 = rake_n[tid_1 * 33 + 20];
            partial_67 = partial_67 + item_87;
            int item_88 = rake_n[tid_1 * 33 + 21];
            partial_67 = partial_67 + item_88;
            int item_89 = rake_n[tid_1 * 33 + 22];
            partial_67 = partial_67 + item_89;
            int item_90 = rake_n[tid_1 * 33 + 23];
            partial_67 = partial_67 + item_90;
            int item_91 = rake_n[tid_1 * 33 + 24];
            partial_67 = partial_67 + item_91;
            int item_92 = rake_n[tid_1 * 33 + 25];
            partial_67 = partial_67 + item_92;
            int item_93 = rake_n[tid_1 * 33 + 26];
            partial_67 = partial_67 + item_93;
            int item_94 = rake_n[tid_1 * 33 + 27];
            partial_67 = partial_67 + item_94;
            int item_95 = rake_n[tid_1 * 33 + 28];
            partial_67 = partial_67 + item_95;
            int item_96 = rake_n[tid_1 * 33 + 29];
            partial_67 = partial_67 + item_96;
            int item_97 = rake_n[tid_1 * 33 + 30];
            partial_67 = partial_67 + item_97;
            int item_98 = rake_n[tid_1 * 33 + 31];
            partial_67 = partial_67 + item_98;
            int acc_99 = partial_67;
            int lane_100 = lane;
            int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, acc_99, 1, 32);
            int other_101 = _shfl_up_6;
            if (lane_100 >= 1) {
                acc_99 = acc_99 + other_101;
            }
            int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, acc_99, 2, 32);
            int other_102 = _shfl_up_7;
            if (lane_100 >= 2) {
                acc_99 = acc_99 + other_102;
            }
            int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, acc_99, 4, 32);
            int other_103 = _shfl_up_8;
            if (lane_100 >= 4) {
                acc_99 = acc_99 + other_103;
            }
            int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, acc_99, 8, 32);
            int other_104 = _shfl_up_9;
            if (lane_100 >= 8) {
                acc_99 = acc_99 + other_104;
            }
            int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, acc_99, 16, 32);
            int other_105 = _shfl_up_10;
            if (lane_100 >= 16) {
                acc_99 = acc_99 + other_105;
            }
            int inclusive_106 = acc_99;
            int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, inclusive_106, 1, 32);
            int exclusive_107 = _shfl_up_11;
            int lane_108 = lane;
            if (lane_108 == 0) {
                exclusive_107 = 0;
            }
            int _shfl_1 = __shfl_sync(0xFFFFFFFF, inclusive_106, 31);
            int aggregate_109 = _shfl_1;
            int running_110 = exclusive_107;
            int item_i_111 = rake_n[tid_1 * 33];
            rake_n[tid_1 * 33] = running_110;
            running_110 = running_110 + item_i_111;
            int item_i_112 = rake_n[tid_1 * 33 + 1];
            rake_n[tid_1 * 33 + 1] = running_110;
            running_110 = running_110 + item_i_112;
            int item_i_113 = rake_n[tid_1 * 33 + 2];
            rake_n[tid_1 * 33 + 2] = running_110;
            running_110 = running_110 + item_i_113;
            int item_i_114 = rake_n[tid_1 * 33 + 3];
            rake_n[tid_1 * 33 + 3] = running_110;
            running_110 = running_110 + item_i_114;
            int item_i_115 = rake_n[tid_1 * 33 + 4];
            rake_n[tid_1 * 33 + 4] = running_110;
            running_110 = running_110 + item_i_115;
            int item_i_116 = rake_n[tid_1 * 33 + 5];
            rake_n[tid_1 * 33 + 5] = running_110;
            running_110 = running_110 + item_i_116;
            int item_i_117 = rake_n[tid_1 * 33 + 6];
            rake_n[tid_1 * 33 + 6] = running_110;
            running_110 = running_110 + item_i_117;
            int item_i_118 = rake_n[tid_1 * 33 + 7];
            rake_n[tid_1 * 33 + 7] = running_110;
            running_110 = running_110 + item_i_118;
            int item_i_119 = rake_n[tid_1 * 33 + 8];
            rake_n[tid_1 * 33 + 8] = running_110;
            running_110 = running_110 + item_i_119;
            int item_i_120 = rake_n[tid_1 * 33 + 9];
            rake_n[tid_1 * 33 + 9] = running_110;
            running_110 = running_110 + item_i_120;
            int item_i_121 = rake_n[tid_1 * 33 + 10];
            rake_n[tid_1 * 33 + 10] = running_110;
            running_110 = running_110 + item_i_121;
            int item_i_122 = rake_n[tid_1 * 33 + 11];
            rake_n[tid_1 * 33 + 11] = running_110;
            running_110 = running_110 + item_i_122;
            int item_i_123 = rake_n[tid_1 * 33 + 12];
            rake_n[tid_1 * 33 + 12] = running_110;
            running_110 = running_110 + item_i_123;
            int item_i_124 = rake_n[tid_1 * 33 + 13];
            rake_n[tid_1 * 33 + 13] = running_110;
            running_110 = running_110 + item_i_124;
            int item_i_125 = rake_n[tid_1 * 33 + 14];
            rake_n[tid_1 * 33 + 14] = running_110;
            running_110 = running_110 + item_i_125;
            int item_i_126 = rake_n[tid_1 * 33 + 15];
            rake_n[tid_1 * 33 + 15] = running_110;
            running_110 = running_110 + item_i_126;
            int item_i_127 = rake_n[tid_1 * 33 + 16];
            rake_n[tid_1 * 33 + 16] = running_110;
            running_110 = running_110 + item_i_127;
            int item_i_128 = rake_n[tid_1 * 33 + 17];
            rake_n[tid_1 * 33 + 17] = running_110;
            running_110 = running_110 + item_i_128;
            int item_i_129 = rake_n[tid_1 * 33 + 18];
            rake_n[tid_1 * 33 + 18] = running_110;
            running_110 = running_110 + item_i_129;
            int item_i_130 = rake_n[tid_1 * 33 + 19];
            rake_n[tid_1 * 33 + 19] = running_110;
            running_110 = running_110 + item_i_130;
            int item_i_131 = rake_n[tid_1 * 33 + 20];
            rake_n[tid_1 * 33 + 20] = running_110;
            running_110 = running_110 + item_i_131;
            int item_i_132 = rake_n[tid_1 * 33 + 21];
            rake_n[tid_1 * 33 + 21] = running_110;
            running_110 = running_110 + item_i_132;
            int item_i_133 = rake_n[tid_1 * 33 + 22];
            rake_n[tid_1 * 33 + 22] = running_110;
            running_110 = running_110 + item_i_133;
            int item_i_134 = rake_n[tid_1 * 33 + 23];
            rake_n[tid_1 * 33 + 23] = running_110;
            running_110 = running_110 + item_i_134;
            int item_i_135 = rake_n[tid_1 * 33 + 24];
            rake_n[tid_1 * 33 + 24] = running_110;
            running_110 = running_110 + item_i_135;
            int item_i_136 = rake_n[tid_1 * 33 + 25];
            rake_n[tid_1 * 33 + 25] = running_110;
            running_110 = running_110 + item_i_136;
            int item_i_137 = rake_n[tid_1 * 33 + 26];
            rake_n[tid_1 * 33 + 26] = running_110;
            running_110 = running_110 + item_i_137;
            int item_i_138 = rake_n[tid_1 * 33 + 27];
            rake_n[tid_1 * 33 + 27] = running_110;
            running_110 = running_110 + item_i_138;
            int item_i_139 = rake_n[tid_1 * 33 + 28];
            rake_n[tid_1 * 33 + 28] = running_110;
            running_110 = running_110 + item_i_139;
            int item_i_140 = rake_n[tid_1 * 33 + 29];
            rake_n[tid_1 * 33 + 29] = running_110;
            running_110 = running_110 + item_i_140;
            int item_i_141 = rake_n[tid_1 * 33 + 30];
            rake_n[tid_1 * 33 + 30] = running_110;
            running_110 = running_110 + item_i_141;
            int item_i_142 = rake_n[tid_1 * 33 + 31];
            rake_n[tid_1 * 33 + 31] = running_110;
            running_110 = running_110 + item_i_142;
            int agg_n = aggregate_109;
            int partial_143 = rake_a[tid_1 * 33];
            int item_144 = rake_a[tid_1 * 33 + 1];
            partial_143 = partial_143 + item_144;
            int item_145 = rake_a[tid_1 * 33 + 2];
            partial_143 = partial_143 + item_145;
            int item_146 = rake_a[tid_1 * 33 + 3];
            partial_143 = partial_143 + item_146;
            int item_147 = rake_a[tid_1 * 33 + 4];
            partial_143 = partial_143 + item_147;
            int item_148 = rake_a[tid_1 * 33 + 5];
            partial_143 = partial_143 + item_148;
            int item_149 = rake_a[tid_1 * 33 + 6];
            partial_143 = partial_143 + item_149;
            int item_150 = rake_a[tid_1 * 33 + 7];
            partial_143 = partial_143 + item_150;
            int item_151 = rake_a[tid_1 * 33 + 8];
            partial_143 = partial_143 + item_151;
            int item_152 = rake_a[tid_1 * 33 + 9];
            partial_143 = partial_143 + item_152;
            int item_153 = rake_a[tid_1 * 33 + 10];
            partial_143 = partial_143 + item_153;
            int item_154 = rake_a[tid_1 * 33 + 11];
            partial_143 = partial_143 + item_154;
            int item_155 = rake_a[tid_1 * 33 + 12];
            partial_143 = partial_143 + item_155;
            int item_156 = rake_a[tid_1 * 33 + 13];
            partial_143 = partial_143 + item_156;
            int item_157 = rake_a[tid_1 * 33 + 14];
            partial_143 = partial_143 + item_157;
            int item_158 = rake_a[tid_1 * 33 + 15];
            partial_143 = partial_143 + item_158;
            int item_159 = rake_a[tid_1 * 33 + 16];
            partial_143 = partial_143 + item_159;
            int item_160 = rake_a[tid_1 * 33 + 17];
            partial_143 = partial_143 + item_160;
            int item_161 = rake_a[tid_1 * 33 + 18];
            partial_143 = partial_143 + item_161;
            int item_162 = rake_a[tid_1 * 33 + 19];
            partial_143 = partial_143 + item_162;
            int item_163 = rake_a[tid_1 * 33 + 20];
            partial_143 = partial_143 + item_163;
            int item_164 = rake_a[tid_1 * 33 + 21];
            partial_143 = partial_143 + item_164;
            int item_165 = rake_a[tid_1 * 33 + 22];
            partial_143 = partial_143 + item_165;
            int item_166 = rake_a[tid_1 * 33 + 23];
            partial_143 = partial_143 + item_166;
            int item_167 = rake_a[tid_1 * 33 + 24];
            partial_143 = partial_143 + item_167;
            int item_168 = rake_a[tid_1 * 33 + 25];
            partial_143 = partial_143 + item_168;
            int item_169 = rake_a[tid_1 * 33 + 26];
            partial_143 = partial_143 + item_169;
            int item_170 = rake_a[tid_1 * 33 + 27];
            partial_143 = partial_143 + item_170;
            int item_171 = rake_a[tid_1 * 33 + 28];
            partial_143 = partial_143 + item_171;
            int item_172 = rake_a[tid_1 * 33 + 29];
            partial_143 = partial_143 + item_172;
            int item_173 = rake_a[tid_1 * 33 + 30];
            partial_143 = partial_143 + item_173;
            int item_174 = rake_a[tid_1 * 33 + 31];
            partial_143 = partial_143 + item_174;
            int acc_175 = partial_143;
            int lane_176 = lane;
            int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, acc_175, 1, 32);
            int other_177 = _shfl_up_12;
            if (lane_176 >= 1) {
                acc_175 = acc_175 + other_177;
            }
            int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, acc_175, 2, 32);
            int other_178 = _shfl_up_13;
            if (lane_176 >= 2) {
                acc_175 = acc_175 + other_178;
            }
            int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, acc_175, 4, 32);
            int other_179 = _shfl_up_14;
            if (lane_176 >= 4) {
                acc_175 = acc_175 + other_179;
            }
            int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, acc_175, 8, 32);
            int other_180 = _shfl_up_15;
            if (lane_176 >= 8) {
                acc_175 = acc_175 + other_180;
            }
            int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, acc_175, 16, 32);
            int other_181 = _shfl_up_16;
            if (lane_176 >= 16) {
                acc_175 = acc_175 + other_181;
            }
            int inclusive_182 = acc_175;
            int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, inclusive_182, 1, 32);
            int exclusive_183 = _shfl_up_17;
            int lane_184 = lane;
            if (lane_184 == 0) {
                exclusive_183 = 0;
            }
            int _shfl_2 = __shfl_sync(0xFFFFFFFF, inclusive_182, 31);
            int aggregate_185 = _shfl_2;
            int running_186 = exclusive_183;
            int item_i_187 = rake_a[tid_1 * 33];
            rake_a[tid_1 * 33] = running_186;
            running_186 = running_186 + item_i_187;
            int item_i_188 = rake_a[tid_1 * 33 + 1];
            rake_a[tid_1 * 33 + 1] = running_186;
            running_186 = running_186 + item_i_188;
            int item_i_189 = rake_a[tid_1 * 33 + 2];
            rake_a[tid_1 * 33 + 2] = running_186;
            running_186 = running_186 + item_i_189;
            int item_i_190 = rake_a[tid_1 * 33 + 3];
            rake_a[tid_1 * 33 + 3] = running_186;
            running_186 = running_186 + item_i_190;
            int item_i_191 = rake_a[tid_1 * 33 + 4];
            rake_a[tid_1 * 33 + 4] = running_186;
            running_186 = running_186 + item_i_191;
            int item_i_192 = rake_a[tid_1 * 33 + 5];
            rake_a[tid_1 * 33 + 5] = running_186;
            running_186 = running_186 + item_i_192;
            int item_i_193 = rake_a[tid_1 * 33 + 6];
            rake_a[tid_1 * 33 + 6] = running_186;
            running_186 = running_186 + item_i_193;
            int item_i_194 = rake_a[tid_1 * 33 + 7];
            rake_a[tid_1 * 33 + 7] = running_186;
            running_186 = running_186 + item_i_194;
            int item_i_195 = rake_a[tid_1 * 33 + 8];
            rake_a[tid_1 * 33 + 8] = running_186;
            running_186 = running_186 + item_i_195;
            int item_i_196 = rake_a[tid_1 * 33 + 9];
            rake_a[tid_1 * 33 + 9] = running_186;
            running_186 = running_186 + item_i_196;
            int item_i_197 = rake_a[tid_1 * 33 + 10];
            rake_a[tid_1 * 33 + 10] = running_186;
            running_186 = running_186 + item_i_197;
            int item_i_198 = rake_a[tid_1 * 33 + 11];
            rake_a[tid_1 * 33 + 11] = running_186;
            running_186 = running_186 + item_i_198;
            int item_i_199 = rake_a[tid_1 * 33 + 12];
            rake_a[tid_1 * 33 + 12] = running_186;
            running_186 = running_186 + item_i_199;
            int item_i_200 = rake_a[tid_1 * 33 + 13];
            rake_a[tid_1 * 33 + 13] = running_186;
            running_186 = running_186 + item_i_200;
            int item_i_201 = rake_a[tid_1 * 33 + 14];
            rake_a[tid_1 * 33 + 14] = running_186;
            running_186 = running_186 + item_i_201;
            int item_i_202 = rake_a[tid_1 * 33 + 15];
            rake_a[tid_1 * 33 + 15] = running_186;
            running_186 = running_186 + item_i_202;
            int item_i_203 = rake_a[tid_1 * 33 + 16];
            rake_a[tid_1 * 33 + 16] = running_186;
            running_186 = running_186 + item_i_203;
            int item_i_204 = rake_a[tid_1 * 33 + 17];
            rake_a[tid_1 * 33 + 17] = running_186;
            running_186 = running_186 + item_i_204;
            int item_i_205 = rake_a[tid_1 * 33 + 18];
            rake_a[tid_1 * 33 + 18] = running_186;
            running_186 = running_186 + item_i_205;
            int item_i_206 = rake_a[tid_1 * 33 + 19];
            rake_a[tid_1 * 33 + 19] = running_186;
            running_186 = running_186 + item_i_206;
            int item_i_207 = rake_a[tid_1 * 33 + 20];
            rake_a[tid_1 * 33 + 20] = running_186;
            running_186 = running_186 + item_i_207;
            int item_i_208 = rake_a[tid_1 * 33 + 21];
            rake_a[tid_1 * 33 + 21] = running_186;
            running_186 = running_186 + item_i_208;
            int item_i_209 = rake_a[tid_1 * 33 + 22];
            rake_a[tid_1 * 33 + 22] = running_186;
            running_186 = running_186 + item_i_209;
            int item_i_210 = rake_a[tid_1 * 33 + 23];
            rake_a[tid_1 * 33 + 23] = running_186;
            running_186 = running_186 + item_i_210;
            int item_i_211 = rake_a[tid_1 * 33 + 24];
            rake_a[tid_1 * 33 + 24] = running_186;
            running_186 = running_186 + item_i_211;
            int item_i_212 = rake_a[tid_1 * 33 + 25];
            rake_a[tid_1 * 33 + 25] = running_186;
            running_186 = running_186 + item_i_212;
            int item_i_213 = rake_a[tid_1 * 33 + 26];
            rake_a[tid_1 * 33 + 26] = running_186;
            running_186 = running_186 + item_i_213;
            int item_i_214 = rake_a[tid_1 * 33 + 27];
            rake_a[tid_1 * 33 + 27] = running_186;
            running_186 = running_186 + item_i_214;
            int item_i_215 = rake_a[tid_1 * 33 + 28];
            rake_a[tid_1 * 33 + 28] = running_186;
            running_186 = running_186 + item_i_215;
            int item_i_216 = rake_a[tid_1 * 33 + 29];
            rake_a[tid_1 * 33 + 29] = running_186;
            running_186 = running_186 + item_i_216;
            int item_i_217 = rake_a[tid_1 * 33 + 30];
            rake_a[tid_1 * 33 + 30] = running_186;
            running_186 = running_186 + item_i_217;
            int item_i_218 = rake_a[tid_1 * 33 + 31];
            rake_a[tid_1 * 33 + 31] = running_186;
            running_186 = running_186 + item_i_218;
            int agg_a = aggregate_185;
            if (tid_1 == 0) {
                agg[0] = agg_w;
                agg[1] = agg_n;
                agg[2] = agg_a;
            }
        }
        __syncthreads();
        int excl_w = rake_w[slot];
        int excl_n = rake_n[slot];
        int excl_a = rake_a[slot];
        int total_w = agg[0];
        int total_n = agg[1];
        int total_a = agg[2];
        if (mine_wide > 0) {
            wide_list[carry_w + excl_w] = g;
        }
        #pragma unroll 1
        for (int s = 0; s < mine_narrow; s++) {
            narrow_list[carry_n + excl_n + s] = g * sub + s;
        }
        {
            #pragma unroll 1
            for (int s2 = 0; s2 < mine_all; s2++) {
                all_list[carry_a + excl_a + s2] = g * sub + s2;
            }
        }
        carry_w = carry_w + total_w;
        carry_n = carry_n + total_n;
        carry_a = carry_a + total_a;
        __syncthreads();
    }
    if (tid_0 == 0) {
        wide_count[0] = carry_w;
        narrow_count[0] = carry_n;
        {
            all_count[0] = carry_a;
        }
    }
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

} // extern "C"
