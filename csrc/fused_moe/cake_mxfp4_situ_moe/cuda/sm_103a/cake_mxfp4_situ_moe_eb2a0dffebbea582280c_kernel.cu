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
#define SMEM_AGG_W_OFF 0
#define SMEM_AGG_W_STAGE_BYTES 128
#define SMEM_AGG_W_STRIDE 128
#define SMEM_AGG_N_OFF 128
#define SMEM_AGG_N_STAGE_BYTES 128
#define SMEM_AGG_N_STRIDE 128
#define SMEM_AGG_A_OFF 256
#define SMEM_AGG_A_STAGE_BYTES 128
#define SMEM_AGG_A_STRIDE 128
#define SMEM_S_COUNTS_OFF 384
#define SMEM_S_COUNTS_STAGE_BYTES 16
#define SMEM_S_COUNTS_STRIDE 16
#define SMEM_S_BASE_E_OFF 400
#define SMEM_S_BASE_E_STAGE_BYTES 3304
#define SMEM_S_BASE_E_STRIDE 3304
#define SMEM_S_BASE_L_OFF 3704
#define SMEM_S_BASE_L_STAGE_BYTES 3304
#define SMEM_S_BASE_L_STRIDE 3304
#define SMEM_S_ALT_E_OFF 7008
#define SMEM_S_ALT_E_STAGE_BYTES 1976
#define SMEM_S_ALT_E_STRIDE 1976
#define SMEM_S_ALT_L_OFF 8984
#define SMEM_S_ALT_L_STAGE_BYTES 1976
#define SMEM_S_ALT_L_STRIDE 1976
#define SMEM_TOTAL 11008
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_mxfp4_situ_moe_eb2a0dffebbea582280c(int* __restrict__ expert_idx, int* __restrict__ mn_limit, int* __restrict__ num_groups_ptr, int* __restrict__ alt_expert_idx, int* __restrict__ alt_mn_limit, int* __restrict__ alt_num_groups_ptr, int* __restrict__ base_active_ptr, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ alt_wide_list, int* __restrict__ alt_wide_count, int* __restrict__ narrow_list, int* __restrict__ narrow_count, int* __restrict__ narrow_count_base, long long* __restrict__ trace, int group_rows, int alt_group_rows, int narrow_tile, int row_unit, int max_rows, int min_total_rows)
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
    int* agg_w = reinterpret_cast<int*>(smem_raw + 0);
    const int agg_w_addr = smem + 0;
    int* agg_n = reinterpret_cast<int*>(smem_raw + 128);
    const int agg_n_addr = smem + 128;
    int* agg_a = reinterpret_cast<int*>(smem_raw + 256);
    const int agg_a_addr = smem + 256;
    int* s_counts = reinterpret_cast<int*>(smem_raw + 384);
    const int s_counts_addr = smem + 384;
    int* s_base_e = reinterpret_cast<int*>(smem_raw + 400);
    const int s_base_e_addr = smem + 400;
    int* s_base_l = reinterpret_cast<int*>(smem_raw + 3704);
    const int s_base_l_addr = smem + 3704;
    int* s_alt_e = reinterpret_cast<int*>(smem_raw + 7008);
    const int s_alt_e_addr = smem + 7008;
    int* s_alt_l = reinterpret_cast<int*>(smem_raw + 8984);
    const int s_alt_l_addr = smem + 8984;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    if (tid_0 == 0) {
        s_counts[0] = num_groups_ptr[0];
    }
    if (tid_0 == 1) {
        int alt_groups_in = 0;
        {
            alt_groups_in = alt_num_groups_ptr[0];
        }
        s_counts[1] = alt_groups_in;
    }
    if (tid_0 == 2) {
        int base_active_in = 0;
        {
            base_active_in = base_active_ptr[0];
        }
        s_counts[2] = base_active_in;
    }
    {
        int tid_1 = tid;
        int va[4];
        int vb[4];
        #pragma unroll 1
        for (int i = tid_1; i < 826; i += 4096) {
            int idx = i;
            if (idx < 826) {
                va[0] = expert_idx[idx];
                vb[0] = mn_limit[idx];
            }
            int idx_0 = i + 1024;
            if (idx_0 < 826) {
                va[1] = expert_idx[idx_0];
                vb[1] = mn_limit[idx_0];
            }
            int idx_1 = i + 2048;
            if (idx_1 < 826) {
                va[2] = expert_idx[idx_1];
                vb[2] = mn_limit[idx_1];
            }
            int idx_2 = i + 3072;
            if (idx_2 < 826) {
                va[3] = expert_idx[idx_2];
                vb[3] = mn_limit[idx_2];
            }
            int idx2 = i;
            if (idx2 < 826) {
                s_base_e[idx2] = va[0];
                s_base_l[idx2] = vb[0];
            }
            int idx2_3 = i + 1024;
            if (idx2_3 < 826) {
                s_base_e[idx2_3] = va[1];
                s_base_l[idx2_3] = vb[1];
            }
            int idx2_4 = i + 2048;
            if (idx2_4 < 826) {
                s_base_e[idx2_4] = va[2];
                s_base_l[idx2_4] = vb[2];
            }
            int idx2_5 = i + 3072;
            if (idx2_5 < 826) {
                s_base_e[idx2_5] = va[3];
                s_base_l[idx2_5] = vb[3];
            }
        }
    }
    {
        int tid_1_1 = tid;
        int va_1[4];
        int vb_1[4];
        #pragma unroll 1
        for (int i_1 = tid_1_1; i_1 < 494; i_1 += 4096) {
            int idx_3 = i_1;
            if (idx_3 < 494) {
                va_1[0] = alt_expert_idx[idx_3];
                vb_1[0] = alt_mn_limit[idx_3];
            }
            int idx_0_1 = i_1 + 1024;
            if (idx_0_1 < 494) {
                va_1[1] = alt_expert_idx[idx_0_1];
                vb_1[1] = alt_mn_limit[idx_0_1];
            }
            int idx_1_1 = i_1 + 2048;
            if (idx_1_1 < 494) {
                va_1[2] = alt_expert_idx[idx_1_1];
                vb_1[2] = alt_mn_limit[idx_1_1];
            }
            int idx_2_1 = i_1 + 3072;
            if (idx_2_1 < 494) {
                va_1[3] = alt_expert_idx[idx_2_1];
                vb_1[3] = alt_mn_limit[idx_2_1];
            }
            int idx2_1 = i_1;
            if (idx2_1 < 494) {
                s_alt_e[idx2_1] = va_1[0];
                s_alt_l[idx2_1] = vb_1[0];
            }
            int idx2_3_1 = i_1 + 1024;
            if (idx2_3_1 < 494) {
                s_alt_e[idx2_3_1] = va_1[1];
                s_alt_l[idx2_3_1] = vb_1[1];
            }
            int idx2_4_1 = i_1 + 2048;
            if (idx2_4_1 < 494) {
                s_alt_e[idx2_4_1] = va_1[2];
                s_alt_l[idx2_4_1] = vb_1[2];
            }
            int idx2_5_1 = i_1 + 3072;
            if (idx2_5_1 < 494) {
                s_alt_e[idx2_5_1] = va_1[3];
                s_alt_l[idx2_5_1] = vb_1[3];
            }
        }
    }
    __syncthreads();
    int base_groups = s_counts[0];
    int alt = 0;
    int num_groups = base_groups;
    int staged_len = 826;
    int rows = group_rows;
    {
        int alt_groups = s_counts[1];
        int base_active = s_counts[2];
        if (base_active == 0 && alt_groups > 0) {
            alt = 1;
            num_groups = alt_groups;
            staged_len = 494;
            rows = alt_group_rows;
        }
    }
    int use_smem = 0;
    if (num_groups <= staged_len) {
        use_smem = 1;
    }
    int gu = rows / row_unit;
    int windows = 0;
    if ((long long)num_groups * (long long)rows >= (long long)min_total_rows) {
        windows = 1;
    }
    int carry_w = 0;
    int carry_n = 0;
    int carry_a = 0;
    int pass_idx = 0;
    #pragma unroll 1
    for (int base_g = 0; base_g < num_groups; base_g += 4096) {
        int mine_w[4];
        int mine_n[4];
        int g = base_g + tid_0 * 4;
        int nwide = 0;
        int nnarrow = 0;
        if (g < num_groups) {
            int v = 0;
            if (use_smem == 1) {
                if (alt == 1) {
                    v = s_alt_e[g];
                } else {
                    v = s_base_e[g];
                }
            } else if (alt == 1) {
                v = alt_expert_idx[g];
            } else {
                v = expert_idx[g];
            }
            int ex = v;
            int first = 1;
            if (g > 0) {
                int v_0 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0 = s_alt_e[g - 1];
                    } else {
                        v_0 = s_base_e[g - 1];
                    }
                } else if (alt == 1) {
                    v_0 = alt_expert_idx[g - 1];
                } else {
                    v_0 = expert_idx[g - 1];
                }
                int prev = v_0;
                if (prev == ex) {
                    first = 0;
                }
            }
            int c = 0;
            if (first == 1) {
                int lo = g + 1;
                int hi = num_groups;
                while (lo < hi) {
                    int mid = lo + (hi - lo >> 1);
                    int v_0_1 = 0;
                    if (use_smem == 1) {
                        if (alt == 1) {
                            v_0_1 = s_alt_e[mid];
                        } else {
                            v_0_1 = s_base_e[mid];
                        }
                    } else if (alt == 1) {
                        v_0_1 = alt_expert_idx[mid];
                    } else {
                        v_0_1 = expert_idx[mid];
                    }
                    int em = v_0_1;
                    if (em == ex) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int v_0_2 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_2 = s_alt_l[lo - 1];
                    } else {
                        v_0_2 = s_base_l[lo - 1];
                    }
                } else if (alt == 1) {
                    v_0_2 = alt_mn_limit[lo - 1];
                } else {
                    v_0_2 = mn_limit[lo - 1];
                }
                int last_limit = v_0_2;
                int _max_0 = ((last_limit - g * rows) > (0) ? (last_limit - g * rows) : (0));
                c = _max_0;
            }
            int dense_only = 0;
            if (windows == 0) {
                dense_only = 1;
            }
            if (max_rows > 0) {
                if (c > max_rows) {
                    dense_only = 1;
                }
            }
            if (c > 0) {
                if (dense_only == 1) {
                    nwide = (c + rows - 1) / rows;
                } else {
                    int best_cover = 2147483647;
                    #pragma unroll 1
                    for (int a = 0; a < gu; a++) {
                        int rem = c - a * narrow_tile;
                        int w = 0;
                        if (rem > 0) {
                            w = (rem + rows - 1) / rows;
                        }
                        int cover = w * rows + a * narrow_tile;
                        if (cover < best_cover) {
                            best_cover = cover;
                            nwide = w;
                            nnarrow = a;
                        }
                        if (rem <= 0) {
                            break;
                        }
                    }
                }
            }
        }
        mine_w[0] = nwide;
        mine_n[0] = nnarrow;
        int g_0 = base_g + tid_0 * 4 + 1;
        int nwide_1 = 0;
        int nnarrow_2 = 0;
        if (g_0 < num_groups) {
            int v_1 = 0;
            if (use_smem == 1) {
                if (alt == 1) {
                    v_1 = s_alt_e[g_0];
                } else {
                    v_1 = s_base_e[g_0];
                }
            } else if (alt == 1) {
                v_1 = alt_expert_idx[g_0];
            } else {
                v_1 = expert_idx[g_0];
            }
            int ex_1 = v_1;
            int first_1 = 1;
            if (g_0 > 0) {
                int v_0_3 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_3 = s_alt_e[g_0 - 1];
                    } else {
                        v_0_3 = s_base_e[g_0 - 1];
                    }
                } else if (alt == 1) {
                    v_0_3 = alt_expert_idx[g_0 - 1];
                } else {
                    v_0_3 = expert_idx[g_0 - 1];
                }
                int prev_1 = v_0_3;
                if (prev_1 == ex_1) {
                    first_1 = 0;
                }
            }
            int c_1 = 0;
            if (first_1 == 1) {
                int lo_1 = g_0 + 1;
                int hi_1 = num_groups;
                while (lo_1 < hi_1) {
                    int mid_1 = lo_1 + (hi_1 - lo_1 >> 1);
                    int v_0_4 = 0;
                    if (use_smem == 1) {
                        if (alt == 1) {
                            v_0_4 = s_alt_e[mid_1];
                        } else {
                            v_0_4 = s_base_e[mid_1];
                        }
                    } else if (alt == 1) {
                        v_0_4 = alt_expert_idx[mid_1];
                    } else {
                        v_0_4 = expert_idx[mid_1];
                    }
                    int em_1 = v_0_4;
                    if (em_1 == ex_1) {
                        lo_1 = mid_1 + 1;
                    } else {
                        hi_1 = mid_1;
                    }
                }
                int v_0_5 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_5 = s_alt_l[lo_1 - 1];
                    } else {
                        v_0_5 = s_base_l[lo_1 - 1];
                    }
                } else if (alt == 1) {
                    v_0_5 = alt_mn_limit[lo_1 - 1];
                } else {
                    v_0_5 = mn_limit[lo_1 - 1];
                }
                int last_limit_1 = v_0_5;
                int _max_1 = ((last_limit_1 - g_0 * rows) > (0) ? (last_limit_1 - g_0 * rows) : (0));
                c_1 = _max_1;
            }
            int dense_only_1 = 0;
            if (windows == 0) {
                dense_only_1 = 1;
            }
            if (max_rows > 0) {
                if (c_1 > max_rows) {
                    dense_only_1 = 1;
                }
            }
            if (c_1 > 0) {
                if (dense_only_1 == 1) {
                    nwide_1 = (c_1 + rows - 1) / rows;
                } else {
                    int best_cover_1 = 2147483647;
                    #pragma unroll 1
                    for (int a_1 = 0; a_1 < gu; a_1++) {
                        int rem_1 = c_1 - a_1 * narrow_tile;
                        int w_1 = 0;
                        if (rem_1 > 0) {
                            w_1 = (rem_1 + rows - 1) / rows;
                        }
                        int cover_1 = w_1 * rows + a_1 * narrow_tile;
                        if (cover_1 < best_cover_1) {
                            best_cover_1 = cover_1;
                            nwide_1 = w_1;
                            nnarrow_2 = a_1;
                        }
                        if (rem_1 <= 0) {
                            break;
                        }
                    }
                }
            }
        }
        mine_w[1] = nwide_1;
        mine_n[1] = nnarrow_2;
        int g_3 = base_g + tid_0 * 4 + 2;
        int nwide_4 = 0;
        int nnarrow_5 = 0;
        if (g_3 < num_groups) {
            int v_2 = 0;
            if (use_smem == 1) {
                if (alt == 1) {
                    v_2 = s_alt_e[g_3];
                } else {
                    v_2 = s_base_e[g_3];
                }
            } else if (alt == 1) {
                v_2 = alt_expert_idx[g_3];
            } else {
                v_2 = expert_idx[g_3];
            }
            int ex_2 = v_2;
            int first_2 = 1;
            if (g_3 > 0) {
                int v_0_6 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_6 = s_alt_e[g_3 - 1];
                    } else {
                        v_0_6 = s_base_e[g_3 - 1];
                    }
                } else if (alt == 1) {
                    v_0_6 = alt_expert_idx[g_3 - 1];
                } else {
                    v_0_6 = expert_idx[g_3 - 1];
                }
                int prev_2 = v_0_6;
                if (prev_2 == ex_2) {
                    first_2 = 0;
                }
            }
            int c_2 = 0;
            if (first_2 == 1) {
                int lo_2 = g_3 + 1;
                int hi_2 = num_groups;
                while (lo_2 < hi_2) {
                    int mid_2 = lo_2 + (hi_2 - lo_2 >> 1);
                    int v_0_7 = 0;
                    if (use_smem == 1) {
                        if (alt == 1) {
                            v_0_7 = s_alt_e[mid_2];
                        } else {
                            v_0_7 = s_base_e[mid_2];
                        }
                    } else if (alt == 1) {
                        v_0_7 = alt_expert_idx[mid_2];
                    } else {
                        v_0_7 = expert_idx[mid_2];
                    }
                    int em_2 = v_0_7;
                    if (em_2 == ex_2) {
                        lo_2 = mid_2 + 1;
                    } else {
                        hi_2 = mid_2;
                    }
                }
                int v_0_8 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_8 = s_alt_l[lo_2 - 1];
                    } else {
                        v_0_8 = s_base_l[lo_2 - 1];
                    }
                } else if (alt == 1) {
                    v_0_8 = alt_mn_limit[lo_2 - 1];
                } else {
                    v_0_8 = mn_limit[lo_2 - 1];
                }
                int last_limit_2 = v_0_8;
                int _max_2 = ((last_limit_2 - g_3 * rows) > (0) ? (last_limit_2 - g_3 * rows) : (0));
                c_2 = _max_2;
            }
            int dense_only_2 = 0;
            if (windows == 0) {
                dense_only_2 = 1;
            }
            if (max_rows > 0) {
                if (c_2 > max_rows) {
                    dense_only_2 = 1;
                }
            }
            if (c_2 > 0) {
                if (dense_only_2 == 1) {
                    nwide_4 = (c_2 + rows - 1) / rows;
                } else {
                    int best_cover_2 = 2147483647;
                    #pragma unroll 1
                    for (int a_2 = 0; a_2 < gu; a_2++) {
                        int rem_2 = c_2 - a_2 * narrow_tile;
                        int w_2 = 0;
                        if (rem_2 > 0) {
                            w_2 = (rem_2 + rows - 1) / rows;
                        }
                        int cover_2 = w_2 * rows + a_2 * narrow_tile;
                        if (cover_2 < best_cover_2) {
                            best_cover_2 = cover_2;
                            nwide_4 = w_2;
                            nnarrow_5 = a_2;
                        }
                        if (rem_2 <= 0) {
                            break;
                        }
                    }
                }
            }
        }
        mine_w[2] = nwide_4;
        mine_n[2] = nnarrow_5;
        int g_6 = base_g + tid_0 * 4 + 3;
        int nwide_7 = 0;
        int nnarrow_8 = 0;
        if (g_6 < num_groups) {
            int v_3 = 0;
            if (use_smem == 1) {
                if (alt == 1) {
                    v_3 = s_alt_e[g_6];
                } else {
                    v_3 = s_base_e[g_6];
                }
            } else if (alt == 1) {
                v_3 = alt_expert_idx[g_6];
            } else {
                v_3 = expert_idx[g_6];
            }
            int ex_3 = v_3;
            int first_3 = 1;
            if (g_6 > 0) {
                int v_0_9 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_9 = s_alt_e[g_6 - 1];
                    } else {
                        v_0_9 = s_base_e[g_6 - 1];
                    }
                } else if (alt == 1) {
                    v_0_9 = alt_expert_idx[g_6 - 1];
                } else {
                    v_0_9 = expert_idx[g_6 - 1];
                }
                int prev_3 = v_0_9;
                if (prev_3 == ex_3) {
                    first_3 = 0;
                }
            }
            int c_3 = 0;
            if (first_3 == 1) {
                int lo_3 = g_6 + 1;
                int hi_3 = num_groups;
                while (lo_3 < hi_3) {
                    int mid_3 = lo_3 + (hi_3 - lo_3 >> 1);
                    int v_0_10 = 0;
                    if (use_smem == 1) {
                        if (alt == 1) {
                            v_0_10 = s_alt_e[mid_3];
                        } else {
                            v_0_10 = s_base_e[mid_3];
                        }
                    } else if (alt == 1) {
                        v_0_10 = alt_expert_idx[mid_3];
                    } else {
                        v_0_10 = expert_idx[mid_3];
                    }
                    int em_3 = v_0_10;
                    if (em_3 == ex_3) {
                        lo_3 = mid_3 + 1;
                    } else {
                        hi_3 = mid_3;
                    }
                }
                int v_0_11 = 0;
                if (use_smem == 1) {
                    if (alt == 1) {
                        v_0_11 = s_alt_l[lo_3 - 1];
                    } else {
                        v_0_11 = s_base_l[lo_3 - 1];
                    }
                } else if (alt == 1) {
                    v_0_11 = alt_mn_limit[lo_3 - 1];
                } else {
                    v_0_11 = mn_limit[lo_3 - 1];
                }
                int last_limit_3 = v_0_11;
                int _max_3 = ((last_limit_3 - g_6 * rows) > (0) ? (last_limit_3 - g_6 * rows) : (0));
                c_3 = _max_3;
            }
            int dense_only_3 = 0;
            if (windows == 0) {
                dense_only_3 = 1;
            }
            if (max_rows > 0) {
                if (c_3 > max_rows) {
                    dense_only_3 = 1;
                }
            }
            if (c_3 > 0) {
                if (dense_only_3 == 1) {
                    nwide_7 = (c_3 + rows - 1) / rows;
                } else {
                    int best_cover_3 = 2147483647;
                    #pragma unroll 1
                    for (int a_3 = 0; a_3 < gu; a_3++) {
                        int rem_3 = c_3 - a_3 * narrow_tile;
                        int w_3 = 0;
                        if (rem_3 > 0) {
                            w_3 = (rem_3 + rows - 1) / rows;
                        }
                        int cover_3 = w_3 * rows + a_3 * narrow_tile;
                        if (cover_3 < best_cover_3) {
                            best_cover_3 = cover_3;
                            nwide_7 = w_3;
                            nnarrow_8 = a_3;
                        }
                        if (rem_3 <= 0) {
                            break;
                        }
                    }
                }
            }
        }
        mine_w[3] = nwide_7;
        mine_n[3] = nnarrow_8;
        int part_w = mine_w[0] + mine_w[1] + mine_w[2] + mine_w[3];
        int part_n = mine_n[0] + mine_n[1] + mine_n[2] + mine_n[3];
        int zero_a = 0;
        int lane_9 = lane;
        int warp_10 = warp;
        int acc = part_w;
        int lane_11 = lane;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, acc, 1, 32);
        int other = _shfl_up_0;
        if (lane_11 >= 1) {
            acc = acc + other;
        }
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, acc, 2, 32);
        int other_12 = _shfl_up_1;
        if (lane_11 >= 2) {
            acc = acc + other_12;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, acc, 4, 32);
        int other_13 = _shfl_up_2;
        if (lane_11 >= 4) {
            acc = acc + other_13;
        }
        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, acc, 8, 32);
        int other_14 = _shfl_up_3;
        if (lane_11 >= 8) {
            acc = acc + other_14;
        }
        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, acc, 16, 32);
        int other_15 = _shfl_up_4;
        if (lane_11 >= 16) {
            acc = acc + other_15;
        }
        int inc_w = acc;
        int acc_16 = part_n;
        int lane_17 = lane;
        int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, acc_16, 1, 32);
        int other_18 = _shfl_up_5;
        if (lane_17 >= 1) {
            acc_16 = acc_16 + other_18;
        }
        int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, acc_16, 2, 32);
        int other_19 = _shfl_up_6;
        if (lane_17 >= 2) {
            acc_16 = acc_16 + other_19;
        }
        int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, acc_16, 4, 32);
        int other_20 = _shfl_up_7;
        if (lane_17 >= 4) {
            acc_16 = acc_16 + other_20;
        }
        int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, acc_16, 8, 32);
        int other_21 = _shfl_up_8;
        if (lane_17 >= 8) {
            acc_16 = acc_16 + other_21;
        }
        int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, acc_16, 16, 32);
        int other_22 = _shfl_up_9;
        if (lane_17 >= 16) {
            acc_16 = acc_16 + other_22;
        }
        int inc_n = acc_16;
        int acc_23 = zero_a;
        int lane_24 = lane;
        int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, acc_23, 1, 32);
        int other_25 = _shfl_up_10;
        if (lane_24 >= 1) {
            acc_23 = acc_23 + other_25;
        }
        int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, acc_23, 2, 32);
        int other_26 = _shfl_up_11;
        if (lane_24 >= 2) {
            acc_23 = acc_23 + other_26;
        }
        int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, acc_23, 4, 32);
        int other_27 = _shfl_up_12;
        if (lane_24 >= 4) {
            acc_23 = acc_23 + other_27;
        }
        int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, acc_23, 8, 32);
        int other_28 = _shfl_up_13;
        if (lane_24 >= 8) {
            acc_23 = acc_23 + other_28;
        }
        int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, acc_23, 16, 32);
        int other_29 = _shfl_up_14;
        if (lane_24 >= 16) {
            acc_23 = acc_23 + other_29;
        }
        int inc_a = acc_23;
        int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, inc_w, 1, 32);
        int exc_w = _shfl_up_15;
        int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, inc_n, 1, 32);
        int exc_n = _shfl_up_16;
        int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, inc_a, 1, 32);
        int exc_a = _shfl_up_17;
        if (lane_9 == 31) {
            agg_w[warp_10] = inc_w;
            agg_n[warp_10] = inc_n;
            agg_a[warp_10] = inc_a;
        }
        __syncthreads();
        int total_w = agg_w[0];
        int total_n = agg_n[0];
        int total_a = agg_a[0];
        int prefix_w = 0;
        int prefix_n = 0;
        int prefix_a = 0;
        if (warp_10 == 1) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w = agg_w[1];
        int item_n = agg_n[1];
        int item_a = agg_a[1];
        total_w = total_w + item_w;
        total_n = total_n + item_n;
        total_a = total_a + item_a;
        if (warp_10 == 2) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_30 = agg_w[2];
        int item_n_31 = agg_n[2];
        int item_a_32 = agg_a[2];
        total_w = total_w + item_w_30;
        total_n = total_n + item_n_31;
        total_a = total_a + item_a_32;
        if (warp_10 == 3) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_33 = agg_w[3];
        int item_n_34 = agg_n[3];
        int item_a_35 = agg_a[3];
        total_w = total_w + item_w_33;
        total_n = total_n + item_n_34;
        total_a = total_a + item_a_35;
        if (warp_10 == 4) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_36 = agg_w[4];
        int item_n_37 = agg_n[4];
        int item_a_38 = agg_a[4];
        total_w = total_w + item_w_36;
        total_n = total_n + item_n_37;
        total_a = total_a + item_a_38;
        if (warp_10 == 5) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_39 = agg_w[5];
        int item_n_40 = agg_n[5];
        int item_a_41 = agg_a[5];
        total_w = total_w + item_w_39;
        total_n = total_n + item_n_40;
        total_a = total_a + item_a_41;
        if (warp_10 == 6) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_42 = agg_w[6];
        int item_n_43 = agg_n[6];
        int item_a_44 = agg_a[6];
        total_w = total_w + item_w_42;
        total_n = total_n + item_n_43;
        total_a = total_a + item_a_44;
        if (warp_10 == 7) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_45 = agg_w[7];
        int item_n_46 = agg_n[7];
        int item_a_47 = agg_a[7];
        total_w = total_w + item_w_45;
        total_n = total_n + item_n_46;
        total_a = total_a + item_a_47;
        if (warp_10 == 8) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_48 = agg_w[8];
        int item_n_49 = agg_n[8];
        int item_a_50 = agg_a[8];
        total_w = total_w + item_w_48;
        total_n = total_n + item_n_49;
        total_a = total_a + item_a_50;
        if (warp_10 == 9) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_51 = agg_w[9];
        int item_n_52 = agg_n[9];
        int item_a_53 = agg_a[9];
        total_w = total_w + item_w_51;
        total_n = total_n + item_n_52;
        total_a = total_a + item_a_53;
        if (warp_10 == 10) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_54 = agg_w[10];
        int item_n_55 = agg_n[10];
        int item_a_56 = agg_a[10];
        total_w = total_w + item_w_54;
        total_n = total_n + item_n_55;
        total_a = total_a + item_a_56;
        if (warp_10 == 11) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_57 = agg_w[11];
        int item_n_58 = agg_n[11];
        int item_a_59 = agg_a[11];
        total_w = total_w + item_w_57;
        total_n = total_n + item_n_58;
        total_a = total_a + item_a_59;
        if (warp_10 == 12) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_60 = agg_w[12];
        int item_n_61 = agg_n[12];
        int item_a_62 = agg_a[12];
        total_w = total_w + item_w_60;
        total_n = total_n + item_n_61;
        total_a = total_a + item_a_62;
        if (warp_10 == 13) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_63 = agg_w[13];
        int item_n_64 = agg_n[13];
        int item_a_65 = agg_a[13];
        total_w = total_w + item_w_63;
        total_n = total_n + item_n_64;
        total_a = total_a + item_a_65;
        if (warp_10 == 14) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_66 = agg_w[14];
        int item_n_67 = agg_n[14];
        int item_a_68 = agg_a[14];
        total_w = total_w + item_w_66;
        total_n = total_n + item_n_67;
        total_a = total_a + item_a_68;
        if (warp_10 == 15) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_69 = agg_w[15];
        int item_n_70 = agg_n[15];
        int item_a_71 = agg_a[15];
        total_w = total_w + item_w_69;
        total_n = total_n + item_n_70;
        total_a = total_a + item_a_71;
        if (warp_10 == 16) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_72 = agg_w[16];
        int item_n_73 = agg_n[16];
        int item_a_74 = agg_a[16];
        total_w = total_w + item_w_72;
        total_n = total_n + item_n_73;
        total_a = total_a + item_a_74;
        if (warp_10 == 17) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_75 = agg_w[17];
        int item_n_76 = agg_n[17];
        int item_a_77 = agg_a[17];
        total_w = total_w + item_w_75;
        total_n = total_n + item_n_76;
        total_a = total_a + item_a_77;
        if (warp_10 == 18) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_78 = agg_w[18];
        int item_n_79 = agg_n[18];
        int item_a_80 = agg_a[18];
        total_w = total_w + item_w_78;
        total_n = total_n + item_n_79;
        total_a = total_a + item_a_80;
        if (warp_10 == 19) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_81 = agg_w[19];
        int item_n_82 = agg_n[19];
        int item_a_83 = agg_a[19];
        total_w = total_w + item_w_81;
        total_n = total_n + item_n_82;
        total_a = total_a + item_a_83;
        if (warp_10 == 20) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_84 = agg_w[20];
        int item_n_85 = agg_n[20];
        int item_a_86 = agg_a[20];
        total_w = total_w + item_w_84;
        total_n = total_n + item_n_85;
        total_a = total_a + item_a_86;
        if (warp_10 == 21) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_87 = agg_w[21];
        int item_n_88 = agg_n[21];
        int item_a_89 = agg_a[21];
        total_w = total_w + item_w_87;
        total_n = total_n + item_n_88;
        total_a = total_a + item_a_89;
        if (warp_10 == 22) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_90 = agg_w[22];
        int item_n_91 = agg_n[22];
        int item_a_92 = agg_a[22];
        total_w = total_w + item_w_90;
        total_n = total_n + item_n_91;
        total_a = total_a + item_a_92;
        if (warp_10 == 23) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_93 = agg_w[23];
        int item_n_94 = agg_n[23];
        int item_a_95 = agg_a[23];
        total_w = total_w + item_w_93;
        total_n = total_n + item_n_94;
        total_a = total_a + item_a_95;
        if (warp_10 == 24) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_96 = agg_w[24];
        int item_n_97 = agg_n[24];
        int item_a_98 = agg_a[24];
        total_w = total_w + item_w_96;
        total_n = total_n + item_n_97;
        total_a = total_a + item_a_98;
        if (warp_10 == 25) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_99 = agg_w[25];
        int item_n_100 = agg_n[25];
        int item_a_101 = agg_a[25];
        total_w = total_w + item_w_99;
        total_n = total_n + item_n_100;
        total_a = total_a + item_a_101;
        if (warp_10 == 26) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_102 = agg_w[26];
        int item_n_103 = agg_n[26];
        int item_a_104 = agg_a[26];
        total_w = total_w + item_w_102;
        total_n = total_n + item_n_103;
        total_a = total_a + item_a_104;
        if (warp_10 == 27) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_105 = agg_w[27];
        int item_n_106 = agg_n[27];
        int item_a_107 = agg_a[27];
        total_w = total_w + item_w_105;
        total_n = total_n + item_n_106;
        total_a = total_a + item_a_107;
        if (warp_10 == 28) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_108 = agg_w[28];
        int item_n_109 = agg_n[28];
        int item_a_110 = agg_a[28];
        total_w = total_w + item_w_108;
        total_n = total_n + item_n_109;
        total_a = total_a + item_a_110;
        if (warp_10 == 29) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_111 = agg_w[29];
        int item_n_112 = agg_n[29];
        int item_a_113 = agg_a[29];
        total_w = total_w + item_w_111;
        total_n = total_n + item_n_112;
        total_a = total_a + item_a_113;
        if (warp_10 == 30) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_114 = agg_w[30];
        int item_n_115 = agg_n[30];
        int item_a_116 = agg_a[30];
        total_w = total_w + item_w_114;
        total_n = total_n + item_n_115;
        total_a = total_a + item_a_116;
        if (warp_10 == 31) {
            prefix_w = total_w;
            prefix_n = total_n;
            prefix_a = total_a;
        }
        int item_w_117 = agg_w[31];
        int item_n_118 = agg_n[31];
        int item_a_119 = agg_a[31];
        total_w = total_w + item_w_117;
        total_n = total_n + item_n_118;
        total_a = total_a + item_a_119;
        exc_w = prefix_w + exc_w;
        exc_n = prefix_n + exc_n;
        exc_a = prefix_a + exc_a;
        if (lane_9 == 0) {
            exc_w = prefix_w;
            exc_n = prefix_n;
            exc_a = prefix_a;
        }
        int run_w = exc_w;
        int run_n = exc_n;
        int g_120 = base_g + tid_0 * 4;
        int nw = mine_w[0];
        int nn = mine_n[0];
        #pragma unroll 1
        for (int t = 0; t < nw; t++) {
            if (alt == 1) {
                alt_wide_list[carry_w + run_w + t] = g_120 + t;
            } else {
                wide_list[carry_w + run_w + t] = g_120 + t;
            }
        }
        int narrow_row0 = (g_120 + nw) * rows;
        #pragma unroll 1
        for (int i_2 = 0; i_2 < nn; i_2++) {
            narrow_list[carry_n + run_n + i_2] = (narrow_row0 + i_2 * narrow_tile) / row_unit;
        }
        run_w = run_w + nw;
        run_n = run_n + nn;
        int g_121 = base_g + tid_0 * 4 + 1;
        int nw_122 = mine_w[1];
        int nn_123 = mine_n[1];
        #pragma unroll 1
        for (int t_1 = 0; t_1 < nw_122; t_1++) {
            if (alt == 1) {
                alt_wide_list[carry_w + run_w + t_1] = g_121 + t_1;
            } else {
                wide_list[carry_w + run_w + t_1] = g_121 + t_1;
            }
        }
        int narrow_row0_124 = (g_121 + nw_122) * rows;
        #pragma unroll 1
        for (int i_3 = 0; i_3 < nn_123; i_3++) {
            narrow_list[carry_n + run_n + i_3] = (narrow_row0_124 + i_3 * narrow_tile) / row_unit;
        }
        run_w = run_w + nw_122;
        run_n = run_n + nn_123;
        int g_125 = base_g + tid_0 * 4 + 2;
        int nw_126 = mine_w[2];
        int nn_127 = mine_n[2];
        #pragma unroll 1
        for (int t_2 = 0; t_2 < nw_126; t_2++) {
            if (alt == 1) {
                alt_wide_list[carry_w + run_w + t_2] = g_125 + t_2;
            } else {
                wide_list[carry_w + run_w + t_2] = g_125 + t_2;
            }
        }
        int narrow_row0_128 = (g_125 + nw_126) * rows;
        #pragma unroll 1
        for (int i_4 = 0; i_4 < nn_127; i_4++) {
            narrow_list[carry_n + run_n + i_4] = (narrow_row0_128 + i_4 * narrow_tile) / row_unit;
        }
        run_w = run_w + nw_126;
        run_n = run_n + nn_127;
        int g_129 = base_g + tid_0 * 4 + 3;
        int nw_130 = mine_w[3];
        int nn_131 = mine_n[3];
        #pragma unroll 1
        for (int t_3 = 0; t_3 < nw_130; t_3++) {
            if (alt == 1) {
                alt_wide_list[carry_w + run_w + t_3] = g_129 + t_3;
            } else {
                wide_list[carry_w + run_w + t_3] = g_129 + t_3;
            }
        }
        int narrow_row0_132 = (g_129 + nw_130) * rows;
        #pragma unroll 1
        for (int i_5 = 0; i_5 < nn_131; i_5++) {
            narrow_list[carry_n + run_n + i_5] = (narrow_row0_132 + i_5 * narrow_tile) / row_unit;
        }
        run_w = run_w + nw_130;
        run_n = run_n + nn_131;
        carry_w = carry_w + total_w;
        carry_n = carry_n + total_n;
        carry_a = carry_a + total_a;
        __syncthreads();
        pass_idx = pass_idx + 1;
    }
    if (tid_0 == 0) {
        int wide_out = carry_w;
        int alt_wide_out = 0;
        int narrow_base_out = carry_n;
        if (alt == 1) {
            wide_out = 0;
            alt_wide_out = carry_w;
            narrow_base_out = 0;
        }
        wide_count[0] = wide_out;
        {
            alt_wide_count[0] = alt_wide_out;
        }
        narrow_count[0] = carry_n;
        {
            narrow_count_base[0] = narrow_base_out;
        }
    }
}

} // extern "C"
